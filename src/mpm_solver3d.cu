#include "mpm_solver3d.h"
#include "collision_object3d.h"
#include "create_file.h"
#include "cuda_utils.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <fstream>

namespace chains {

MPMSolver3D::MPMSolver3D(
    const thrust::host_vector<MaterialPoint3D> &particles,
    Eigen::Vector3d gridOrigin,
    Eigen::Vector3i gridResolution,
    double gridStride,
    double gridBoundaryFrictionCoefficient,
    double blendCoefficient,
    InterpolationType interpolationType
) : _blend_coefficient(blendCoefficient)
{
    // Build particles
    _particles.resize(particles.size());
    thrust::copy(particles.begin(), particles.end(), _particles.begin());

    // Build GridSettings
    Grid3DSettings* grid_settings;
    grid_settings = (Grid3DSettings*)malloc(sizeof(Grid3DSettings));
    cudaMalloc((void**) &_grid_settings, sizeof(Grid3DSettings));
    *grid_settings= Grid3DSettings(
        gridOrigin,
        gridResolution,
        gridStride,
        gridBoundaryFrictionCoefficient
    );
    cudaMemcpy(_grid_settings, grid_settings, sizeof(Grid3DSettings), cudaMemcpyHostToDevice );
    free(grid_settings);

    // Build interpolator
    Interpolator3D* interpolator;
    interpolator = (Interpolator3D*)malloc(sizeof(Interpolator3D));
    cudaMalloc((void**) &_interpolator, sizeof(Interpolator3D));
    *interpolator = Interpolator3D(interpolationType);
    cudaMemcpy(_interpolator, interpolator, sizeof(Interpolator3D), cudaMemcpyHostToDevice );
    free(interpolator);

    // Build grid
    int grid_size = gridResolution(0)*gridResolution(1)*gridResolution(2);
    _host_grid.resize(grid_size);
    for (int i = 0; i < grid_size; ++i) {
        _host_grid[i] = CollocatedGridData3D(
            Eigen::Vector3i(
                i % gridResolution(0),
                ((i / gridResolution(0)) % gridResolution(1)),
                ((i / gridResolution(0)) / gridResolution(1))
            )
        );
    }
    _grid.resize(_host_grid.size());
    thrust::copy(_host_grid.begin(), _host_grid.end(), _grid.begin());

    _enable_particles_collision = false;

    // IMPORTANT: Compute the initialize volume of each particle
    initialize();

}

MPMSolver3D::~MPMSolver3D() {
    free(_grid_settings);
    free(_interpolator);
}

void MPMSolver3D::simulateOneStep() {
    double dt = 1e-4;    // TODO: CFL

    // New grid for this step
    resetGrid();

    // PIC: Transfer mass, velocity and elastic force
    //  NOTICE: this includes computing explicit grid forces
    particlesToGrid(); 

    computeExternalForces();

    updateGridVelocities(dt);
    gridCollision(dt);
    solveLinearSystem(dt);

    // PIC: Transfer velocity
    //  NOTICE: this includes updating paticles' deformation gradient 
    gridToParticles(dt, _blend_coefficient);

    if (_enable_particles_collision)
        particlesCollision(dt);
 
    advectParticles(dt);
}

void MPMSolver3D::switchIfEnableParticlesCollision() {
    _enable_particles_collision = _enable_particles_collision==true? false : true;
}

void MPMSolver3D::initialize() {
    CollocatedGridData3D* grid_ptr = thrust::raw_pointer_cast(&_grid[0]);
    Grid3DSettings* grid_settings_ptr = _grid_settings; 
    Interpolator3D* interpolator_ptr = _interpolator; 

    auto P2GMass = [=] __device__ (MaterialPoint3D& mp) {
        Eigen::Vector3d ori = grid_settings_ptr->origin;
        Eigen::Vector3i res = grid_settings_ptr->resolution;
        double h = grid_settings_ptr->stride;
        double rng = interpolator_ptr->_range;

        Eigen::Vector3d mp_pos = mp.position;
    
        // Grid vertex range inside the interpolation kernel
        int index_l[3], index_u[3];
        for (int i = 0; i < 3; i ++) {
            index_l[i] = ceil(mp_pos(i)/h - rng);
            index_u[i] = floor(mp_pos(i)/h + rng);
        }

        for (int x = index_l[0]; x <= index_u[0]; x ++) {
            for (int y = index_l[1]; y <= index_u[1]; y ++) {
                for (int z = index_l[2]; z <= index_u[2]; z ++) {
                     if (x < 0 || x > res(0)
                      || y < 0 || y > res(1)
                      || z < 0 || z > res(2)) continue; // Ensure the vertex is inside the grid
                    // 1D index to access grid data
                    int gd_index = x + res(0)*(y + res(1)*z);
                    // Grid vertex world position
                    Eigen::Vector3d gd_pos = ori + h * Eigen::Vector3d(x, y, z);

                    double w = interpolator_ptr->weight3D(mp_pos, gd_pos, h);

                    // PIC-FLIP
                    //  mass
                    double m_ip = mp.mass * w;
                    atomicAdd(&(grid_ptr[gd_index].mass), m_ip);
                }
            }
        }
    };

    auto G2PVolume = [=] __device__ (MaterialPoint3D& mp) {
        Eigen::Vector3d ori = grid_settings_ptr->origin;
        Eigen::Vector3i res = grid_settings_ptr->resolution;
        double h = grid_settings_ptr->stride;
        double rng = interpolator_ptr->_range;

        Eigen::Vector3d mp_pos = mp.position;
    
        // Grid vertex range inside the interpolation kernel
        int index_l[3], index_u[3];
        for (int i = 0; i < 3; i ++) {
            index_l[i] = ceil(mp_pos(i)/h - rng);
            index_u[i] = floor(mp_pos(i)/h + rng);
        }

        double mp_density = 0.0;
        double inv_cell_vol = 1.0 / (h*h*h);

        for (int x = index_l[0]; x <= index_u[0]; x ++) {
            for (int y = index_l[1]; y <= index_u[1]; y ++) {
                for (int z = index_l[2]; z <= index_u[2]; z ++) {
                     if (x < 0 || x > res(0)
                      || y < 0 || y > res(1)
                      || z < 0 || z > res(2)) continue; // Ensure the vertex is inside the grid
                    // 1D index to access grid data
                    int gd_index = x + res(0)*(y + res(1)*z);
                    // Grid vertex world position
                    Eigen::Vector3d gd_pos = ori + h * Eigen::Vector3d(x, y, z);

                    double w = interpolator_ptr->weight3D(mp_pos, gd_pos, h);

                    // PIC-FLIP
                    double m_ip = mp.mass * w;
                    //  accumulate density
                    mp_density += m_ip * inv_cell_vol;
                }
            }
        }

        // Set the initial volume of particle
        mp.volume0 = mp.mass / mp_density;
    };

    // Transfer mass from particles to grid
    thrust::for_each(thrust::device, _particles.begin(), _particles.end(), P2GMass);

    cudaDeviceSynchronize();

    // Compute particle volumes and densities back
    thrust::for_each(thrust::device, _particles.begin(), _particles.end(), G2PVolume);
}

void MPMSolver3D::resetGrid() {
    thrust::for_each(
        thrust::device,
        _grid.begin(),
        _grid.end(),
        [=] __device__ (CollocatedGridData3D& gd) {
            gd.reset();
        }
    );
}

void MPMSolver3D::particlesToGrid() {
    CollocatedGridData3D* grid_ptr = thrust::raw_pointer_cast(&_grid[0]);
    Grid3DSettings* grid_settings_ptr = _grid_settings; 
    Interpolator3D* interpolator_ptr = _interpolator; 

    auto P2G = [=] __device__ (MaterialPoint3D& mp) {
        Eigen::Vector3d ori = grid_settings_ptr->origin;
        Eigen::Vector3i res = grid_settings_ptr->resolution;
        double h = grid_settings_ptr->stride;
        double rng = interpolator_ptr->_range;

        Eigen::Vector3d mp_pos = mp.position;
    
        // Grid vertex range inside the interpolation kernel
        int index_l[3], index_u[3];
        for (int i = 0; i < 3; i ++) {
            index_l[i] = ceil(mp_pos(i)/h - rng);
            index_u[i] = floor(mp_pos(i)/h + rng);
        }

        Eigen::Matrix3d vol_stress = -1.0 * mp.volumeTimesCauchyStress();

        for (int x = index_l[0]; x <= index_u[0]; x ++) {
            for (int y = index_l[1]; y <= index_u[1]; y ++) {
                for (int z = index_l[2]; z <= index_u[2]; z ++) {
                     if (x < 0 || x > res(0)
                      || y < 0 || y > res(1)
                      || z < 0 || z > res(2)) continue; // Ensure the vertex is inside the grid
                    // 1D index to access grid data
                    int gd_index = x + res(0)*(y + res(1)*z);
                    // Grid vertex world position
                    Eigen::Vector3d gd_pos = ori + h * Eigen::Vector3d(x, y, z);

                    double w = interpolator_ptr->weight3D(mp_pos, gd_pos, h);
                    Eigen::Vector3d gradw = interpolator_ptr->weightGradient3D(mp_pos, gd_pos, h);

                    // PIC-FLIP
                    double m_ip = mp.mass * w;
                    Eigen::Vector3d f_ip = vol_stress * gradw;
                    //  mass
                    atomicAdd(&(grid_ptr[gd_index].mass), m_ip);
                    //  velocity
                    //      WARNING: grid velocity is not normalized here
                    atomicAdd(&(grid_ptr[gd_index].velocity(0)), mp.velocity(0)*m_ip);
                    atomicAdd(&(grid_ptr[gd_index].velocity(1)), mp.velocity(1)*m_ip);
                    atomicAdd(&(grid_ptr[gd_index].velocity(2)), mp.velocity(2)*m_ip);
                    //  elastic force
                    atomicAdd(&(grid_ptr[gd_index].force(0)), f_ip(0));
                    atomicAdd(&(grid_ptr[gd_index].force(1)), f_ip(1));
                    atomicAdd(&(grid_ptr[gd_index].force(2)), f_ip(2));
                }
            }
        }
    };

    thrust::for_each(thrust::device, _particles.begin(), _particles.end(), P2G);
}

void MPMSolver3D::computeExternalForces() {
    computeGravityForces();
}

void MPMSolver3D::updateGridVelocities(double deltaTimeInSeconds) {
    thrust::for_each(
        thrust::device,
        _grid.begin(),
        _grid.end(),
        [=] __device__ (CollocatedGridData3D& gd) {
            gd.updateVelocity(deltaTimeInSeconds);
        }
    );
}

void MPMSolver3D::solveLinearSystem(double deltaTimeInSeconds) {
    // TODO
}

void MPMSolver3D::gridToParticles(double deltaTimeInSeconds, double blendCoefficient) {
    CollocatedGridData3D* grid_ptr = thrust::raw_pointer_cast(&_grid[0]);
    Grid3DSettings* grid_settings_ptr = _grid_settings; 
    Interpolator3D* interpolator_ptr = _interpolator; 

    auto computeVelocityAndItsGradient = [=] __device__ (MaterialPoint3D& mp) -> thrust::pair<Eigen::Vector3d, Eigen::Matrix3d> {
        Eigen::Vector3d ori = grid_settings_ptr->origin;
        Eigen::Vector3i res = grid_settings_ptr->resolution;
        double h = grid_settings_ptr->stride;
        double rng = interpolator_ptr->_range;

        Eigen::Vector3d mp_pos = mp.position;
    
        // Grid vertex range inside the interpolation kernel
        int index_l[3], index_u[3];
        for (int i = 0; i < 3; i ++) {
            index_l[i] = ceil(mp_pos(i)/h - rng);
            index_u[i] = floor(mp_pos(i)/h + rng);
        }

        Eigen::Matrix3d vel_grad(Eigen::Matrix3d::Zero());
        Eigen::Vector3d vel_pic(Eigen::Vector3d::Zero());
        Eigen::Vector3d vel_flip(mp.velocity);

        for (int x = index_l[0]; x <= index_u[0]; x ++) {
            for (int y = index_l[1]; y <= index_u[1]; y ++) {
                for (int z = index_l[2]; z <= index_u[2]; z ++) {
                     if (x < 0 || x > res(0)
                      || y < 0 || y > res(1)
                      || z < 0 || z > res(2)) continue; // Ensure the vertex is inside the grid
                    // 1D index to access grid data
                    int gd_index = x + res(0)*(y + res(1)*z);
                    // Grid vertex world position
                    Eigen::Vector3d gd_pos = ori + h * Eigen::Vector3d(x, y, z);

                    double w = interpolator_ptr->weight3D(mp_pos, gd_pos, h);
                    Eigen::Vector3d gradw = interpolator_ptr->weightGradient3D(mp_pos, gd_pos, h);

                    // PIC-FLIP
                    CollocatedGridData3D grid_data = grid_ptr[gd_index];
                    vel_pic = grid_data.velocity_star * w;
                    vel_flip = (grid_data.velocity_star-grid_data.velocity) * w;
                    vel_grad += grid_data.velocity_star * gradw.transpose();
                }
            }
        }

        Eigen::Vector3d vel = blendCoefficient*vel_flip + (1.0-blendCoefficient)*vel_pic;

        return thrust::make_pair(vel, vel_grad);
    };

    thrust::for_each(
        thrust::device,
        _particles.begin(),
        _particles.end(),
        [=] __device__ (MaterialPoint3D& mp) {
            auto vel_pair = computeVelocityAndItsGradient(mp);
            mp.updateDeformationGradient(vel_pair.second, deltaTimeInSeconds);
            mp.velocity = vel_pair.first;
        }
    );
}

void MPMSolver3D::advectParticles(double deltaTimeInSeconds) {
    thrust::for_each(
        thrust::device,
        _particles.begin(),
        _particles.end(),
        [=] __device__ (MaterialPoint3D& mp) {
            mp.updatePosition(deltaTimeInSeconds);
        }
    );
}


void MPMSolver3D::computeGravityForces() {
    auto addGravityForce = [=] __device__ (CollocatedGridData3D& gd) {
        gd.force(1) -= 9.80665; // m/s^2
    };

    thrust::for_each(thrust::device, _grid.begin(), _grid.end(), addGravityForce);
}

void MPMSolver3D::gridCollision(double deltaTimeInSeconds) {
    CollocatedGridData3D* grid_ptr = thrust::raw_pointer_cast(&_grid[0]);
    Grid3DSettings* grid_settings_ptr = _grid_settings; 

    thrust::for_each(
        thrust::device,
        _grid.begin(),
        _grid.end(),
        [=] __device__ (CollocatedGridData3D& gd) {
            Eigen::Vector3d ori = grid_settings_ptr->origin;
            Eigen::Vector3d tar = grid_settings_ptr->target;
            double h = grid_settings_ptr->stride;
            double coeff = grid_settings_ptr->boundary_friction_coefficient;
            gd.velocity_star = applyBoundaryCollision(
                // "Distorted" position of this grid point
                (ori + gd.index.cast<double>()*h) + (deltaTimeInSeconds*gd.velocity_star),
                gd.velocity_star, ori, tar, coeff
            );
        }
    );
}

void MPMSolver3D::particlesCollision(double deltaTimeInSeconds) {
    Grid3DSettings* grid_settings_ptr = _grid_settings; 
    thrust::for_each(
        thrust::device,
        _particles.begin(),
        _particles.end(),
        [=] __device__ (MaterialPoint3D& mp) {
            Eigen::Vector3d ori = grid_settings_ptr->origin;
            Eigen::Vector3d tar = grid_settings_ptr->target;
            double coeff = grid_settings_ptr->boundary_friction_coefficient;
            mp.velocity = applyBoundaryCollision(
                mp.position + deltaTimeInSeconds*mp.velocity,
                mp.velocity, ori, tar, coeff 
            );
        }
    );
}


void MPMSolver3D::registerGLBufferWithCUDA(const GLuint buffer) {
    // Enable CUDA to directly map and access the buffer
    cudaError_t ret;
    ret = cudaGraphicsGLRegisterBuffer(&vbo_resource, buffer, cudaGraphicsMapFlagsWriteDiscard);
    assert(ret == cudaSuccess);
}

void MPMSolver3D::updateGLBufferWithCUDA() {
    cudaError_t ret;
    float4 *bufptr;
    size_t size;

    ret = cudaGraphicsMapResources(1, &vbo_resource, nullptr);
    assert(ret == cudaSuccess);

    ret = cudaGraphicsResourceGetMappedPointer((void **)&bufptr, &size, vbo_resource);
    assert(ret == cudaSuccess);

    assert(bufptr != nullptr && size >= _particles.size() * sizeof(float4));
    thrust::transform(
        thrust::device,
        _particles.begin(),
        _particles.end(),
        bufptr,
        [=] __device__ (MaterialPoint3D& mp) -> float4 {
            return make_float4(5.0 * mp.position(0) - 2.5, 5.0 * mp.position(1), 5.0 * mp.position(2) - 2.5, 1.0);
        }
    );

    ret = cudaGraphicsUnmapResources(1, &vbo_resource, nullptr);
    assert(ret == cudaSuccess);
}

void MPMSolver3D::writeToFile(std::string filePath) {
}

}   // namespace chains