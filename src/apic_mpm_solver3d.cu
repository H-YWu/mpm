#include "apic_mpm_solver3d.h"
#include "collision_object3d.h"
#include "create_file.h"
#include "cuda_utils.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/tabulate.h>
#include <iostream>
#include <limits>
#include <openvdb/openvdb.h>
#include <openvdb/io/Stream.h>

namespace chains {

struct InstantiateCollocatedGridData3DWithIndex {
    Eigen::Vector3i resolution;

    InstantiateCollocatedGridData3DWithIndex(Eigen::Vector3i gridResolution) : resolution(gridResolution) {}

    __host__ __device__
    CollocatedGridData3D operator()(const int& idx) {
        return CollocatedGridData3D(
            Eigen::Vector3i(
                idx % resolution(0),
                ((idx / resolution(0)) % resolution(1)),
                ((idx / resolution(0)) / resolution(1))
            )
        );
    }
};

APICMPMSolver3D::APICMPMSolver3D(
    const thrust::host_vector<APICMaterialPoint3D> &particles,
    Eigen::Vector3f gridOrigin,
    Eigen::Vector3i gridResolution,
    float gridStride,
    float gridBoundaryFrictionCoefficient,
    InterpolationType interpolationType,
    float deltaTime
) : _delta_time(deltaTime)
{
    std::cout << "[INFO] copying particles to device" << std::endl;
    // Build particles
    _particles.resize(particles.size());
    thrust::copy(particles.begin(), particles.end(), _particles.begin());
    CUDA_CHECK_LAST_ERROR();

    // Build GridSettings on host and device
    _host_grid_settings = (Grid3DSettings*)malloc(sizeof(Grid3DSettings));
    *_host_grid_settings= Grid3DSettings(
        gridOrigin,
        gridResolution,
        gridStride,
        gridBoundaryFrictionCoefficient
    );
    //  Copy to device
    CUDA_CHECK(cudaMalloc((void**) &_grid_settings, sizeof(Grid3DSettings)));
    CUDA_CHECK(cudaMemcpy(_grid_settings, _host_grid_settings, sizeof(Grid3DSettings), cudaMemcpyHostToDevice));

    // Build interpolator on host and device
    _host_interpolator = (Interpolator3D*)malloc(sizeof(Interpolator3D));
    *_host_interpolator = Interpolator3D(interpolationType);
    //  Copy to device
    CUDA_CHECK(cudaMalloc((void**) &_interpolator, sizeof(Interpolator3D)));
    CUDA_CHECK(cudaMemcpy(_interpolator, _host_interpolator, sizeof(Interpolator3D), cudaMemcpyHostToDevice));

    std::cout << "[INFO] building grid of "
              << gridResolution(0) << " x " << gridResolution(1) << " x " << gridResolution(2)
              << " on device" << std::endl;
    // Build grid
    int grid_size = gridResolution(0)*gridResolution(1)*gridResolution(2);
    _grid.resize(grid_size);
    thrust::tabulate(
        thrust::device,
        _grid.begin(),
        _grid.end(),
        InstantiateCollocatedGridData3DWithIndex(gridResolution)
    );
    CUDA_CHECK_LAST_ERROR();

    _enable_particles_collision = false;

    std::cout << "[INFO] initializing particle initial volume" << std::endl;
    // IMPORTANT: Compute the initialize volume of each particle
    initialize();
}

APICMPMSolver3D::~APICMPMSolver3D() {
    free(_host_grid_settings);
    free(_host_interpolator);
    CUDA_CHECK(cudaFree(_grid_settings));
    CUDA_CHECK(cudaFree(_interpolator));
}

void APICMPMSolver3D::simulateOneStep() {
    float dt = _delta_time;    // TODO: CFL

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
    gridToParticles(dt);

    if (_enable_particles_collision) {
        particlesCollision(dt);
    }
 
    advectParticles(dt);
}

void APICMPMSolver3D::switchIfEnableParticlesCollision() {
    _enable_particles_collision = _enable_particles_collision==true? false : true;
}

void APICMPMSolver3D::initialize() {
    CollocatedGridData3D* grid_ptr = thrust::raw_pointer_cast(&_grid[0]);
    Grid3DSettings* grid_settings_ptr = _grid_settings; 
    Interpolator3D* interpolator_ptr = _interpolator; 

    auto P2GMass = [=] __device__ (APICMaterialPoint3D& mp) {
        Eigen::Vector3f ori = grid_settings_ptr->origin;
        Eigen::Vector3i res = grid_settings_ptr->resolution;
        float h = grid_settings_ptr->stride;
        float rng = interpolator_ptr->_range;
        auto interp_type = interpolator_ptr->_type;

        Eigen::Vector3f mp_pos = mp.position;
    
        // Grid vertex range inside the interpolation kernel
        int index_l[3], index_u[3];
        for (int i = 0; i < 3; i ++) {
            index_l[i] = ceil(mp_pos(i)/h - rng);
            index_u[i] = floor(mp_pos(i)/h + rng);
        }

        // Precompute Dinv
        float k = 0.0;
        if (interp_type == InterpolationType::CUBIC_BSPLINE) k = 1/3.0; 
        if (interp_type == InterpolationType::QUADRATIC_BSPLINE) k = 1/4.0; 
        // TODO: linear case
        mp.D = k * h*h * Eigen::Matrix3f::Identity();
        mp.Dinv = mp.D.inverse();

        for (int x = index_l[0]; x <= index_u[0]; x ++) {
            for (int y = index_l[1]; y <= index_u[1]; y ++) {
                for (int z = index_l[2]; z <= index_u[2]; z ++) {
                     if (x < 0 || x >= res(0)
                      || y < 0 || y >= res(1)
                      || z < 0 || z >= res(2)) continue; // Ensure the vertex is inside the grid
                    // 1D index to access grid data
                    int gd_index = x + res(0)*(y + res(1)*z);
                    // Grid vertex world position
                    Eigen::Vector3f gd_pos = ori + h * Eigen::Vector3f(x, y, z);

                    float w = interpolator_ptr->weight3D(mp_pos, gd_pos, h);

                    // APIC 
                    //  mass
                    float m_ip = mp.mass * w;
                    atomicAdd(&(grid_ptr[gd_index].mass), m_ip);
                }
            }
        }
    };

    auto G2PVolume = [=] __device__ (APICMaterialPoint3D& mp) {
        Eigen::Vector3f ori = grid_settings_ptr->origin;
        Eigen::Vector3i res = grid_settings_ptr->resolution;
        float h = grid_settings_ptr->stride;
        float rng = interpolator_ptr->_range;

        Eigen::Vector3f mp_pos = mp.position;
    
        // Grid vertex range inside the interpolation kernel
        int index_l[3], index_u[3];
        for (int i = 0; i < 3; i ++) {
            index_l[i] = ceil(mp_pos(i)/h - rng);
            index_u[i] = floor(mp_pos(i)/h + rng);
        }

        float mp_density = 0.0;
        float inv_cell_vol = 1.0 / (h*h*h);

        for (int x = index_l[0]; x <= index_u[0]; x ++) {
            for (int y = index_l[1]; y <= index_u[1]; y ++) {
                for (int z = index_l[2]; z <= index_u[2]; z ++) {
                     if (x < 0 || x >= res(0)
                      || y < 0 || y >= res(1)
                      || z < 0 || z >= res(2)) continue; // Ensure the vertex is inside the grid
                    // 1D index to access grid data
                    int gd_index = x + res(0)*(y + res(1)*z);
                    // Grid vertex world position
                    Eigen::Vector3f gd_pos = ori + h * Eigen::Vector3f(x, y, z);

                    float w = interpolator_ptr->weight3D(mp_pos, gd_pos, h);

                    // APIC 
                    float m_ip = grid_ptr[gd_index].mass * w;
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
    CUDA_CHECK_LAST_ERROR();

    // Compute particle volumes and densities back
    thrust::for_each(thrust::device, _particles.begin(), _particles.end(), G2PVolume);
    CUDA_CHECK_LAST_ERROR();
}

void APICMPMSolver3D::resetGrid() {
    thrust::for_each(
        thrust::device,
        _grid.begin(),
        _grid.end(),
        [=] __device__ (CollocatedGridData3D& gd) {
            gd.reset();
        }
    );
    CUDA_CHECK_LAST_ERROR();
}

void APICMPMSolver3D::particlesToGrid() {
    CollocatedGridData3D* grid_ptr = thrust::raw_pointer_cast(&_grid[0]);
    Grid3DSettings* grid_settings_ptr = _grid_settings; 
    Interpolator3D* interpolator_ptr = _interpolator; 

    auto P2G = [=] __device__ (APICMaterialPoint3D& mp) {
        Eigen::Vector3f ori = grid_settings_ptr->origin;
        Eigen::Vector3i res = grid_settings_ptr->resolution;
        float h = grid_settings_ptr->stride;
        float rng = interpolator_ptr->_range;

        Eigen::Vector3f mp_pos = mp.position;
    
        // Grid vertex range inside the interpolation kernel
        int index_l[3], index_u[3];
        for (int i = 0; i < 3; i ++) {
            index_l[i] = ceil(mp_pos(i)/h - rng);
            index_u[i] = floor(mp_pos(i)/h + rng);
        }

        Eigen::Matrix3f C = mp.B * mp.Dinv;
        Eigen::Matrix3f vol_stress = -mp.volumeTimesCauchyStress();

        for (int x = index_l[0]; x <= index_u[0]; x ++) {
            for (int y = index_l[1]; y <= index_u[1]; y ++) {
                for (int z = index_l[2]; z <= index_u[2]; z ++) {
                     if (x < 0 || x >= res(0)
                      || y < 0 || y >= res(1)
                      || z < 0 || z >= res(2)) continue; // Ensure the vertex is inside the grid
                    // 1D index to access grid data
                    int gd_index = x + res(0)*(y + res(1)*z);
                    // Grid vertex world position
                    Eigen::Vector3f gd_pos = ori + h * Eigen::Vector3f(x, y, z);

                    float w = interpolator_ptr->weight3D(mp_pos, gd_pos, h);
                    Eigen::Vector3f gradw = interpolator_ptr->weightGradient3D(mp_pos, gd_pos, h);

                    // APIC
                    float m_ip = mp.mass * w;
                    Eigen::Vector3f f_ip = vol_stress * gradw;
                    //  mass
                    atomicAdd(&(grid_ptr[gd_index].mass), m_ip);
                    //  velocity
                    //      WARNING: grid velocity is not normalized here
                    Eigen::Vector3f v_ip = m_ip * (mp.velocity + C*(gd_pos-mp_pos));
                    atomicAdd(&(grid_ptr[gd_index].velocity(0)), v_ip(0));
                    atomicAdd(&(grid_ptr[gd_index].velocity(1)), v_ip(1));
                    atomicAdd(&(grid_ptr[gd_index].velocity(2)), v_ip(2));
                    //  elastic force
                    atomicAdd(&(grid_ptr[gd_index].force(0)), f_ip(0));
                    atomicAdd(&(grid_ptr[gd_index].force(1)), f_ip(1));
                    atomicAdd(&(grid_ptr[gd_index].force(2)), f_ip(2));
                }
            }
        }
    };

    thrust::for_each(thrust::device, _particles.begin(), _particles.end(), P2G);
    CUDA_CHECK_LAST_ERROR();
}

void APICMPMSolver3D::computeExternalForces() {
    computeGravityForces();
}

void APICMPMSolver3D::updateGridVelocities(float deltaTimeInSeconds) {
    thrust::for_each(
        thrust::device,
        _grid.begin(),
        _grid.end(),
        [=] __device__ (CollocatedGridData3D& gd) {
            gd.updateVelocity(deltaTimeInSeconds);
        }
    );
    CUDA_CHECK_LAST_ERROR();
}

void APICMPMSolver3D::solveLinearSystem(float deltaTimeInSeconds) {
    // TODO
}

void APICMPMSolver3D::gridToParticles(float deltaTimeInSeconds) {
    CollocatedGridData3D* grid_ptr = thrust::raw_pointer_cast(&_grid[0]);
    Grid3DSettings* grid_settings_ptr = _grid_settings; 
    Interpolator3D* interpolator_ptr = _interpolator; 

    auto computeVelocityAndItsGradient = [=] __device__ (APICMaterialPoint3D& mp) -> thrust::pair<Eigen::Vector3f, Eigen::Matrix3f> {
        Eigen::Vector3f ori = grid_settings_ptr->origin;
        Eigen::Vector3i res = grid_settings_ptr->resolution;
        float h = grid_settings_ptr->stride;
        float rng = interpolator_ptr->_range;

        Eigen::Vector3f mp_pos = mp.position;
    
        // Grid vertex range inside the interpolation kernel
        int index_l[3], index_u[3];
        for (int i = 0; i < 3; i ++) {
            index_l[i] = ceil(mp_pos(i)/h - rng);
            index_u[i] = floor(mp_pos(i)/h + rng);
        }

        Eigen::Matrix3f vel_grad(Eigen::Matrix3f::Zero());
        Eigen::Vector3f vel_apic(Eigen::Vector3f::Zero());
        mp.B.setZero();

        for (int x = index_l[0]; x <= index_u[0]; x ++) {
            for (int y = index_l[1]; y <= index_u[1]; y ++) {
                for (int z = index_l[2]; z <= index_u[2]; z ++) {
                     if (x < 0 || x >= res(0)
                      || y < 0 || y >= res(1)
                      || z < 0 || z >= res(2)) continue; // Ensure the vertex is inside the grid
                    // 1D index to access grid data
                    int gd_index = x + res(0)*(y + res(1)*z);
                    // Grid vertex world position
                    Eigen::Vector3f gd_pos = ori + h * Eigen::Vector3f(x, y, z);

                    float w = interpolator_ptr->weight3D(mp_pos, gd_pos, h);
                    Eigen::Vector3f gradw = interpolator_ptr->weightGradient3D(mp_pos, gd_pos, h);

                    // APIC 
                    CollocatedGridData3D grid_data = grid_ptr[gd_index];
                    Eigen::Vector3f v_ip = grid_data.velocity_star * w;
                    vel_apic += v_ip;
                    vel_grad += grid_data.velocity_star * gradw.transpose();
                    mp.B += v_ip * (gd_pos-mp_pos).transpose();
                }
            }
        }

        return thrust::make_pair(vel_apic, vel_grad);
    };

    thrust::for_each(
        thrust::device,
        _particles.begin(),
        _particles.end(),
        [=] __device__ (APICMaterialPoint3D& mp) {
            auto vel_pair = computeVelocityAndItsGradient(mp);
            mp.updateDeformationGradient(vel_pair.second, deltaTimeInSeconds);
            mp.velocity = vel_pair.first;
        }
    );
    CUDA_CHECK_LAST_ERROR();
}

void APICMPMSolver3D::advectParticles(float deltaTimeInSeconds) {
    thrust::for_each(
        thrust::device,
        _particles.begin(),
        _particles.end(),
        [=] __device__ (APICMaterialPoint3D& mp) {
            mp.updatePosition(deltaTimeInSeconds);
        }
    );
    CUDA_CHECK_LAST_ERROR();
}


void APICMPMSolver3D::computeGravityForces() {
    auto addGravityForce = [=] __device__ (CollocatedGridData3D& gd) {
        if (gd.mass > std::numeric_limits<float>::epsilon()) {
            gd.force(1) -= 9.80665 * gd.mass; // m/s^2
        }
    };

    thrust::for_each(thrust::device, _grid.begin(), _grid.end(), addGravityForce);
    CUDA_CHECK_LAST_ERROR();
}

void APICMPMSolver3D::gridCollision(float deltaTimeInSeconds) {
    CollocatedGridData3D* grid_ptr = thrust::raw_pointer_cast(&_grid[0]);
    Grid3DSettings* grid_settings_ptr = _grid_settings; 

    thrust::for_each(
        thrust::device,
        _grid.begin(),
        _grid.end(),
        [=] __device__ (CollocatedGridData3D& gd) {
            Eigen::Vector3f ori = grid_settings_ptr->origin;
            Eigen::Vector3f tar = grid_settings_ptr->target;
            float h = grid_settings_ptr->stride;
            float coeff = grid_settings_ptr->boundary_friction_coefficient;
            gd.velocity_star = applyBoundaryCollision(
                // "Distorted" position of this grid point
                (ori + h*gd.index.cast<float>()) + (deltaTimeInSeconds*gd.velocity_star),
                gd.velocity_star, ori, tar, coeff
            );
        }
    );
    CUDA_CHECK_LAST_ERROR();
}

void APICMPMSolver3D::particlesCollision(float deltaTimeInSeconds) {
    Grid3DSettings* grid_settings_ptr = _grid_settings; 
    thrust::for_each(
        thrust::device,
        _particles.begin(),
        _particles.end(),
        [=] __device__ (APICMaterialPoint3D& mp) {
            Eigen::Vector3f ori = grid_settings_ptr->origin;
            Eigen::Vector3f tar = grid_settings_ptr->target;
            float coeff = grid_settings_ptr->boundary_friction_coefficient;
            mp.velocity = applyBoundaryCollision(
                mp.position + deltaTimeInSeconds*mp.velocity,
                mp.velocity, ori, tar, coeff
            );
        }
    );
    CUDA_CHECK_LAST_ERROR();
}

void APICMPMSolver3D::registerGLBufferWithCUDA(const GLuint buffer) {
    // Enable CUDA to directly map and access the buffer
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&_cuda_vbo_resource, buffer, cudaGraphicsMapFlagsWriteDiscard));
}

void APICMPMSolver3D::saveGLBuffer(const GLuint buffer) {
    // Storing the OpenGL buffer ID for future use
    _vbo_buffer = buffer;
}

void APICMPMSolver3D::updateGLBufferWithCUDA() {
    float4* buf_ptr;
    size_t buf_size;

    CUDA_CHECK(cudaGraphicsMapResources(1, &_cuda_vbo_resource, nullptr));

    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void **)&buf_ptr, &buf_size, _cuda_vbo_resource));

    assert(buf_ptr != nullptr && buf_size >= _particles.size() * sizeof(float4));
    thrust::transform(
        thrust::device,
        _particles.begin(),
        _particles.end(),
        buf_ptr,
        [=] __device__ (APICMaterialPoint3D& mp) -> float4 {
            return make_float4(
                mp.position(0),
                mp.position(1),
                mp.position(2), 
                1.0
            );
        }
    );

    CUDA_CHECK(cudaGraphicsUnmapResources(1, &_cuda_vbo_resource, nullptr));
}

void APICMPMSolver3D::updateGLBufferByCPU() {
    // Map OpenGL buffer for writing
    glBindBuffer(GL_ARRAY_BUFFER, _vbo_buffer);
    float4 *buf_ptr = (float4*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
    GLint buf_size;
    glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &buf_size);
    
    assert(buf_ptr != nullptr && buf_size >= _particles.size() * sizeof(float4));
    // Copy data from _particles to host particles 
    std::vector<APICMaterialPoint3D> h_particles(_particles.size());
    thrust::copy(_particles.begin(), _particles.end(), h_particles.begin());
    // Transform and copy data from h_particles to buf_ptr (OpenGL buffer)
    for (size_t i = 0; i < h_particles.size(); i ++) {
        const APICMaterialPoint3D& mp = h_particles[i];
        buf_ptr[i] = make_float4(
            mp.position(0),
            mp.position(1),
            mp.position(2),
            1.0f
        );
    }

    // Unmap OpenGL buffer
    glUnmapBuffer(GL_ARRAY_BUFFER);
}

void APICMPMSolver3D::writeToOpenVDB(std::string filePath) {
    // Copy particles to host.
    std::vector<APICMaterialPoint3D> h_particles(_particles.size());
    thrust::copy(_particles.begin(), _particles.end(), h_particles.begin());

    // Initialize the OpenVDB library.  This must be called at least
    // once per program and may safely be called multiple times.
    openvdb::initialize();

    // Create an empty floating-point grid with background value 0.
    openvdb::FloatGrid::Ptr grid = openvdb::FloatGrid::create();
    grid->setName("density");
    // Get an accessor for coordinate-based access to voxels.
    openvdb::FloatGrid::Accessor accessor = grid->getAccessor();

    // Build the density grid.
    // Rasterize particles to grid.
    for (auto &mp : h_particles) {
        Eigen::Vector3f mp_pos = mp.position;

        Eigen::Vector3f ori = _host_grid_settings->origin;
        Eigen::Vector3i res = _host_grid_settings->resolution;
        float h = _host_grid_settings->stride;
        float rng = _host_interpolator->_range;

        float cell_vol_inv = 1.0 / (h*h*h);

        // Grid vertex range inside the interpolation kernel.
        int index_l[3], index_u[3];
        for (int i = 0; i < 3; i ++) {
            index_l[i] = ceil(mp_pos(i)/h - rng);
            index_u[i] = floor(mp_pos(i)/h + rng);
        }

        for (int x = index_l[0]; x <= index_u[0]; x ++) {
            for (int y = index_l[1]; y <= index_u[1]; y ++) {
                for (int z = index_l[2]; z <= index_u[2]; z ++) {
                     if (x < 0 || x >= res(0)
                      || y < 0 || y >= res(1)
                      || z < 0 || z >= res(2)) continue; // Ensure that the vertex is inside the grid.
                    // 1D index to access grid data
                    int gd_index = x + res(0)*(y + res(1)*z);
                    // Grid vertex world position
                    Eigen::Vector3f gd_pos = ori + h * Eigen::Vector3f(x, y, z);

                    // APIC 
                    float w = _host_interpolator->weight3D(mp_pos, gd_pos, h);
                    // Add density
                    openvdb::Coord coord(x, y, z);
                    float currentDensity = accessor.getValue(coord);
                    accessor.setValue(coord, currentDensity + mp.mass*w*cell_vol_inv);
                }
            }
        }
    }

    // Write grid to file.
    openvdb::io::File file(filePath);
    openvdb::GridPtrVec grids;
    grids.push_back(grid);
    file.write(grids);
    file.close();
}

}   // namespace chains