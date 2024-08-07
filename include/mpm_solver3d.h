#ifndef CHAINS_MPM_SOLVER_3D_H_
#define CHAINS_MPM_SOLVER_3D_H_

#include <glad/glad.h>

#include "enums.h"
#include "material_point3d.h"
#include "grid3d.h"
#include "interpolation3d.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cuda_gl_interop.h>
#include <string>

namespace chains {


/* PIC-FLIP 3D Material Point Method Solver */

class MPMSolver3D {
public:
    bool _enable_particles_collision;

    MPMSolver3D(
        const thrust::host_vector<MaterialPoint3D> &particles,
        Eigen::Vector3f gridOrigin,
        Eigen::Vector3i gridResolution,
        float gridStride,
        float gridBoundaryFrictionCoefficient,
        float blendCoefficient,
        InterpolationType interpolationType,
        float deltaTime
    );
    ~MPMSolver3D();

    // Simulation
    void simulateOneStep();

    void switchIfEnableParticlesCollision();

    void initialize();
    
    void resetGrid();

    void particlesToGrid();
 
    void computeExternalForces();
    
    // Explicit integration
    void updateGridVelocities(float deltaTimeInSeconds);
    // Implicit integration
    void solveLinearSystem(float deltaTimeInSeconds);

    void gridToParticles(float deltaTimeInSeconds, float blendCoefficient);

    void advectParticles(float deltaTimeInSeconds);

    void computeGravityForces();

    void gridCollision(float deltaTimeInSeconds);

    void particlesCollision(float deltaTimeInSeconds);

    // Rendering
    //  Online: OpenGL
    //  GPU/CUDA
    void registerGLBufferWithCUDA(const GLuint buffer);
    void updateGLBufferWithCUDA();
    //  CPU
    void saveGLBuffer(const GLuint buffer);
    void updateGLBufferByCPU();
    //  Offline
    void writeToOpenVDB(std::string filePath);

protected:
    // Lagrangian
    thrust::device_vector<MaterialPoint3D> _particles;
    // Eulerian
    thrust::device_vector<CollocatedGridData3D> _grid;
    Grid3DSettings* _grid_settings;
    Grid3DSettings* _host_grid_settings;
    // Transfer: FLIP weight in PIC-FLIP blending
    float _blend_coefficient;
    // Interpolator for particle-grid transfer
    Interpolator3D* _interpolator;
    Interpolator3D* _host_interpolator;
    // Default time step
    float _delta_time;


    // Rendering
    //  OpenGL
    struct cudaGraphicsResource *_cuda_vbo_resource;    // WARNING: not supported on WSL
    GLuint _vbo_buffer; // work on CPU
};

}   // namespace chains

#endif  // CHAINS_MPM_SOLVER_3D_H_