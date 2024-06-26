#ifndef CHAINS_MPM_SOLVER_3D_H_
#define CHAINS_MPM_SOLVER_3D_H_

#include <glad/glad.h>

#include "material_point3d.h"
#include "grid3d.h"
#include "interpolation3d.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cuda_gl_interop.h>
#include <string>

namespace chains {

enum class IntegrationType {
    EXPLICIT,
    SEMI_IMPLICIT,
    FULL_IMPLICIT
};

class MPMSolver3D {
public:
    bool _enable_particles_collision;

    MPMSolver3D(
        const thrust::host_vector<MaterialPoint3D> &particles,
        Eigen::Vector3d gridOrigin,
        Eigen::Vector3i gridResolution,
        double gridStride,
        double gridBoundaryFrictionCoefficient,
        double blendCoefficient,
        InterpolationType interpolationType,
        double deltaTime
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
    void updateGridVelocities(double deltaTimeInSeconds);
    // Implicit integration
    void solveLinearSystem(double deltaTimeInSeconds);

    void gridToParticles(double deltaTimeInSeconds, double blendCoefficient);

    void advectParticles(double deltaTimeInSeconds);

    void computeGravityForces();

    void gridCollision(double deltaTimeInSeconds);

    void particlesCollision(double deltaTimeInSeconds);

    // Rendering
    //  Online: OpenGL
    //  GPU/CUDA
    void registerGLBufferWithCUDA(const GLuint buffer);
    void updateGLBufferWithCUDA();
    //  CPU
    void saveGLBuffer(const GLuint buffer);
    void updateGLBufferByCPU();
    //  Offline
    void writeToFile(std::string filePath);

protected:
    // Lagrangian
    thrust::device_vector<MaterialPoint3D> _particles;
    // Eulerian
    thrust::device_vector<CollocatedGridData3D> _grid;
    Grid3DSettings* _grid_settings;
    // Transfer: FLIP weight in PIC-FLIP blending
    double _blend_coefficient;
    // Interpolator for particle-grid transfer
    Interpolator3D* _interpolator;
    // Default time step
    double _delta_time;


    // Rendering
    //  OpenGL
    struct cudaGraphicsResource *_cuda_vbo_resource;    // WARNING: not supported on WSL
    GLuint _vbo_buffer; // work on CPU
};

}   // namespace chains

#endif  // CHAINS_MPM_SOLVER_3D_H_