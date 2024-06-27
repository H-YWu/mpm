#ifndef CHAINS_GRID3D_H_
#define CHAINS_GRID3D_H_

#include <Eigen/Dense>
#include <cuda_runtime.h>

namespace chains {

struct Grid3DSettings {
    Eigen::Vector3f origin, target; // target is the farthest point from origin
    Eigen::Vector3i resolution;
    float stride;
    float boundary_friction_coefficient;

    __host__ __device__
    Grid3DSettings(
        Eigen::Vector3f gridOrigin,
        Eigen::Vector3i gridResolution,
        float gridStride,
        float gridBoundaryFrictionCoefficient
    );

    __host__ __device__
    ~Grid3DSettings() { }
};

struct CollocatedGridData3D {
    Eigen::Vector3i index;
    Eigen::Vector3f force;
    Eigen::Vector3f velocity, velocity_star;
    float mass;

    __host__ __device__
    CollocatedGridData3D() { }
    __host__ __device__
    CollocatedGridData3D(Eigen::Vector3i idx);

    __host__ __device__
    ~CollocatedGridData3D() { }

    __host__ __device__
    void reset();

    // Explicit integration
    __host__ __device__
    void updateVelocity(float deltaTimeInSeconds);
};

} // namespace chains

#endif  // CHAINS_GRID3D_H_