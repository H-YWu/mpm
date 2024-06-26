#ifndef CHAINS_GRID3D_H_
#define CHAINS_GRID3D_H_

#include <Eigen/Dense>
#include <cuda_runtime.h>

namespace chains {

struct Grid3DSettings {
    Eigen::Vector3d origin, target; // target is the farthest point from origin
    Eigen::Vector3i resolution;
    double stride;
    double boundary_friction_coefficient;

    __host__ __device__
    Grid3DSettings(
        Eigen::Vector3d gridOrigin,
        Eigen::Vector3i gridResolution,
        double gridStride,
        double gridBoundaryFrictionCoefficient
    );

    __host__ __device__
    ~Grid3DSettings() { }
};

struct CollocatedGridData3D {
    Eigen::Vector3i index;
    Eigen::Vector3d force;
    Eigen::Vector3d velocity, velocity_star;
    double mass;

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
    void updateVelocity(double deltaTimeInSeconds);
};

} // namespace chains

#endif  // CHAINS_GRID3D_H_