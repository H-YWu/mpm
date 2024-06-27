#include "grid3d.h"

#include <limits>

namespace chains {

__host__ __device__
Grid3DSettings::Grid3DSettings(
    Eigen::Vector3f gridOrigin,
    Eigen::Vector3i gridResolution,
    float gridStride,
    float gridBoundaryFrictionCoefficient
) : origin(gridOrigin), resolution(gridResolution),
    stride(gridStride), boundary_friction_coefficient(gridBoundaryFrictionCoefficient) {
    target = origin + stride*resolution.cast<float>();
}


__host__ __device__
CollocatedGridData3D::CollocatedGridData3D(Eigen::Vector3i idx)
: index(idx),
  force(0.0, 0.0, 0.0),
  velocity(0.0, 0.0, 0.0), velocity_star(0.0, 0.0, 0.0),
  mass(0.0) {
}

__host__ __device__
void CollocatedGridData3D::reset() {
    mass = 0.0;
    force.setZero();
    velocity.setZero();
    velocity_star.setZero();
}

__host__ __device__
void CollocatedGridData3D::updateVelocity(float deltaTimeInSeconds) {
    if (mass > std::numeric_limits<float>::epsilon()) {
        velocity /= mass;   // IMPORTANT: normalized weights for velocity here since we did not do this in transfer
        velocity_star = velocity + deltaTimeInSeconds * force / mass;
    }
}

}   // namespace chains