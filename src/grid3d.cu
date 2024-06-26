#include "grid3d.h"

#include <limits>

namespace chains {

__host__ __device__
Grid3DSettings::Grid3DSettings(
    Eigen::Vector3d gridOrigin,
    Eigen::Vector3i gridResolution,
    double gridStride,
    double gridBoundaryFrictionCoefficient
) : origin(gridOrigin), resolution(gridResolution),
    stride(gridStride), boundary_friction_coefficient(gridBoundaryFrictionCoefficient) {
    target = origin + resolution.cast<double>()*stride;
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
void CollocatedGridData3D::updateVelocity(double deltaTimeInSeconds) {
    if (mass > std::numeric_limits<double>::epsilon()) {
        velocity /= mass;   // IMPORTANT: normalized weights for velocity here since we did not do this in transfer
        velocity_star = velocity + deltaTimeInSeconds * force / mass;
    }
}

}   // namespace chains