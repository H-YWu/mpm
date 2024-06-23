#include "collision_object3d.h"

namespace chains {

__device__
Eigen::Vector3d applyBoundaryCollision(
    const Eigen::Vector3d& position,
    const Eigen::Vector3d& velocity,
    Eigen::Vector3d grid000,
    Eigen::Vector3d grid111,
    double gridBoundaryFrictionCoefficient
) {
    double vn;
    Eigen::Vector3d vt, normal,
        updated_velocity(velocity),
        relative_velocity(velocity);

    bool is_collided;

    // Boundary of grid
    for (int i = 0; i < 3; i ++) {
        is_collided = false;
        normal.setZero();
        if (position(i) <= grid000(i)) {
            is_collided = true;
            normal(i) = 1.0;
        }
        if (position(i) >= grid111(i)) {
            is_collided = true;
            normal(i) = -1.0;
        }
        if (is_collided) {
            // Grid boundary is still,
            //  so the relative velocity is the same as the velocity
            vn = relative_velocity.dot(normal);
            if (vn > 0.0) continue; // Separating: no collision
            vt = relative_velocity - vn*normal;
            if (vt.norm() <= -gridBoundaryFrictionCoefficient*vn) {
                // If a sticking impulse is required
                relative_velocity.setZero();
                updated_velocity = relative_velocity;
                break;
            } else {
                relative_velocity = vt + gridBoundaryFrictionCoefficient * vn * vt.normalized();
                updated_velocity = relative_velocity;
            }
        }
    }

    return updated_velocity;
}

}   // namespace chains