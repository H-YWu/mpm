#include "collision_object3d.h"

#include <limits>

namespace chains {

__host__ __device__
Eigen::Vector3f applyBoundaryCollision(
    const Eigen::Vector3f& position,
    const Eigen::Vector3f& velocity,
    Eigen::Vector3f grid000,
    Eigen::Vector3f grid111,
    float gridBoundaryFrictionCoefficient
) {
    float vn;
    Eigen::Vector3f vel_t, normal,
        // Grid boundary is still,
        //  so the relative velocity is the same as the velocity
        updated_velocity(velocity),
        relative_velocity(velocity);

    bool is_collided = false;
    normal.setZero();

    // Boundary of grid
    for (int i = 0; i < 3; i ++) {
        if (position(i) <= grid000(i) + std::numeric_limits<float>::epsilon()) {
            is_collided = true;
            normal(i) = 1.0;
        }
        if (position(i) >= grid111(i) - std::numeric_limits<float>::epsilon()) {
            is_collided = true;
            normal(i) = -1.0;
        }
    }

    if (is_collided) {
        // Grid boundary is still,
        //  so the relative velocity is the same as the velocity
        vn = relative_velocity.dot(normal);
        if (vn >= std::numeric_limits<float>::epsilon()) return updated_velocity; // Separating: no collision
        vel_t = relative_velocity - vn*normal;
        if (vel_t.norm() <= -gridBoundaryFrictionCoefficient*vn + std::numeric_limits<float>::epsilon()) {
            // If a sticking impulse is required
            relative_velocity.setZero();
        } else {
            relative_velocity = vel_t + gridBoundaryFrictionCoefficient * vn * vel_t.normalized();
        }
        //  the relative velocity is the same as the velocity
        updated_velocity = relative_velocity;
    }

    return updated_velocity;
}

}   // namespace chains