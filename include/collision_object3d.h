#ifndef CHAINS_COLLISION_OBJECT3D_H_
#define CHAINS_COLLISION_OBJECT3D_H_

#include <Eigen/Dense>
#include <cuda_runtime.h>

namespace chains {

struct LevelSetCollisionObject3D
{
    Eigen::VectorXd level_set;
};

__host__ __device__
Eigen::Vector3f applyBoundaryCollision(
    const Eigen::Vector3f& position,
    const Eigen::Vector3f& velocity,
    Eigen::Vector3f grid000,
    Eigen::Vector3f grid111,
    float gridBoundaryFrictionCoefficient
);

}   // namespace chains

#endif  // CHAINS_COLLISION_OBJECT3D_H_