#ifndef CHAINS_COLLISION_OBJECT3D_H_
#define CHAINS_COLLISION_OBJECT3D_H_

#include <Eigen/Dense>
#include <cuda_runtime.h>

namespace chains {

struct LevelSetCollisionObject3D
{
public:
    Eigen::VectorXd level_set;
};

__device__
Eigen::Vector3d applyBoundaryCollision(
    const Eigen::Vector3d& position,
    const Eigen::Vector3d& velocity,
    Eigen::Vector3d grid000,
    Eigen::Vector3d grid111,
    double gridBoundaryFrictionCoefficient
);

}   // namespace chains

#endif  // CHAINS_COLLISION_OBJECT3D_H_