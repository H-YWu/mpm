#ifndef CHAINS_SVD_EIGEN_H_
#define CHAINS_SVD_EIGEN_H_

#include <Eigen/Dense>
#include <cuda_runtime.h>

namespace chains {

__device__
void svd3x3(const Eigen::Matrix3f& a, Eigen::Matrix3f& u, Eigen::Vector3f& s, Eigen::Matrix3f& v);

}   // namespace chains

#endif  // CHAINS_SVD_EIGEN_H_