#include "svd_eigen.h"

#include <3x3_SVD_CUDA/svd3x3/svd3_cuda.h>

namespace chains {

__device__
void svd3x3(const Eigen::Matrix3f& a, Eigen::Matrix3f& u, Eigen::Vector3f& s, Eigen::Matrix3f& v) {
    svd(
        a(0,0), a(0,1), a(0,2), a(1,0), a(1,1), a(1,2), a(2,0), a(2,1), a(2,2),
        u(0,0), u(0,1), u(0,2), u(1,0), u(1,1), u(1,2), u(2,0), u(2,1), u(2,2),
        s(0),
        s(1),
        s(2),
        v(0,0), v(0,1), v(0,2), v(1,0), v(1,1), v(1,2), v(2,0), v(2,1), v(2,2)
    );
}

}   // namespace chains