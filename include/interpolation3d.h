#ifndef CHAINS_INTERPOLATION3D_H_
#define CHAINS_INTERPOLATION3D_H_

#include "enums.h"

#include <Eigen/Dense>
#include <cuda_runtime.h>

namespace chains {

class Interpolator3D {
public:
    float _range;
    InterpolationType _type;

    __host__ __device__
    Interpolator3D(InterpolationType interpolationType);

    __host__ __device__
    ~Interpolator3D() { }

    __host__ __device__
    float weight3D(Eigen::Vector3f particlePosition, Eigen::Vector3f gridPosition, float stride) const;

    __host__ __device__
    const Eigen::Vector3f weightGradient3D(Eigen::Vector3f particlePosition, Eigen::Vector3f gridPosition, float stride) const;

private:

    // Set interpolation-kernel-supported range (absolute distance) 
    //  Called in constructor
    __host__ __device__
    void setRange();

    // For weight3D:

    __host__ __device__
    float linearN3D(float x) const;

    __host__ __device__
    float quadraticBSplineN3D(float x) const;

    __host__ __device__
    float cubicBSplineN3D(float x) const;

    // For weightGradient3D:

    __host__ __device__
    float linearNDot3D(float x) const;

    __host__ __device__
    float quadraticBSplineNDot3D(float x) const;

    __host__ __device__
    float cubicBSplineNDot3D(float x) const;
};

}   // namespace chains

#endif // CHAINS_INTERPOLATION3D_H_