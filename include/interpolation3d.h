#ifndef CHAINS_INTERPOLATION3D_H_
#define CHAINS_INTERPOLATION3D_H_

#include <Eigen/Dense>
#include <cuda_runtime.h>

namespace chains {

enum class InterpolationType {
    LINEAR,
    QUADRATIC_BSPLINE,
    CUBIC_BSPLINE
};

class Interpolator3D {
public:
    double _range;

    __host__ __device__
    Interpolator3D(InterpolationType interpolationType);

    __host__ __device__
    ~Interpolator3D() { }

    __host__ __device__
    double weight3D(Eigen::Vector3d particlePosition, Eigen::Vector3d gridPosition, double stride) const;

    __host__ __device__
    const Eigen::Vector3d weightGradient3D(Eigen::Vector3d particlePosition, Eigen::Vector3d gridPosition, double stride) const;

private:
    InterpolationType _type;

    // Set interpolation-kernel-supported range (absolute distance) 
    //  Called in constructor
    __host__ __device__
    void setRange();

    // For weight3D:

    __host__ __device__
    double linearN3D(double x) const;

    __host__ __device__
    double quadraticBSplineN3D(double x) const;

    __host__ __device__
    double cubicBSplineN3D(double x) const;

    // For weightGradient3D:

    __host__ __device__
    double linearNDot3D(double x) const;

    __host__ __device__
    double quadraticBSplineNDot3D(double x) const;

    __host__ __device__
    double cubicBSplineNDot3D(double x) const;
};

}   // namespace chains

#endif // CHAINS_INTERPOLATION3D_H_