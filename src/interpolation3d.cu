#include "interpolation3d.h"

namespace chains {

__host__ __device__
Interpolator3D::Interpolator3D(InterpolationType interpolationType)
: _type(interpolationType) {
    setRange();
}


__host__ __device__
double Interpolator3D::weight3D(Eigen::Vector3d particlePosition, Eigen::Vector3d gridPosition, double stride) const {
    double x = (particlePosition[0] - gridPosition[0]) / stride;
    double y = (particlePosition[1] - gridPosition[1]) / stride;
    double z = (particlePosition[2] - gridPosition[2]) / stride;
    switch (_type)
    {
    case InterpolationType::LINEAR:
        return linearN3D(x) * linearN3D(y) * linearN3D(z);
        break;
    
    case InterpolationType::QUADRATIC_BSPLINE:
        return quadraticBSplineN3D(x) * quadraticBSplineN3D(y) * quadraticBSplineN3D(z);
        break;
    
    case InterpolationType::CUBIC_BSPLINE:
    default:
        return cubicBSplineN3D(x) * cubicBSplineN3D(y) * cubicBSplineN3D(z);
        break;
    }
}

__host__ __device__
const Eigen::Vector3d Interpolator3D::weightGradient3D(Eigen::Vector3d particlePosition, Eigen::Vector3d gridPosition, double stride) const {
    double x = (particlePosition[0] - gridPosition[0]) / stride;
    double y = (particlePosition[1] - gridPosition[1]) / stride;
    double z = (particlePosition[2] - gridPosition[2]) / stride;
    double grad_Nx = 0.0;
    double grad_Ny = 0.0;
    double grad_Nz = 0.0;
     switch (_type)
    {
    case InterpolationType::LINEAR:
        grad_Nx = (linearNDot3D(x) * linearN3D(y) * linearN3D(z)) / stride;
        grad_Ny = (linearN3D(x) * linearNDot3D(y) * linearN3D(z)) / stride;
        grad_Nz = (linearN3D(x) * linearN3D(y) * linearNDot3D(z)) / stride;
        break;
    
    case InterpolationType::QUADRATIC_BSPLINE:
        grad_Nx = (quadraticBSplineNDot3D(x) * quadraticBSplineN3D(y) * quadraticBSplineN3D(z)) / stride;
        grad_Ny = (quadraticBSplineN3D(x) * quadraticBSplineNDot3D(y) * quadraticBSplineN3D(z)) / stride;
        grad_Nz = (quadraticBSplineN3D(x) * quadraticBSplineN3D(y) * quadraticBSplineNDot3D(z)) / stride;
        break;
    
    case InterpolationType::CUBIC_BSPLINE:
    default:
        grad_Nx = (cubicBSplineNDot3D(x) * cubicBSplineN3D(y) * cubicBSplineN3D(z)) / stride;
        grad_Ny = (cubicBSplineN3D(x) * cubicBSplineNDot3D(y) * cubicBSplineN3D(z)) / stride;
        grad_Nz = (cubicBSplineN3D(x) * cubicBSplineN3D(y) * cubicBSplineNDot3D(z)) / stride;
        break;
    }
    return Eigen::Vector3d(grad_Nx, grad_Ny, grad_Nz);

}

__host__ __device__
void Interpolator3D::setRange() {
    switch (_type)
    {
    case InterpolationType::LINEAR:
        _range = 1.0;
        break;
    
    case InterpolationType::QUADRATIC_BSPLINE:
        _range = 1.5;
        break;
    
    case InterpolationType::CUBIC_BSPLINE:
    default:
        _range = 2.0;
        break;
    }
}

__host__ __device__
double Interpolator3D::linearN3D(double x) const {
    double xabs = std::abs(x);
    if (xabs < 1.0) {   // Absolute value is always >= 0.0
        return 1.0 - xabs;
    } else return 0.0;
}

__host__ __device__
double Interpolator3D::linearNDot3D(double x) const {
    double xabs = std::abs(x);
    if (xabs < 1.0 && xabs > 0.0) {   // Absolute value is always >= 0.0 
        // Avoid undefined derivative
        return -x/xabs; 
    } else return 0.0;
}

__host__ __device__
double Interpolator3D::quadraticBSplineN3D(double x) const {
    double xabs = std::abs(x);
    if (xabs < 0.5) {   // Absolute value is always >= 0.0
        return 0.75 - xabs*xabs;
    } else if (xabs >= 0.5 && xabs < 1.5) {
        return 0.5*pow(1.5-xabs, 2);
    } else return 0.0;
}

__host__ __device__
double Interpolator3D::quadraticBSplineNDot3D(double x) const {
    double xabs = std::abs(x);
    if (xabs < 0.5) {   // Absolute value is always >= 0.0 
        return - 2.0*x; 
    } else if (xabs >= 0.5 && xabs < 1.5) {
        return -(1.5*x)/xabs + x;
    } else return 0.0;
}

__host__ __device__
double Interpolator3D::cubicBSplineN3D(double x) const {
    double xabs = std::abs(x);
    if (xabs < 1.0) {   // Absolute value is always >= 0.0
        return 0.5*pow(xabs, 3) - xabs*xabs + 2.0/3.0;
    } else if (xabs >= 1.0 && xabs < 2.0) {
        return pow(2.0-xabs, 3) / .6;
    } else return 0.0;
}

__host__ __device__
double Interpolator3D::cubicBSplineNDot3D(double x) const {
    double xabs = std::abs(x);
    if (xabs < 1.0) {   // Absolute value is always >= 0.0 
        return 1.5*x*xabs - 2.0*x; 
    } else if (xabs >= 1.0 && xabs < 2.0) {
        return -(2.0*x)/xabs + 2.0*x - (x*xabs)/2.0;
    } else return 0.0;
}

}   // namespace chains