#ifndef CHAINS_MATERIAL_POINT3D_H_
#define CHAINS_MATERIAL_POINT3D_H_

#include <Eigen/Dense>
#include <thrust/pair.h>
#include <cuda_runtime.h>

namespace chains {

enum class ConstitutiveModel {
    FIXED_COROTATED,
    NEO_HOOKEAN
}; 

class MaterialPoint3D {
public:
    double volume0, mass;
    Eigen::Vector3d position, velocity;

    __host__ __device__
    MaterialPoint3D() { }

    __host__ __device__
    MaterialPoint3D(
        Eigen::Vector3d particlePosition,
        Eigen::Vector3d particleVelocity,
        double particleMass,
        double hardeningCoefficient,
        double YoungModulus,
        double PoissonRatio,
        double criticalCompression,
        double criticleStretch,
        ConstitutiveModel constitutiveModel
    );

    __host__ __device__
    ~MaterialPoint3D() { }

    __host__ __device__
    const Eigen::Matrix3d volumeTimesCauchyStress() const;

    __host__ __device__
    void updatePosition(double deltaTimeInSeconds);

    __host__ __device__
    void updateDeformationGradient(Eigen::Matrix3d velocityGradient, double deltaTimeInSeconds);

protected:
    Eigen::Matrix3d deformation_gradient_elastic, deformation_gradient_plastic;
    double mu0, lambda0;    // Initial Lame coefficients
    double hardening_coefficient;
    double critical_compression, critical_stretch;
    ConstitutiveModel constitutive_model;

    // Constitutive models 
    // Hyperelasticity

    // Energy:

    __host__ __device__
    double fixedCorotatedEnergy() const;
    __host__ __device__
    double NeoHookeanEnergy() const;

    // Stress:
    //  Piola-Kirchhoff Stress x F^T

    __host__ __device__
    const Eigen::Matrix3d fixedCorotatedEnergyDerivativeTimesDeformationGradientTranspose() const;
    __host__ __device__
    const Eigen::Matrix3d NeoHookeanEnergyEnergyDerivativeTimesDeformationGradientTranspose() const;

    // Finite-strain multiplicative plasticity law 
    __host__ __device__
    const thrust::pair<double, double> plasticLame() const;
};

}   // namespace chains

#endif  // CHAINS_MATERIAL_POINT3D_H_