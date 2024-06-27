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

struct MaterialPoint3D {
    float volume0, mass;
    Eigen::Vector3f position, velocity;
    Eigen::Matrix3f deformation_gradient_elastic, deformation_gradient_plastic;
    float mu0, lambda0;    // Initial Lame coefficients
    float hardening_coefficient;
    float critical_compression, critical_stretch;
    ConstitutiveModel constitutive_model;

    __host__ __device__
    MaterialPoint3D() { }

    __host__ __device__
    MaterialPoint3D(
        Eigen::Vector3f particlePosition,
        Eigen::Vector3f particleVelocity,
        float particleMass,
        float hardeningCoefficient,
        float YoungModulus,
        float PoissonRatio,
        float criticalCompression,
        float criticleStretch,
        ConstitutiveModel constitutiveModel
    );

    __host__ __device__
    ~MaterialPoint3D() { }

    __device__
    const Eigen::Matrix3f volumeTimesCauchyStress();

    __device__
    void updatePosition(float deltaTimeInSeconds);

    __device__
    void updateDeformationGradient(Eigen::Matrix3f velocityGradient, float deltaTimeInSeconds);

    // Constitutive models 
    // Hyperelasticity

    // Energy:

    __device__
    float fixedCorotatedEnergy() const;
    __device__
    float NeoHookeanEnergy() const;

    // Stress:
    //  Piola-Kirchhoff Stress x F^T

    __device__
    const Eigen::Matrix3f fixedCorotatedEnergyDerivativeTimesDeformationGradientTranspose() const;
    __device__
    const Eigen::Matrix3f NeoHookeanEnergyEnergyDerivativeTimesDeformationGradientTranspose();

    // Finite-strain multiplicative plasticity law 
    __device__
    const thrust::pair<float, float> plasticLame() const;
};

}   // namespace chains

#endif  // CHAINS_MATERIAL_POINT3D_H_