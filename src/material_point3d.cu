#include "material_point3d.h"
#include "svd_eigen.h"

#include <thrust/tuple.h>
#include <cmath>

namespace chains {

__host__ __device__
MaterialPoint3D::MaterialPoint3D(
    Eigen::Vector3f particlePosition,
    Eigen::Vector3f particleVelocity,
    float particleMass,
    float hardeningCoefficient,
    float YoungModulus,
    float PoissonRatio,
    float criticalCompression,
    float criticleStretch,
    ConstitutiveModel constitutiveModel
) : position(particlePosition), velocity(particleVelocity),
    mass(particleMass), hardening_coefficient(hardeningCoefficient),
    critical_compression(criticalCompression), critical_stretch(criticleStretch),
    constitutive_model(constitutiveModel)
{
    mu0 = YoungModulus / (2.0*(1.0+PoissonRatio));
    lambda0 = (YoungModulus*PoissonRatio) / ((1.0+PoissonRatio)*(1.0-2.0*PoissonRatio));

    deformation_gradient_elastic.setIdentity();
    deformation_gradient_plastic.setIdentity();
}

__device__
const Eigen::Matrix3f MaterialPoint3D::volumeTimesCauchyStress() {
    switch (constitutive_model)
    {
    case ConstitutiveModel::NEO_HOOKEAN:
        return volume0 * NeoHookeanEnergyEnergyDerivativeTimesDeformationGradientTranspose();
        break;
    
    case ConstitutiveModel::FIXED_COROTATED:
    default:
        return volume0 * fixedCorotatedEnergyDerivativeTimesDeformationGradientTranspose();
        break;
    }
}

__device__
void MaterialPoint3D::updatePosition(float deltaTimeInSeconds) {
    position += deltaTimeInSeconds * velocity;
}

__device__
void MaterialPoint3D::updateDeformationGradient(Eigen::Matrix3f velocityGradient, float deltaTimeInSeconds) {
    deformation_gradient_elastic = 
        (Eigen::Matrix3f::Identity() + (deltaTimeInSeconds * velocityGradient))
        * deformation_gradient_elastic;
    
    Eigen::Matrix3f F_full(deformation_gradient_elastic * deformation_gradient_plastic);

    Eigen::Matrix3f U, V;
    Eigen::Vector3f s;
    svd3x3(deformation_gradient_elastic, U, s, V);
    // Clamp the singular values to the permitted range
    s= s.array().min(1.0f+critical_stretch).max(1.0f-critical_compression);
    Eigen::Matrix3f S; S.setZero();
    for (int i = 0; i < 3; i ++) S(i,i) = s(i);

    deformation_gradient_elastic = U * S * V.transpose();
    deformation_gradient_plastic = V * S.inverse() * U.transpose() * F_full;
}

__device__
float MaterialPoint3D::fixedCorotatedEnergy() const {
    float mu_p, lambda_p;
    thrust::tie(mu_p, lambda_p) = plasticLame();

    Eigen::Matrix3f U, V;
    Eigen::Vector3f s;
    svd3x3(deformation_gradient_elastic, U, s, V);

    float Je = deformation_gradient_elastic.determinant();

    /* Fixed Corotated Model */
    //  \phi = \mu_P * \sum_{i=1}^{3} (\sigma_i - 1)^2 + \frac{\lambda_P}{2} (J_E - 1)^2
    float energy = 0.0;
    for (int i = 0; i < 3; i ++) {
        energy += pow(s(i)-1.0, 2);
    }
    energy *= mu_p;
    energy += 0.5 * lambda_p * pow(Je-1.0, 2);
    return energy;
}

__device__
float MaterialPoint3D::NeoHookeanEnergy() const {
    float mu_p, lambda_p;
    thrust::tie(mu_p, lambda_p) = plasticLame();

    float Je = deformation_gradient_elastic.determinant();

    float traceFTF = (deformation_gradient_elastic.transpose() * deformation_gradient_elastic).trace();

    /* Neo-Hookean Model */
    //  \phi = \frac{\mu_P}{2}(\tr(F^T F) - 3) - \mu_P \log(J_E) _ \frac{\lambda_P}{2} \log^2(J_E)
    return mu_p * (0.5*(traceFTF-3.0) - log(Je))
                    + 0.5 * lambda_p * pow(log(Je), 2);
}

__device__
const Eigen::Matrix3f MaterialPoint3D::fixedCorotatedEnergyDerivativeTimesDeformationGradientTranspose() const {
    float mu_p, lambda_p;
    thrust::tie(mu_p, lambda_p) = plasticLame();

    Eigen::Matrix3f U, V;
    Eigen::Vector3f s;
    svd3x3(deformation_gradient_elastic, U, s, V);
    Eigen::Matrix3f R = U*V.transpose();
    float Je = deformation_gradient_elastic.determinant();

    /* Fixed Corotated Model */
    //  First Piola-Kirchoff Stress: \frac{\partial \phi}{\partial F_E} = 
    //      2 \mu_P (F_E - R) + \lambda_P (J_E - 1) J_E F_E^{-T}
    //  \frac{\partial \phi}{\partial F_E} F_E^T =
    //      2 \mu_P (F_E - R) F_E^T + \lambda_P (J_E - 1) J_E F_E^{-T} F_E^T
    //      = 2 \mu_P (F_E - R) F_E^T + \lambda_P (J_E - 1) J_E I
    return 2.0*mu_p*(deformation_gradient_elastic-R)*deformation_gradient_elastic.transpose()
           + lambda_p*(Je-1.0)*Je*Eigen::Matrix3f::Identity();
}

__device__
const Eigen::Matrix3f MaterialPoint3D::NeoHookeanEnergyEnergyDerivativeTimesDeformationGradientTranspose() {
    float mu_p, lambda_p;
    thrust::tie(mu_p, lambda_p) = plasticLame();

    // Project the deformation gradient so that the determinant > 0
    Eigen::Matrix3f U, V;
    Eigen::Vector3f s;
    svd3x3(deformation_gradient_elastic, U, s, V);
    s = s.array().max(std::numeric_limits<float>::epsilon());
    Eigen::Matrix3f S; S.setZero();
    for (int i = 0; i < 3; i ++) S(i,i) = s(i);
    deformation_gradient_elastic = U * S * V.transpose();
    float Je = deformation_gradient_elastic.determinant(); 

    Eigen::Matrix3f I = Eigen::Matrix3f::Identity();

    /* Neo-Hookean Model */
    //  First Piola-Kirchoff Stress: \frac{\partial \phi}{\partial F_E} = 
    //      \mu_P (F_E - F_E^{-T}) + \lambda_P \log(J_E) F_E^{-T}
    //  \frac{\partial \phi}{\partial F_E} F_E^T =
    //      \mu_P (F_E - F_E^{-T}) F_E^T  + \lambda_P \log(J_E) F_E^{-T} F_E^T
    //      \mu_P (F_E F_E^T - I)  + \lambda_P \log(J_E) I
    return mu_p * (deformation_gradient_elastic*deformation_gradient_elastic.transpose()-I)
           + lambda_p * log(Je) * I;
}

__device__
const thrust::pair<float, float> MaterialPoint3D::plasticLame() const {
    float Jp = deformation_gradient_plastic.determinant();
    float decayingFactor = exp(hardening_coefficient * (1.0 - Jp));
    return thrust::make_pair(mu0 * decayingFactor, lambda0 * decayingFactor);
}

}   // namespace chains