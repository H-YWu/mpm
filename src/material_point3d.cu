#include "material_point3d.h"

#include <Eigen/SVD>
#include <thrust/tuple.h>
#include <cmath>

namespace chains {

__host__ __device__
MaterialPoint3D::MaterialPoint3D(
    Eigen::Vector3d particlePosition,
    Eigen::Vector3d particleVelocity,
    double particleMass,
    double hardeningCoefficient,
    double YoungModulus,
    double PoissonRatio,
    double criticalCompression,
    double criticleStretch,
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

__host__ __device__
const Eigen::Matrix3d MaterialPoint3D::volumeTimesCauchyStress() const {
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

__host__ __device__
void MaterialPoint3D::updatePosition(double deltaTimeInSeconds) {
    position += deltaTimeInSeconds * velocity;
}

__host__ __device__
void MaterialPoint3D::updateDeformationGradient(Eigen::Matrix3d velocityGradient, double deltaTimeInSeconds) {
    deformation_gradient_elastic = 
        (Eigen::Matrix3d::Identity() + (deltaTimeInSeconds * velocityGradient))
        * deformation_gradient_plastic;
    
    Eigen::Matrix3d F_full(deformation_gradient_elastic * deformation_gradient_plastic);

    Eigen::JacobiSVD<Eigen::Matrix3d> svd(deformation_gradient_elastic, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Vector3d sig_val = svd.singularValues();
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    // Clamp the singular values to the permitted range
    double lower_bnd = 1.0 - critical_compression;
    double upper_bnd = 1.0 + critical_stretch;
    for (int i = 0; i < 3; ++i) {
        if (sig_val(i) < lower_bnd) {
            sig_val(i) = lower_bnd;
        } else if (sig_val(i) > upper_bnd) {
            sig_val(i) = upper_bnd;
        }
    }
    // Update elastic and plastic deformation gradient 
    Eigen::Matrix3d sigma = sig_val.asDiagonal();
    deformation_gradient_elastic = U * sigma * V.transpose();
    deformation_gradient_plastic = V * sigma.inverse() * U.transpose() * F_full;
}

__host__ __device__
double MaterialPoint3D::fixedCorotatedEnergy() const {
    double mu_p, lambda_p;
    thrust::tie(mu_p, lambda_p) = plasticLame();

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(deformation_gradient_elastic, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Vector3d sigmas = svd.singularValues();
    double Je = deformation_gradient_elastic.determinant();

    /* Fixed Corotated Model */
    //  \phi = \mu_P * \sum_{i=1}^{3} (\sigma_i - 1)^2 + \frac{\lambda_P}{2} (J_E - 1)^2
    double energy = 0.0;
    for (int i = 0; i < 3; i ++) {
        energy += std::pow(sigmas(i)-1.0, 2);
    }
    energy *= mu_p;
    energy += 0.5 * lambda_p * std::pow(Je-1.0, 2);
    return energy;
}

__host__ __device__
double MaterialPoint3D::NeoHookeanEnergy() const {
    double mu_p, lambda_p;
    thrust::tie(mu_p, lambda_p) = plasticLame();

    double Je = deformation_gradient_elastic.determinant();
    double traceFTF = (deformation_gradient_elastic.transpose() * deformation_gradient_elastic).trace();

    /* Neo-Hookean Model */
    //  \phi = \frac{\mu_P}{2}(\tr(F^T F) - 3) - \mu_P \log(J_E) _ \frac{\lambda_P}{2} \log^2(J_E)
    return mu_p * (0.5*(traceFTF-3.0) - std::log(Je))
                    + 0.5 * lambda_p * std::pow(std::log(Je), 2);
}

__host__ __device__
const Eigen::Matrix3d MaterialPoint3D::fixedCorotatedEnergyDerivativeTimesDeformationGradientTranspose() const {
    double mu_p, lambda_p;
    thrust::tie(mu_p, lambda_p) = plasticLame();

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(deformation_gradient_elastic, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d VT = svd.matrixV().transpose();
    Eigen::Matrix3d R = U*VT;
    double Je = deformation_gradient_elastic.determinant();

    /* Fixed Corotated Model */
    //  First Piola-Kirchoff Stress: \frac{\partial \phi}{\partial F_E} = 
    //      2 \mu_P (F_E - R) + \lambda_P (J_E - 1) J_E F_E^{-T}
    //  \frac{\partial \phi}{\partial F_E} F_E^T =
    //      2 \mu_P (F_E - R) F_E^T + \lambda_P (J_E - 1) J_E F_E^{-T} F_E^T
    //      = 2 \mu_P (F_E - R) F_E^T + \lambda_P (J_E - 1) J_E I
    return 2.0*mu_p*(deformation_gradient_elastic-R)*deformation_gradient_elastic.transpose()
           + lambda_p*(Je-1.0)*Je*Eigen::Matrix3d::Identity();
}

__host__ __device__
const Eigen::Matrix3d MaterialPoint3D::NeoHookeanEnergyEnergyDerivativeTimesDeformationGradientTranspose() const {
    double mu_p, lambda_p;
    thrust::tie(mu_p, lambda_p) = plasticLame();

    double Je = deformation_gradient_elastic.determinant();
    Eigen::Matrix3d I = Eigen::Matrix3d::Identity();

    /* Neo-Hookean Model */
    //  First Piola-Kirchoff Stress: \frac{\partial \phi}{\partial F_E} = 
    //      \mu_P (F_E - F_E^{-T}) + \lambda_P \log(J_E) F_E^{-T}
    //  \frac{\partial \phi}{\partial F_E} F_E^T =
    //      \mu_P (F_E - F_E^{-T}) F_E^T  + \lambda_P \log(J_E) F_E^{-T} F_E^T
    //      \mu_P (F_E F_E^T - I)  + \lambda_P \log(J_E) I
    return mu_p * (deformation_gradient_elastic*deformation_gradient_elastic.transpose()-I)
           + lambda_p * std::log(Je) * I;
}

__host__ __device__
const thrust::pair<double, double> MaterialPoint3D::plasticLame() const {
    double Jp = deformation_gradient_plastic.determinant();
    double decayingFactor = std::exp(hardening_coefficient * (1.0 - Jp));
    return thrust::make_pair(mu0 * decayingFactor, lambda0 * decayingFactor);
}

}   // namespace chains