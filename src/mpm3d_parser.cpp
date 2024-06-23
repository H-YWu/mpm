#include "mpm3d_parser.h"

#include <yaml-cpp/yaml.h>

namespace chains {

ConstitutiveModel parseConstitutiveModel(const std::string& model) {
    if (model == "FixedCorotated") return ConstitutiveModel::FIXED_COROTATED;
    if (model == "NeoHookean") return ConstitutiveModel::NEO_HOOKEAN;
    throw std::runtime_error("Unsupported constitutive model: " + model);
}

InterpolationType parseInterpolationType(const std::string& type) {
    if (type == "Linear") return InterpolationType::LINEAR;
    if (type == "QuadraticBSpline") return InterpolationType::QUADRATIC_BSPLINE;
    if (type == "CubicBSpline") return InterpolationType::CUBIC_BSPLINE;
    throw std::runtime_error("Unsupported interpolation type: " + type);
}

MPM3DConfiguration parseYAML(const std::string& filePath) {

    YAML::Node config_yaml_node = YAML::LoadFile(filePath);

    MPM3DConfiguration mpm_config;

    // Parse ParticleGroups
    for (const auto& group : config_yaml_node["ParticleGroups"]) {
        mpm_config.group_paths.push_back(group["File"].as<std::string>());
        mpm_config.masses.push_back(group["Mass"].as<double>());
        mpm_config.velocities.push_back(Eigen::Vector3d(
            group["Velocity"][0].as<double>(),
            group["Velocity"][1].as<double>(),
            group["Velocity"][2].as<double>()
        ));
        mpm_config.youngs.push_back(group["YoungModulus"].as<double>());
        mpm_config.poissons.push_back(group["PoissonRatio"].as<double>());
        mpm_config.hardenings.push_back(group["HardeningCoefficient"].as<double>());
        mpm_config.compressions.push_back(group["CriticalCompression"].as<double>());
        mpm_config.stretches.push_back(group["CriticalStretch"].as<double>());
        mpm_config.constitutive_models.push_back(parseConstitutiveModel(group["ConstitutiveModel"].as<std::string>()));
    }

    // Parse Grid
    mpm_config.origin = Eigen::Vector3d(
        config_yaml_node["Grid"]["Origin"][0].as<double>(),
        config_yaml_node["Grid"]["Origin"][1].as<double>(),
        config_yaml_node["Grid"]["Origin"][2].as<double>()
    );
    mpm_config.resolution = Eigen::Vector3i(
        config_yaml_node["Grid"]["Resolution"][0].as<int>(),
        config_yaml_node["Grid"]["Resolution"][1].as<int>(),
        config_yaml_node["Grid"]["Resolution"][2].as<int>()
    );
    mpm_config.stride = config_yaml_node["Grid"]["Stride"].as<double>();
    mpm_config.friction_coeff = config_yaml_node["Grid"]["BoundaryFrictionCoefficient"].as<double>();

    // Parse Transfer
    mpm_config.blend_coeff = config_yaml_node["Transfer"]["BlendCoefficient"].as<double>();

    // Parse Interpolation
    mpm_config.interp_type = parseInterpolationType(config_yaml_node["Interpolation"]["Type"].as<std::string>());

    // Parse Interpolation
    mpm_config.offline = config_yaml_node["Render"]["Offline"].as<bool>();
    mpm_config.particle_frag_shader_path = config_yaml_node["Render"]["ParticleFragmentShader"].as<std::string>();

    return mpm_config;
}

}   // namespace chains