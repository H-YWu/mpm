#ifndef CHAINS_MPM3D_PARSER_H_
#define CHAINS_MPM3D_PARSER_H_

#include "material_point3d.h"
#include "interpolation3d.h"

#include <Eigen/Dense>
#include <vector>
#include <string>

namespace chains {

struct MPM3DConfiguration {
    // Particle groups
    std::vector<std::string> group_paths;
    std::vector<Eigen::Vector3f> velocities;
    std::vector<float>
        masses,
        youngs, poissons, hardenings,
        compressions, stretches;
    std::vector<ConstitutiveModel> constitutive_models;
    // Grid
    Eigen::Vector3f origin;
    Eigen::Vector3i resolution;
    float stride, friction_coeff;
    // Transfer
    float blend_coeff;
    // Interpolation
    InterpolationType interp_type;
    // Simulate
    float delta_time;
    // Render
    bool offline;
    std::string particle_frag_shader_path;
};

MPM3DConfiguration parseYAML(const std::string& filePath);

}   // namespace chains

#endif  // CHAINS_MPM3D_PARSER_H_