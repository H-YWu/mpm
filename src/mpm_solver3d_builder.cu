#include "mpm_solver3d_builder.h"

#include <thrust/host_vector.h>
#include <iostream>
#include <fstream>
#include <sstream>

namespace chains {

bool readXYZFile(const std::string& filePath, std::vector<Eigen::Vector3d>& points) {
    std::ifstream xyz_ifstream(filePath);
    if (!xyz_ifstream.is_open()) {
        std::cerr << "[ERROR] cannot open file " << filePath << std::endl;
        return false;
    }

    std::string line;
    while (std::getline(xyz_ifstream, line)) {
        std::stringstream ss(line);
        Eigen::Vector3d point;
        if (!(ss >> point.x() >> point.y() >> point.z())) {
            std::cerr << "[ERROR] cannot parse line: " << line  << " when trying to parse file " << filePath << std::endl;
            return false;
        }
        points.push_back(point);
    }

    xyz_ifstream.close();
    return true;
}

MPMSolver3D buildMPMSolver3DFromYAML(MPM3DConfiguration config, int& particlesNumber) {
    // Build all particles
    size_t groups_num = config.group_paths.size();
    std::vector<thrust::host_vector<MaterialPoint3D>> particle_groups(groups_num);
    size_t total_size = 0;
    std::cout << "[INFO] building " << groups_num << " group(s) of particle on host" << std::endl;
    for (size_t i = 0; i < groups_num; i ++) {
        std::vector<Eigen::Vector3d> coords;
        readXYZFile(config.group_paths[i], coords);
        for (size_t j = 0; j < coords.size(); j ++) {
            particle_groups[i].push_back(
                MaterialPoint3D(
                    config.stride * coords[j],
                    config.velocities[i],
                    config.masses[i],
                    config.hardenings[i],
                    config.youngs[i],
                    config.poissons[i],
                    config.compressions[i],
                    config.stretches[i],
                    config.constitutive_models[i]
                )
            );
        }
        std::cout << "[INFO] particle group " << i << " has " << particle_groups[i].size() << " particles" << std::endl;
        total_size += particle_groups[i].size();
    }
    std::cout << "[INFO] stacking all " << total_size << " particles on host" << std::endl;
    thrust::host_vector<MaterialPoint3D> particles;
    particlesNumber = total_size;
    particles.reserve(total_size);
    for (const auto& group : particle_groups) {
        thrust::copy(group.begin(), group.end(), std::back_inserter(particles));
    }

    std::cout << "[INFO] building MPM Solver 3D" << std::endl;
    MPMSolver3D solver(
        particles,
        config.origin,
        config.resolution,
        config.stride,
        config.friction_coeff,
        config.blend_coeff,
        config.interp_type,
        config.delta_time
    );

    return solver;
}

}   // namespace chains