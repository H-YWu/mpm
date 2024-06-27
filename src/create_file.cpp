#include "create_file.h"

#include <sstream>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <iomanip>  // for std::put_time
#include <cmath>

namespace chains {

void createDirctory(const std::filesystem::path &directoryPath) {
    if (!std::filesystem::exists(directoryPath)) {
        if (std::filesystem::create_directory(directoryPath)) {
            std::cout << "[INFO] Directory " << directoryPath << " is created successfully" << std::endl;
        } else {
            std::cerr << "[ERROR] Failed to create directory " << directoryPath << std::endl;
        }
    } else {
        std::cout << "[INFO] Directory " << directoryPath << " already exists" << std::endl;
    }
}

std::string createNewDirectoryInParentDirectoryWithPrefixAndCurrentTime(const std::string &parentDirectoryPath, const std::string& prefix) {
    std::string p_path = parentDirectoryPath;
    if (p_path[p_path.size()-1] != '/') p_path.push_back('/');
    std::filesystem::path parent_dir = p_path;
    std::cout << "[INFO] Trying to create the parent directory " << parent_dir << std::endl;
    createDirctory(parent_dir);

    // Get current time
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    // Format the time as a string
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time), "%Y-%m-%d_%H-%M-%S");
    std::string dir_name = prefix + "_" + ss.str();  // Add prefix before the time

    std::filesystem::path output_dir = p_path + dir_name;   // Construct the full path
    std::cout << "[INFO] Trying to create the output directory " << output_dir << std::endl;
    createDirctory(output_dir);

    return output_dir.string();
}

std::string createFileWithPaddingNumberNameInDirectory(int number, size_t padding, const std::string &directory, const std::string &fileExtension) {
    std::string number_str = std::to_string(number);
    std::string file_name = directory + "/" + std::string(padding - std::min(padding, number_str.length()), '0') + number_str + fileExtension;
    std::ofstream out_file(file_name);

    if (out_file.is_open()) {
        std::cout << "[INFO] File " << file_name << " is created successfully" << std::endl;
    } else {
        std::cerr << "[ERROR] Failed to create file: " << file_name << std::endl;
    }

    return file_name;
}

}
