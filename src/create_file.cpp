#include "create_file.h"

#include <sstream>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <iomanip>  // for std::put_time
#include <cmath>

namespace chains {

std::string createDirectoryWithPrefixAndCurrentTime(const std::string& prefix) {
    namespace fs = std::filesystem;
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);

    // Format the time as a string
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time), "%Y-%m-%d_%H-%M-%S");
    std::string dir_name = prefix + "_" + ss.str();  // Add prefix before the time

    // Construct the full path
    fs::path output_dir = "./output/" + dir_name;

    // Check if the directory already exists
    if (!fs::exists(output_dir)) {
        // Create the directory
        if (fs::create_directory(output_dir)) {
            std::cout << "[INFO] Directory created: " << output_dir << std::endl;
        } else {
            std::cerr << "[ERROR] Failed to create directory: " << output_dir << std::endl;
        }
    } else {
        std::cout << "[INFO] Directory already exists: " << output_dir << std::endl;
    }

    return output_dir.string();
}

std::string createFileWithPaddingNumberNameInDirectory(int number, size_t padding, std::string directory) {
    std::string number_str = std::to_string(number);
    std::string file_name = directory + "/" + std::string(padding - std::min(padding, number_str.length()), '0') + number_str + ".?";
    std::ofstream out_file(file_name);

    if (out_file.is_open()) {
        std::cout << "[INFO] File created: " << file_name << std::endl;
    } else {
        std::cerr << "[ERROR] Failed to create file: " << file_name << std::endl;
    }

    return file_name;
}

}
