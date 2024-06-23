#ifndef CHAINS_SHADER_LOADER_H_
#define CHAINS_SHADER_LOADER_H_

#include <glad/glad.h>
#include <string>

namespace chains {

GLuint loadShader(const std::string& vertexPath, const std::string& fragmentPath);

}   // namespace chains

#endif  // CHAINS_SHADER_LOADER_H_