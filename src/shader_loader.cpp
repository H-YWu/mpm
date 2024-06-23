#include "shader_loader.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

namespace chains {

void checkErrors(GLuint id, const std::string& type) {
	GLint success;
	if (type == "Shader") {
		glGetShaderiv(id, GL_COMPILE_STATUS, &success);
		if (!success) {
			GLint logLength;
			glGetShaderiv(id, GL_INFO_LOG_LENGTH, &logLength);
			std::vector<GLchar> infoLog((logLength > 1) ? logLength : 1);
			glGetShaderInfoLog(id, logLength, nullptr, infoLog.data());
			std::cerr << infoLog.data() << std::endl;
		}
	} else {  // Program
		glGetProgramiv(id, GL_LINK_STATUS, &success);
		if (!success) {
			GLint logLength;
			glGetProgramiv(id, GL_INFO_LOG_LENGTH, &logLength);
			std::vector<GLchar> infoLog((logLength > 1) ? logLength : 1);
			glGetProgramInfoLog(id, logLength, nullptr, infoLog.data());
			std::cerr << infoLog.data() << std::endl;
		}
	}
}

GLuint loadShader(
	const std::string& vertexPath,
    const std::string& fragmentPath
) {
	// 1. retrieve the vertex/fragment source code from filePath
	std::string vertex_code;
	std::string fragment_code;
	std::ifstream vertex_shader_file;
	std::ifstream fragment_shader_file;
	// ensure ifstream objects can throw exceptions:
	vertex_shader_file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
	fragment_shader_file.exceptions(std::ifstream::failbit |
									std::ifstream::badbit);
	try {
		// open files
		vertex_shader_file.open(vertexPath);
		fragment_shader_file.open(fragmentPath);
		std::stringstream vertex_shader_stream, fragment_shader_stream;
		// read file's buffer contents into streams
		vertex_shader_stream << vertex_shader_file.rdbuf();
		fragment_shader_stream << fragment_shader_file.rdbuf();
		// close file handlers
		vertex_shader_file.close();
		fragment_shader_file.close();
		// convert stream into string
		vertex_code = vertex_shader_stream.str();
		fragment_code = fragment_shader_stream.str();
	} catch (const std::ifstream::failure& e) {
		std::cerr << "[ERROR] Failed to load shader source file!" << std::endl;
	}

	// 2. compile shaders
	auto compile_shader = [](GLuint& id, const std::string& code) {
		const char* shader_code = code.c_str();
		glShaderSource(id, 1, &shader_code, nullptr);
		glCompileShader(id);
		checkErrors(id, "Shader");
	};

	GLuint vertex_shader = glCreateShader(GL_VERTEX_SHADER);
	compile_shader(vertex_shader, vertex_code);
	GLuint fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
	compile_shader(fragment_shader, fragment_code);

	// id Program
	GLuint shader_program = glCreateProgram();
	glAttachShader(shader_program, vertex_shader);
	glAttachShader(shader_program, fragment_shader);
	glLinkProgram(shader_program);
	checkErrors(shader_program, "Program");

	// delete the shaders as they're linked into our program now and no longer
	// necessary
	glDeleteShader(vertex_shader);
	glDeleteShader(fragment_shader);

	return shader_program;
}

}   // namespace chains