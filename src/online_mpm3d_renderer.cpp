#include "online_mpm3d_renderer.h"
#include "shader_loader.h"

#include <cmath>
#include <algorithm>

namespace chains {

OnlineMPM3DRenderer::OnlineMPM3DRenderer(
    int windowWidth,
    int windowHeight,
    int particlesNumber,
    Eigen::Vector3f gridOrigin,
    Eigen::Vector3f gridTarget,
    float gridStride,
    const std::string& particleFragmentPath
) : _window_width(windowWidth),
    _window_height(windowHeight),
    _radius(gridStride*0.8),
    _particles_num(particlesNumber),
    _grid_origin(gridOrigin),
    _grid_target(gridTarget),
    _grid_displacement(gridTarget-gridOrigin)
{
    _aspect_ratio = static_cast<float>(_window_width) / _window_height;

    float far_dis = 0.0f;
    for (int i = 0; i < 3; i ++) {
        far_dis += pow(5.0f*_grid_displacement(i), 2);
    }
    far_dis = sqrt(far_dis);

    Eigen::Vector3f center = 0.5f * (gridOrigin+gridTarget);

    auto target_pos = glm::vec3(center(0), center(1), center(2));
    auto camera_pos = target_pos + glm::vec3(_grid_displacement(0), -0.4*_grid_displacement(1), 1.3*_grid_displacement(2));

    _view_mat =
        glm::lookAt(
            camera_pos,
            target_pos, 
            glm::vec3(0.0f, 1.0f, 0.0f)   // up vector
        );
    _projection_mat = glm::perspective(
        glm::radians(_field_of_view),
        _aspect_ratio,
        0.1f,
        far_dis 
    );

    // Plane
    _plane_shader = loadShader("../shader/plane.vert", "../shader/plane.frag");

    float temp[12] = {  // floor of the grid
        gridOrigin(0), gridOrigin(1), gridOrigin(2),
        gridTarget(0), gridOrigin(1), gridOrigin(2),
        gridTarget(0), gridOrigin(1), gridTarget(2),
        gridOrigin(0), gridOrigin(1), gridTarget(2)
    };
    std::copy(std::begin(temp), std::end(temp), std::begin(_plane_vertices));

    glUseProgram(_plane_shader);
    glUniformMatrix4fv(glGetUniformLocation(_plane_shader, "projection"), 1, GL_FALSE, glm::value_ptr(_projection_mat));
    glUniformMatrix4fv(glGetUniformLocation(_plane_shader, "view"), 1, GL_FALSE, glm::value_ptr(_view_mat));

    glGenVertexArrays(1, &_plane_buffers.vao);
    glBindVertexArray(_plane_buffers.vao);

    glGenBuffers(1, &_plane_buffers.vbo);
    glBindBuffer(GL_ARRAY_BUFFER, _plane_buffers.vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(_plane_vertices), _plane_vertices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glGenBuffers(1, &_plane_buffers.ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _plane_buffers.ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(_plane_indices), _plane_indices, GL_STATIC_DRAW);

    // Material Point 
    _particle_shader = loadShader("../shader/particle.vert", particleFragmentPath);

    glUseProgram(_particle_shader);
    glUniformMatrix4fv(glGetUniformLocation(_particle_shader, "projection"), 1, GL_FALSE, glm::value_ptr(_projection_mat));
    glUniformMatrix4fv(glGetUniformLocation(_particle_shader, "view"), 1, GL_FALSE, glm::value_ptr(_view_mat));
    glUniform1f(glGetUniformLocation(_particle_shader, "radius"), _radius);
    glUniform1f(glGetUniformLocation(_particle_shader, "scale"), _window_width / _aspect_ratio * (1.0 / tanf(_field_of_view * 0.5f)));

    glGenVertexArrays(1, &_particle_buffers.vao);
    glBindVertexArray(_particle_buffers.vao);

    glGenBuffers(1, &_particle_buffers.vbo);
    glBindBuffer(GL_ARRAY_BUFFER, _particle_buffers.vbo);
    glBufferData(GL_ARRAY_BUFFER, _particles_num * 4 * sizeof(float), 0, GL_DYNAMIC_DRAW);

    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);

    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
}

void OnlineMPM3DRenderer::render() {
    glClearColor(0.302f, 0.153f, 0.102f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    renderFloor();
    renderMaterialPoint();
}

void OnlineMPM3DRenderer::renderFloor() {
    glUseProgram(_plane_shader);
    glBindVertexArray(_plane_buffers.vao); // Bind the vertex array object (VAO) for the floor

    // Set model matrix uniform in the shader (scale and rotate)
    glm::mat4 model(1.0f); // Identity matrix for model transformation
    glUniformMatrix4fv(glGetUniformLocation(_plane_shader, "model"), 1, GL_FALSE, glm::value_ptr(model));

    // Draw the floor using indexed vertices (assuming a simple plane defined by 6 vertices)
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
}

void OnlineMPM3DRenderer::renderMaterialPoint() {
    glUseProgram(_particle_shader);
    glBindVertexArray(_particle_buffers.vao);

    // Set the model matrix uniform in the shader
    glm::mat4 model(1.0f); // Identity matrix for model transformation
    glUniformMatrix4fv(glGetUniformLocation(_particle_shader, "model"), 1, GL_FALSE, glm::value_ptr(model));

    // Draw the particles as points
    glDrawArrays(GL_POINTS, 0, GLsizei(_particles_num));
}

GLuint OnlineMPM3DRenderer::getMaterilPointVBO() {
    return _particle_buffers.vbo;
}

}   // namespace chains