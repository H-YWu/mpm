#include "online_mpm3d_renderer.h"
#include "shader_loader.h"

namespace chains {

OnlineMPM3DRenderer::OnlineMPM3DRenderer
    (int windowWidth, int windowHeight, int particlesNumber, const std::string& particleFragmentPath)
    : _window_width(windowWidth), _window_height(windowHeight), _particles_num(particlesNumber)
{
    _aspect_ratio = static_cast<float>(_window_width) / _window_height;

    _view_mat = _origin_camera;
    _projection_mat = glm::perspective(
        glm::radians(_field_of_view),
        static_cast<float>(_window_width) / _window_height,
        0.1f,
        100.0f
    );

    // Plane
    _plane_shader = loadShader("../shader/plane.vert", "../shader/plane.frag");

    glUseProgram(_plane_shader);
    glUniformMatrix4fv(glGetUniformLocation(_plane_shader, "projection"), 1, GL_FALSE, glm::value_ptr(_projection_mat));

    glGenVertexArrays(1, &_plane_buffers.vao);
    glBindVertexArray(_plane_buffers.vao);

    glGenBuffers(1, &_plane_buffers.vbo);
    glBindBuffer(GL_ARRAY_BUFFER, _plane_buffers.vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(_plane_vertices), _plane_vertices, GL_STATIC_DRAW);

    // position attribute
    glVertexAttribPointer(0, 3, GL_DOUBLE, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glGenBuffers(1, &_plane_buffers.ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _plane_buffers.ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(_plane_indices), _plane_indices, GL_STATIC_DRAW);

    // Material Point 
    _particle_shader = loadShader("../shader/particle.vert", particleFragmentPath);

    glUseProgram(_particle_shader);
    glUniformMatrix4fv(glGetUniformLocation(_particle_shader, "projection"), 1, GL_FALSE, glm::value_ptr(_projection_mat));
    glUniform1f(glGetUniformLocation(_particle_shader, "radius"), radius_);
    glUniform1f(glGetUniformLocation(_particle_shader, "scale"), _window_width / _aspect_ratio * (1.0 / tanf(_field_of_view * 0.5f)));

    glGenVertexArrays(1, &_particle_buffers.vao);
    glBindVertexArray(_particle_buffers.vao);

    glGenBuffers(1, &_particle_buffers.vbo);
    glBindBuffer(GL_ARRAY_BUFFER, _particle_buffers.vbo);
    glBufferData(GL_ARRAY_BUFFER, _particles_num * 4 * sizeof(float), 0, GL_DYNAMIC_DRAW);

    glVertexAttribPointer(0, 4, GL_DOUBLE, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);

    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
}

void OnlineMPM3DRenderer::render() {
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    renderWall();
    renderFloor();
    renderMaterialPoint();
}

void OnlineMPM3DRenderer::renderWall() {
    glUseProgram(_plane_shader);
    glBindVertexArray(_plane_buffers.vao); // Bind the vertex array object (VAO) for the floor

    // Set view matrix uniform in the shader
    glUniformMatrix4fv(glGetUniformLocation(_plane_shader, "view"), 1, GL_FALSE, glm::value_ptr(_view_mat));

    // Set model matrix uniform in the shader (translate and scale)
    glm::mat4 model(1.0f);
    model = glm::translate(model, glm::vec3(0.0f, 1.5f, -1.5f));
    model = glm::scale(model, glm::vec3(3.0f, 3.0f, 3.0f));
    glUniformMatrix4fv(glGetUniformLocation(_plane_shader, "model"), 1, GL_FALSE, glm::value_ptr(model));

    // Draw the wall using indexed vertices (assuming a simple plane defined by 6 vertices)
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
}

void OnlineMPM3DRenderer::renderFloor() {
    glUseProgram(_plane_shader);
    glBindVertexArray(_plane_buffers.vao); // Bind the vertex array object (VAO) for the floor

    // Set view matrix uniform in the shader
    glUniformMatrix4fv(glGetUniformLocation(_plane_shader, "view"), 1, GL_FALSE, glm::value_ptr(_view_mat));

    // Set model matrix uniform in the shader (scale and rotate)
    glm::mat4 model(1.0f); // Identity matrix for model transformation
    model = glm::scale(model, glm::vec3(5.0f, 5.0f, 5.0f));
    model = glm::rotate(model, glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
    glUniformMatrix4fv(glGetUniformLocation(_plane_shader, "model"), 1, GL_FALSE, glm::value_ptr(model));

    // Draw the floor using indexed vertices (assuming a simple plane defined by 6 vertices)
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
}

void OnlineMPM3DRenderer::renderMaterialPoint() {
    glUseProgram(_particle_shader);
    glBindVertexArray(_particle_buffers.vao);

    // Set the view matrix uniform in the shader
    glUniformMatrix4fv(glGetUniformLocation(_particle_shader, "view"), 1, GL_FALSE, glm::value_ptr(_view_mat));
    // Set the model matrix uniform in the shader
    glm::mat4 model(1.0f); // Identity matrix for model transformation
    glUniformMatrix4fv(glGetUniformLocation(_particle_shader, "model"), 1, GL_FALSE, glm::value_ptr(model));

    // Draw the particles as points
    glDrawArrays(GL_POINTS, 0, GLsizei(_particles_num));
}

GLuint OnlineMPM3DRenderer::getMaterilPointVBO() {
    return _particle_buffers.vbo;
}

void OnlineMPM3DRenderer::viewFromOriginCamera() {
    _view_mat = _origin_camera;
}

void OnlineMPM3DRenderer::viewFromUpCamera() {
    _view_mat = _up_camera;
}

void OnlineMPM3DRenderer::viewFromFrontCamera() {
    _view_mat = _front_camera;
}

void OnlineMPM3DRenderer::viewFromSideCamera() {
    _view_mat = _side_camera;
}

}   // namespace chains