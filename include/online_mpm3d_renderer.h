#ifndef CHAINS_ONLINE_MPM3D_RENDERER_H_
#define CHAINS_ONLINE_MPM3D_RENDERER_H_

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <string>

namespace chains {

struct GLBuffers {
    GLuint vao;
    GLuint vbo;
    GLuint ebo;
};

struct CUDABUffers {
    GLuint vao;
    GLuint vbo;
};

class OnlineMPM3DRenderer {
public:
    OnlineMPM3DRenderer(int windowWidth, int windowHeight, int particlesNumber, const std::string& particleFragmentPath);

    void render();

    GLuint getMaterilPointVBO();

    // Adjust viewing
    void viewFromOriginCamera();
    void viewFromUpCamera();
    void viewFromFrontCamera();
    void viewFromSideCamera();
private:
    // Plane
    //  vertices
    const GLfloat _plane_vertices[12] = {
         0.5f,  0.5f, 0.0f,
        -0.5f,  0.5f, 0.0f,
        -0.5f, -0.5f, 0.0f,
         0.5f, -0.5f, 0.0f,
    };
    //  faces
    const GLuint _plane_indices[6] = {
        0, 1, 2,
        2, 3, 0
    };

    // Camera
    glm::mat4 _origin_camera = 
        glm::lookAt(
            glm::vec3(5.5f, 1.2f, 5.5f),  // camera position
            glm::vec3(0.0f, 1.2f, 0.0f),  // target position
            glm::vec3(0.0f, 1.0f, 0.0f)   // up vector
        );
    glm::mat4 _up_camera =
        glm::lookAt(
            glm::vec3(0.0f, 6.8f, 0.0f),  // camera position
            glm::vec3(0.0f, 1.2f, 0.0f),  // target position
            glm::vec3(0.0f, 0.0f, -1.0f)  // up vector
        );
    glm::mat4 _front_camera =
        glm::lookAt(
            glm::vec3(0.0f, 1.2f, 7.0f),  // camera position
            glm::vec3(0.0f, 1.2f, 0.0f),  // target position
            glm::vec3(0.0f, 1.0f, 0.0f)   // up vector
        );
    glm::mat4 _side_camera =
        glm::lookAt(
            glm::vec3(7.0f, 1.2f, 0.0f),  // camera position
            glm::vec3(0.0f, 1.2f, 0.0f),  // target position
            glm::vec3(0.0f, 1.0f, 0.0f)   // up vector
        );

    // Window
    int _window_width;
    int _window_height;
    float _aspect_ratio;

    // Plane
    GLBuffers _plane_buffers;
    GLuint _plane_shader;

    // Material point
    CUDABUffers _particle_buffers;
    GLuint _particle_shader;
    GLfloat radius_ = 0.015;
    int _particles_num;

    // Model
    glm::mat4 _view_mat;
    glm::mat4 _projection_mat;
    float _field_of_view = 45.0;

    void renderWall();
    void renderFloor();
    void renderMaterialPoint();
};

}   // namespace chains

#endif  // CHAINS_ONLINE_MPM3D_RENDERER_H_