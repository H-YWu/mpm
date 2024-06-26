#ifndef CHAINS_ONLINE_MPM3D_RENDERER_H_
#define CHAINS_ONLINE_MPM3D_RENDERER_H_

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <Eigen/Dense>
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
    OnlineMPM3DRenderer(
        int windowWidth,
        int windowHeight,
        int particlesNumber,
        Eigen::Vector3f gridOrigin,
        Eigen::Vector3f gridTarget,
        float gridStride,
        const std::string& particleFragmentPath
    );

    void render();

    GLuint getMaterilPointVBO();

private:
    // Plane (floor of the grid)
    //  vertices
    GLfloat _plane_vertices[12];
    //  faces
    const GLuint _plane_indices[6] = {
        0, 1, 2,
        2, 3, 0
    };

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
    GLfloat _radius;
    int _particles_num;

    // Grid
    Eigen::Vector3f _grid_origin;
    Eigen::Vector3f _grid_target;
    Eigen::Vector3f _grid_displacement;

    // Transform 
    glm::mat4 _view_mat;    // camera
    glm::mat4 _projection_mat;
    float _field_of_view = 45.0;

    void renderFloor();
    void renderMaterialPoint();
};

}   // namespace chains

#endif  // CHAINS_ONLINE_MPM3D_RENDERER_H_