# A Naive GPU Implementation of Material Point Method

![Hello, MPM!](./figs/hello_mpm.png)

## Tested Environment

- Operating System: Windows Subsystem for Linux (Windows 11, Ubuntu 22.04.3 LTS)
- CPU: 13th Gen Intel(R) Core(TM) i9-13900HX
- RAM: 32GB
- GPU: NVIDIA GeForce RTX 4080 Laptop
- Softwares:
    - CUDA 12.4 WSL

## Dependencies

- You have to manually install on your machine:
    - CUDA (version 12.5 may not work)
    - OpenVDB
- CMake will automatically fetch:
    - GLFW
    - glad
    - Eigen
    - yaml-cpp
    - glm
    - Dear ImGui
- Download this hearder-only library using git submodule:
    - 3x3_SVD_CUDA ([my adapted version](https://github.com/H-YWu/3x3_SVD_CUDA) from [Fast CUDA 3x3 SVD](https://github.com/kuiwuchn/3x3_SVD_CUDA))

## Build & Run

1. build the project (check [the issue section](#known-issues) first if you encounter any problem):

```bash
mdkir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j8
```

2. copy the three directories _./config_, _./shader_ and _./data_ to the directory _./build/src/demo/_

3. goto _./build/src/demo/<program_dir>_ and run:

```bash
./<executable> ../config/<config_file_name>
```

## Features

- Grid
    - [x] Collocated grid
    - [ ] Staggered grid
- Interpolation
    - [x] Cubic BSpline 
    - [x] Quadratic BSpline 
    - [x] Linear
- Transfer
    - [x] PIC-FLIP
    - [x] APIC
    - [ ] PolyPIC 
- Integration
    - [x] Explicit
    - [ ] Semi-implicit 
    - [ ] Implicit
    - [ ] CFL
- Collision
    - [x] Grid boundary
    - [ ] Level set collision objects 

## Render

### Online

I simply use OpenGL to render material points as points.

### Offline

In each frame, the program write the density grid to a _.vdb_ file, then I use Houdini to render the OpenVDB volume, please see [this repository](https://github.com/H-YWu/mpm_data) for details.

## Known Issues

- You may encounter "duplicated reinstall targets" when configuring the project, I simply comment out the reinstall target line in _\_deps/eigen-src/CMakelists.txt_ and reconfigure to address it.
- CUDA OpenGL interop is not supported yet on WSL according to [CUDA on WSL User Guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#features-not-yet-supported), so I implemented CPU-version functions to write particles data for rendering.
- Util the last time I update this repository, CUDA kernels using `Eigen` cannot be compiled by `nvcc` with MS Visual Studio, according to [Using Eigen in CUDA kernels](https://eigen.tuxfamily.org/dox/TopicCUDA.html). However, it might work in the future, so I also configured Windows OS in _CMakeLists.txt_. Hopefully this will be fixed soon!
- Currently there will be some errors if you compile this code with CUDA 12.5.

Please contact the author or create an issue in [this GitHub repository](https://github.com/H-YWu/mpm) if you find any new problem!

## References

\[1\] Alexey Stomakhin, Craig Schroeder, Lawrence Chai, Joseph Teran, and Andrew Selle. 2013. A material point method for snow simulation. ACM Trans. Graph. 32, 4, Article 102 (July 2013), 10 pages. https://doi.org/10.1145/2461912.2461948

\[2\] Chenfanfu Jiang, Craig Schroeder, Joseph Teran, Alexey Stomakhin, and Andrew Selle. 2016. The material point method for simulating continuum materials. In ACM SIGGRAPH 2016 Courses (SIGGRAPH '16). Association for Computing Machinery, New York, NY, USA, Article 24, 1–52. https://doi.org/10.1145/2897826.2927348

\[3\] Chenfanfu Jiang, Craig Schroeder, Andrew Selle, Joseph Teran, and Alexey Stomakhin. 2015. The affine particle-in-cell method. ACM Trans. Graph. 34, 4, Article 51 (August 2015), 10 pages. https://doi.org/10.1145/2766996

\[4\] Sung Tzu-Wei, Yist Lin, and Chen Li-Yu, (2018), GitHub repository, https://github.com/WindQAQ/MPM

\[5\] Kui Wu, and Xinlei Wang, (2018), GitHub repository, https://github.com/kuiwuchn/3x3_SVD_CUDA