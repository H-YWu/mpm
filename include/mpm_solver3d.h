#ifndef CHAINS_MPM_SOLVER_3D_H_
#define CHAINS_MPM_SOLVER_3D_H_

namespace chains {

class MPMSolver3D {
public:
    MPMSolver3D();
    ~MPMSolver3D();

    void step();

};

}   // namespace chains

#endif  // CHAINS_MPM_SOLVER_3D_H_