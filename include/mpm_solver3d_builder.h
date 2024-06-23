#ifndef CHAINS_MPM_SOLVER3D_BUILDER_H_
#define CHAINS_MPM_SOLVER3D_BUILDER_H_

#include "mpm_solver3d.h"
#include "mpm3d_parser.h"

namespace chains {

MPMSolver3D buildMPMSolver3DFromYAML(MPM3DConfiguration config, int& particlesNumber);

}   // namespace chains

#endif  // CHAINS_MPM_SOLVER3D_BUILDER_H_