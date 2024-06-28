#ifndef CHAINS_ENUMS_H_
#define CHAINS_ENUMS_H_

namespace chains {

enum class InterpolationType {
    LINEAR,
    QUADRATIC_BSPLINE,
    CUBIC_BSPLINE
};

enum class ConstitutiveModel {
    FIXED_COROTATED,
    NEO_HOOKEAN
}; 

enum class IntegrationType {
    EXPLICIT,
    SEMI_IMPLICIT,
    FULL_IMPLICIT
};


} // namespace chains

#endif  // CHAINS_ENUMS_H_