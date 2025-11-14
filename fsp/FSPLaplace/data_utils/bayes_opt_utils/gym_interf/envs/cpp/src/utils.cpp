#include "utils.h"

#include <cmath>

namespace utils {

double dist(const Vector& vector1, const Vector& vector2)
{
    double result2 = 0;
    for (size_t i = 0; i < 3; ++i) {
        result2 += (vector1[i] - vector2[i]) * (vector1[i] - vector2[i]);
    }
    return sqrt(result2);
}

Vector backTrack(
    const Vector& target, const Vector& normalVector, const Vector& center)
{
    const Vector targetCenter = {
        target[0] - center[0], 
        target[1] - center[1], 
        target[2] - center[2]
    };

    const double dotScalar = 
        targetCenter[0] * normalVector[0] + 
        targetCenter[1] * normalVector[1] + 
        targetCenter[2] * normalVector[2];

    return {
        target[0] - dotScalar * normalVector[0],
        target[1] - dotScalar * normalVector[1],
        target[2] - dotScalar * normalVector[2]
    };
}


} // utils
