#pragma once

#include "types.h"

namespace utils {

Vector backTrack(
    const Vector& target, const Vector& normalVector, const Vector& center);

double dist(const Vector& vector1, const Vector& vector2);

} // utils
