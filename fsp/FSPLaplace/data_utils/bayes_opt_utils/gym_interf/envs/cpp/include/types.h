#pragma once

#include <array>
#include <complex>

using Vector = std::array<double, 3>;
using Complex = std::complex<double>;

std::ostream& operator << (std::ostream& out, const Complex& c);
std::ostream& operator << (std::ostream& out, const Vector& v);
