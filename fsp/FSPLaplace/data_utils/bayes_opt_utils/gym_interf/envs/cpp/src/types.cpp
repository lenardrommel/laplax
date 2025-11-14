#include "types.h"

#include <iostream>

std::ostream& operator << (std::ostream& out, const Complex& c)
{
	out << "{ " << std::real(c) << " + i" << std::imag(c) << " }";
}

std::ostream& operator << (std::ostream& out, const Vector& v)
{
	out << "{ " << v[0] << ", " << v[1] << ", " << v[2] << "}";
}
