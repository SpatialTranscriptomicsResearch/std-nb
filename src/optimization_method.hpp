#ifndef OPTIMIZATION_METHOD_HPP
#define OPTIMIZATION_METHOD_HPP

#include <iostream>

namespace Optimize {

enum class Method {
  Gradient = 0,
  RPROP = 1 << 0,
  lBFGS = 1 << 1,
  AdaGrad = 1 << 2,
  Adam = 1 << 3,
};

std::ostream &operator<<(std::ostream &os, const Method &which);
std::istream &operator>>(std::istream &is, Method &which);
}

#endif
