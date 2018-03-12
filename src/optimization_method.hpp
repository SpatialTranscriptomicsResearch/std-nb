#ifndef OPTIMIZATION_METHOD_HPP
#define OPTIMIZATION_METHOD_HPP

#include <iostream>

namespace Optimize {

enum class Method {
  Gradient = 0,
  RPROP = 1 << 0,
  AdaGrad = 1 << 1,
  Adam = 1 << 2,
};

std::ostream &operator<<(std::ostream &os, const Method &which);
std::istream &operator>>(std::istream &is, Method &which);
}  // namespace Optimize

#endif
