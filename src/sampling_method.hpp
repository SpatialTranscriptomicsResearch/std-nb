#ifndef SAMPLING_METHOD_HPP
#define SAMPLING_METHOD_HPP

#include <iostream>

namespace Sampling {

enum class Method {
  Mean = 0,
  Multinomial = 1 << 0,
  MH = 1 << 1,
  HMC = 1 << 2,
  RPROP = 1 << 3,
  Trial = 1 << 4,
  TrialMean = 1 << 5,
  lBFGS = 1 << 6,
};

std::ostream &operator<<(std::ostream &os, const Method &which);
std::istream &operator>>(std::istream &is, Method &which);
}

#endif
