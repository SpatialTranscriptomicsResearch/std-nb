#include <cmath>
#include <iostream>
#include "pdist.hpp"

double log_poisson(size_t k, double lambda) {
  return k * log(lambda) - lambda - lgamma(k + 1);
}

double log_gamma(double x, double k, double theta) {
  return (k - 1) * log(x) - x / theta - lgamma(k) - k * log(theta);
}

double log_beta(double x, double a, double b) {
  double y = (a - 1) * log(x) + (b - 1) * log(1 - x);
  double z = lgamma(a + b) - lgamma(a) - lgamma(b);
  /*
  std::cout << "x=" << x << " a=" << a << " b=" << b << " y=" << y << " z=" << z
            << " y+z=" << y + z << std::endl;
  */
  return y + z;
}
