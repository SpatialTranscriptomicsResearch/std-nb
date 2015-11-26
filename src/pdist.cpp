#include <cmath>
#include <iostream>
#include "pdist.hpp"

using namespace std;

double log_poisson(size_t k, double lambda) {
  return k * log(lambda) - lambda - lgamma(k + 1);
}

double log_dirichlet(const vector<double> &p, const vector<double> &alpha) {
  double l = 0;
  double sum = 0;
  for (auto &a : alpha) {
    l -= lgamma(a);
    sum += a;
  }
  l += lgamma(sum);

  const size_t N = p.size();
  for (size_t n = 0; n < N; ++n) l += (alpha[n] - 1) * log(p[n]);

  return l;
}

double log_gamma(double x, double k, double theta) {
  return (k - 1) * log(x) - x / theta - lgamma(k) - k * log(theta);
}

double log_beta(double x, double a, double b) {
  double y = (a - 1) * log(x) + (b - 1) * log(1 - x);
  double z = lgamma(a + b) - lgamma(a) - lgamma(b);
  /*
  cout << "x=" << x << " a=" << a << " b=" << b << " y=" << y << " z=" << z
            << " y+z=" << y + z << endl;
  */
  return y + z;
}

double log_negative_binomial(double x, double r, double p) {
  return lgamma(x + r) - lgamma(x+1) - lgamma(r) + x * log(p) + r * log(1-p);
}
