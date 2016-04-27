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

double log_gamma(double x, double shape, double scale) {
  return (shape - 1) * log(x) - x / scale - lgamma(shape) - shape * log(scale);
}

double log_beta(double p, double a, double b) {
  double x = (a - 1) * log(p) + (b - 1) * log(1 - p);
  double y = lgamma(a + b) - lgamma(a) - lgamma(b);
  /*
  cout << "p=" << p << " a=" << a << " b=" << b << " x=" << x << " y=" << y
            << " x+y=" << x + y << endl;
  */
  return x + y;
}

double log_beta_odds(double x, double a, double b) {
  return  lgamma(a + b) - lgamma(a) - lgamma(b) + (a-1) * log(x) - (a+b-2) * log(1+x);
}

double log_beta_neg_odds(double x, double a, double b) {
  return  lgamma(a + b) - lgamma(a) - lgamma(b) + (b-1) * log(x) - (a+b-2) * log(1+x);
  return log_beta_odds(x, a, 1/b);
}

double log_negative_binomial(size_t x, double r, double p) {
  return lgamma(x + r) - lgamma(x+1) - lgamma(r) + x * log(p) + r * log(1-p);
}

double log_negative_binomial(size_t x, double r, double p1, double p2) {
  double logp = log(p1 + p2);
  return lgamma(x + r) - lgamma(x+1) - lgamma(r) + x * (log(p1) - logp) + r * (log(p2) - logp);
}
