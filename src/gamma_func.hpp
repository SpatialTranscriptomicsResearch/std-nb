#ifndef GAMMA_FUNC_HPP
#define GAMMA_FUNC_HPP

#include <cmath>

double digamma(double x);

double trigamma(double x);

inline double lgamma_diff(double a, double b) {
  double s = a + b;
  if (s != a)
    return lgamma(s) - lgamma(a);
  else
    return 0;
}

inline double lgamma_diff_1p(double a, double b) {
  double s = a + b;
  if (s != a + 1)
    return lgamma(s) - lgamma(a + 1);
  else
    return 0;
}

inline double digamma_diff(double a, double b) {
  double s = a + b;
  if (s != a)
    return digamma(s) - digamma(a);
  else
    return 0;
}

inline double digamma_diff_1p(double a, double b) {
  double s = a + b;
  if (s != a + 1)
    return digamma(s) - digamma(a + 1);
  else
    return 0;
}

inline double trigamma_diff(double a, double b) {
  double s = a + b;
  if (s != a)
    return trigamma(s) - trigamma(a);
  else
    return 0;
}

#endif
