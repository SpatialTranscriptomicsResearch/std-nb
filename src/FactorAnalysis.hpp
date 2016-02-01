#ifndef FACTORANALYSIS_HPP
#define FACTORANALYSIS_HPP

#include <cstdint>
#include <vector>
#include "types.hpp"

namespace FactorAnalysis {
struct Priors {
  Priors(Float c_ = 10, Float d_ = 10, Float e_ = 10, Float f_ = 10,
         Float gamma_ = 1, Float alpha_ = 0.5)
      : c(c_), d(d_), e(e_), f(f_), gamma(gamma_), alpha(alpha_){};

  // priors for the gamma distribution of r[g][t]
  // Float c0;
  // Float r0;
  Float c;
  Float d;

  // priors for the gamma distribution of p[g][t]
  // Float c;
  // Float epsilon;
  Float e;
  Float f;

  Float gamma;
  Float alpha;
};

struct Parameters {
  /** Maximal number of propositions for Metropolis-Hastings sampling */
  double n_iter = 100;
  /** Temperature for Metropolis-Hastings sampling of r[g][t] */
  double temperature = 1.0;
  /** Std. dev. for proposition scaling in Metropolis-Hastings sampling */
  double prop_sd = 0.5;
};

Float digamma(Float x);
Float trigamma(Float x);

template <typename T>
T square(T x) {
  return x * x;
}
}
#endif
