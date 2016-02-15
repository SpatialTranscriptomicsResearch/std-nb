#ifndef FACTORANALYSIS_HPP
#define FACTORANALYSIS_HPP

#include <cstdint>
#include <vector>
#include "types.hpp"

namespace FactorAnalysis {
struct Priors {
  Priors(Float phi_r_1_ = 10, Float phi_r_2_ = 10, Float phi_p_1_ = 2,
         Float phi_p_2_ = 2, Float gamma_ = 1, Float alpha_ = 0.5)
      : phi_r_1(phi_r_1_),
        phi_r_2(phi_r_2_),
        phi_p_1(phi_p_1_),
        phi_p_2(phi_p_2_),
        gamma(gamma_),
        alpha(alpha_){};

  // priors for the gamma distribution of r[g][t]
  // Float c0;
  // Float r0;
  Float phi_r_1;
  Float phi_r_2;

  // priors for the gamma distribution of p[g][t]
  // Float c;
  // Float epsilon;
  Float phi_p_1;
  Float phi_p_2;

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
  /** Whether to enforce the means of the scaling variables to be unity */
  bool enforce_means = false;
};

Float digamma(Float x);
Float trigamma(Float x);

template <typename T>
T square(T x) {
  return x * x;
}
}
#endif
