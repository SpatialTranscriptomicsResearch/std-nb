#ifndef FACTORANALYSIS_HPP
#define FACTORANALYSIS_HPP

#include <cstdint>
#include <vector>
#include "types.hpp"

namespace FactorAnalysis {
struct Priors {
  Priors(Float phi_r_1_ = 10, Float phi_r_2_ = 10, Float phi_p_1_ = 2,
         Float phi_p_2_ = 2, Float theta_r_1_ = 1, Float theta_r_2_ = 1,
         Float theta_p_1_ = 0.05, Float theta_p_2_ = 0.95, Float spot_a_ = 10,
         Float spot_b_ = 10, Float experiment_a_ = 10, Float experiment_b_ = 10,
         Float gamma_ = 1, Float alpha_ = 0.5)
      : phi_r_1(phi_r_1_),
        phi_r_2(phi_r_2_),
        phi_p_1(phi_p_1_),
        phi_p_2(phi_p_2_),
        theta_r_1(theta_r_1_),
        theta_r_2(theta_r_2_),
        theta_p_1(theta_p_1_),
        theta_p_2(theta_p_2_),
        spot_a(spot_a_),
        spot_b(spot_b_),
        experiment_a(experiment_a_),
        experiment_b(experiment_b_),
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
  //
  // priors for the gamma distribution of r[t]
  Float theta_r_1;
  Float theta_r_2;

  // priors for the gamma distribution of p[t]
  Float theta_p_1;
  Float theta_p_2;

  Float spot_a;
  Float spot_b;
  Float experiment_a;
  Float experiment_b;

  Float alpha;
};

struct Parameters {
  /** Maximal number of propositions for Metropolis-Hastings sampling */
  double n_iter = 100;
  /** Temperature for Metropolis-Hastings sampling of r[g][t] */
  double temperature = 1.0;
  /** Std. dev. for proposition scaling in Metropolis-Hastings sampling */
  double prop_sd = 0.5;
  /** For which random variables should we enforce the means? */
  enum class ForceMean {
    None = 0,
    Theta = 1,
    Phi = 2,
    Spot = 4,
    Experiment = 8
  };
  ForceMean enforce_mean;
};

inline constexpr Parameters::ForceMean operator&(Parameters::ForceMean x,
                                                 Parameters::ForceMean y) {
  return static_cast<Parameters::ForceMean>(static_cast<int>(x) &
                                            static_cast<int>(y));
}

inline constexpr Parameters::ForceMean operator|(Parameters::ForceMean x,
                                                 Parameters::ForceMean y) {
  return static_cast<Parameters::ForceMean>(static_cast<int>(x) |
                                            static_cast<int>(y));
}

inline Parameters::ForceMean &operator&=(Parameters::ForceMean &x,
                                         Parameters::ForceMean y) {
  x = x & y;
  return x;
}

inline Parameters::ForceMean &operator|=(Parameters::ForceMean &x,
                                         Parameters::ForceMean y) {
  x = x | y;
  return x;
}

std::istream &operator>>(std::istream &is, Parameters::ForceMean &force);
std::ostream &operator<<(std::ostream &os, const Parameters::ForceMean &force);

Float digamma(Float x);
Float trigamma(Float x);

template <typename T>
T square(T x) {
  return x * x;
}
}
#endif
