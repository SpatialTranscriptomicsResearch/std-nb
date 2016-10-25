#ifndef PARAMETERS_HPP
#define PARAMETERS_HPP

#include <cstdint>
#include "types.hpp"
#include "target.hpp"

namespace PoissonFactorization {
struct Hyperparameters {
  Hyperparameters(Float phi_r_1_ = 10, Float phi_r_2_ = 10, Float phi_p_1_ = 2,
                  Float phi_p_2_ = 2, Float theta_r_1_ = 1,
                  Float theta_r_2_ = 1, Float theta_p_1_ = 0.05,
                  Float theta_p_2_ = 0.95, Float spot_a_ = 10,
                  Float spot_b_ = 10, Float alpha_ = 0.5, Float sigma_ = 1)
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
        alpha(alpha_),
        sigma(sigma_){};

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

  Float alpha;

  // characteristic length-scale for field
  Float sigma;
};

struct Parameters {
  /** Maximal number of propositions for Metropolis-Hastings sampling */
  double n_iter = 100;
  /** Temperature for Metropolis-Hastings sampling of r[g][t] */
  double temperature = 1.0;
  /** How long to enforce certain means or sums
   * 0 means forever
   * anything else is the given number of iterations
   */
  size_t enforce_iter = 10;
  bool phi_prior_maximum_likelihood = false;
  bool respect_phi_prior_likelihood = false;
  bool respect_theta_prior_likelihood = false;
  bool expected_contributions = false;
  bool store_lambda = false;
  bool theta_local_priors = false;
  Hyperparameters hyperparameters;
  Target targets = DefaultTarget();
  bool targeted(Target target) const;
};

Float digamma(Float x);
Float trigamma(Float x);

}
#endif
