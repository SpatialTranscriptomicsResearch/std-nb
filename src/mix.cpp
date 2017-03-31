#include "PartialModel.hpp"
#include "aux.hpp"
#include "odds.hpp"

using namespace std;

namespace STD {

void Theta::initialize_factor(size_t t) {
  // randomly initialize p of Θ
  LOG(debug) << "Initializing P of Θ";
  if (true)  // TODO make this CLI-switchable
    prior.p[t] = prob_to_neg_odds(
        sample_beta<Float>(parameters.hyperparameters.theta_p_1,
                           parameters.hyperparameters.theta_p_2));
  else
    prior.p[t] = 1;

  // initialize r of Θ
  LOG(debug) << "Initializing R of Θ";
  // NOTE: std::gamma_distribution takes a shape and scale parameter
  prior.r[t] = std::gamma_distribution<Float>(
      parameters.hyperparameters.theta_r_1,
      1 / parameters.hyperparameters.theta_r_2)(EntropySource::rng);

  // initialize Θ
  LOG(debug) << "Initializing Θ";
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t s = 0; s < dim1; ++s)
    // NOTE: std::gamma_distribution takes a shape and scale parameter
    matrix(s, t) = std::gamma_distribution<Float>(
        prior.r(t), 1 / prior.p(t))(EntropySource::rng);
}

void Theta::initialize() {
  // initialize Θ
  LOG(debug) << "Initializing Θ from Gamma distribution";
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t s = 0; s < dim1; ++s) {
    const size_t thread_num = omp_get_thread_num();
    for (size_t t = 0; t < dim2; ++t)
      // NOTE: std::gamma_distribution takes a shape and scale parameter
      matrix(s, t) = std::gamma_distribution<Float>(
          prior.r(t), 1 / prior.p(t))(EntropySource::rngs[thread_num]);
  }
}

// TODO ensure no NaNs or infinities are generated
double Theta::log_likelihood() const {
  double l = 0;
  for (size_t t = 0; t < dim2; ++t)
    l += log_likelihood_factor(t);
  return l;
}

// TODO ensure no NaNs or infinities are generated
double Theta::log_likelihood_factor(
    size_t t) const {
  double l = 0;

#pragma omp parallel for reduction(+ : l) if (DO_PARALLEL)
  for (size_t s = 0; s < dim1; ++s) {
    // NOTE: log_gamma takes a shape and scale parameter
    auto cur = log_gamma(matrix(s, t), prior.r(t), 1.0 / prior.p(t));
    if (false and cur > 0)
      LOG(debug) << "ll_cur > 0 for (s,t) = (" << s << ", " << t << "): " << cur
                 << " theta = " << matrix(s, t) << " r = " << prior.r(t)
                 << " p = " << prior.p(t) << " (r - 1) * log(theta) = "
                 << ((prior.r(t) - 1) * log(matrix(s, t)))
                 << " - theta / 1/p = " << (-matrix(s, t) / 1 / prior.p(t))
                 << " - lgamma(r) = " << (-lgamma(prior.r(t)))
                 << " - r * log(1/p) = " << (-prior.r(t) * log(1 / prior.p(t)));
    l += cur;
  }

  /*
   * NOTE deactivated due to removal of parameters.respect_theta_prior_likelihood
  if (parameters.respect_theta_prior_likelihood)
    // NOTE: log_gamma takes a shape and scale parameter
    l += log_gamma(prior.r(t), parameters.hyperparameters.theta_r_1,
        1.0 / parameters.hyperparameters.theta_r_2);

  if (parameters.respect_theta_prior_likelihood)
    l += log_beta_neg_odds(prior.p(t), parameters.hyperparameters.theta_p_1,
        parameters.hyperparameters.theta_p_2);
  */

  return l;
}
}
