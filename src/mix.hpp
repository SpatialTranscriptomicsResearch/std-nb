#ifndef MIX_HPP
#define MIX_HPP

#include "io.hpp"
#include "log.hpp"
#include "odds.hpp"
#include "pdist.hpp"
#include "priors.hpp"
#include "sampling.hpp"

namespace PoissonFactorization {
namespace Mix {

enum class Kind {
  Dirichlet,
  Gamma
  //, HierGamma
};

template <Kind kind>
struct Traits {};

template <>
struct Traits<Kind::Gamma> {
  typedef PRIOR::THETA::Gamma prior_type;
};

template <>
struct Traits<Kind::Dirichlet> {
  typedef PRIOR::THETA::Dirichlet prior_type;
};

template <Kind kind>
struct Weights {
  typedef typename Traits<kind>::prior_type prior_type;
  Weights(size_t G_, size_t S_, size_t T_, const Parameters &params)
      : G(G_),
        S(S_),
        T(T_),
        theta(S, T),
        parameters(params),
        prior(G, S, T, parameters) {
    initialize();
  };
  size_t G, S, T;
  Matrix theta;
  Parameters parameters;
  prior_type prior;

  void initialize_factor(size_t t);
  void initialize();
  // template <Features::Kind feat_kind>
  template <typename Features>
  void sample(const Features &features, const IMatrix &contributions_gene_type,
              const Vector &spot_scaling,
              const Vector &experiment_scaling_long);
  void store(const std::string &prefix,
             const std::vector<std::string> &spot_names,
             const std::vector<std::string> &factor_names) const {
    write_matrix(theta, prefix + "theta.txt", spot_names, factor_names);
    prior.store(prefix, spot_names, factor_names);
  };

  double log_likelihood_factor(const IMatrix &counts, size_t t) const;
  // TODO

  void lift_sub_model(const Weights<kind> &sub_model, size_t t1, size_t t2) {
    prior.lift_sub_model(sub_model.prior, t1, t2);
    for (size_t s = 0; s < S; ++s)
      theta(s, t1) = sub_model.theta(s, t2);
  }
};

template <>
void Weights<Kind::Gamma>::initialize_factor(size_t t) {
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
  for (size_t s = 0; s < S; ++s)
    // NOTE: std::gamma_distribution takes a shape and scale parameter
    theta(s, t) = std::gamma_distribution<Float>(
        prior.r(t), 1 / prior.p(t))(EntropySource::rng);
}

template <>
void Weights<Kind::Dirichlet>::initialize_factor(size_t t) {
  // TODO
  std::vector<double> a(S);
  for (size_t s = 0; s < S; ++s)
    a[s] = prior.alpha[s];
  auto x
      = sample_dirichlet<Float>(a, EntropySource::rngs[omp_get_thread_num()]);
  // for (size_t s = 0; s < S; ++s)
  //   phi(g, t) = x[g];
}

template <>
void Weights<Kind::Gamma>::initialize() {
  // initialize Θ
  LOG(debug) << "Initializing Θ from Gamma distribution";
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t s = 0; s < S; ++s) {
    const size_t thread_num = omp_get_thread_num();
    for (size_t t = 0; t < T; ++t)
      // NOTE: std::gamma_distribution takes a shape and scale parameter
      theta(s, t) = std::gamma_distribution<Float>(
          prior.r(t), 1 / prior.p(t))(EntropySource::rngs[thread_num]);
  }
}

template <>
void Weights<Kind::Dirichlet>::initialize() {
  LOG(debug) << "Initializing Θ from Dirichlet distribution" << std::endl;
  std::vector<double> a(T);
  for (size_t t = 0; t < T; ++t)
    a[t] = prior.alpha[t];
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t s = 0; s < S; ++s) {
    const size_t thread_num = omp_get_thread_num();
    auto x = sample_dirichlet<Float>(a, EntropySource::rngs[thread_num]);
    for (size_t t = 0; t < T; ++t)
      theta(s, t) = x[t];
  }
}

template <>
// TODO ensure no NaNs or infinities are generated
double Weights<Kind::Gamma>::log_likelihood_factor(const IMatrix &counts,
                                                   size_t t) const {
  double l = 0;

#pragma omp parallel for reduction(+ : l) if (DO_PARALLEL)
  for (size_t s = 0; s < S; ++s) {
    // NOTE: log_gamma takes a shape and scale parameter
    auto cur = log_gamma(theta(s, t), prior.r(t), 1.0 / prior.p(t));
    if (false and cur > 0)
      LOG(debug) << "ll_cur > 0 for (s,t) = (" + std::to_string(s) + ", " + std::to_string(t) + "): " + std::to_string(cur)
        + " theta = " + std::to_string(theta(s,t))
        + " r = " + std::to_string(prior.r(t))
        + " p = " + std::to_string(prior.p(t))
        + " (r - 1) * log(theta) = " + std::to_string((prior.r(t)- 1) * log(theta(s,t)))
        + " - theta / 1/p = " + std::to_string(- theta(s,t) / 1/prior.p(t))
        + " - lgamma(r) = " + std::to_string(- lgamma(prior.r(t)))
        + " - r * log(1/p) = " + std::to_string(- prior.r(t) * log(1/prior.p(t)));
    l += cur;
  }

  // NOTE: log_gamma takes a shape and scale parameter
  l += log_gamma(prior.r(t), parameters.hyperparameters.theta_r_1,
                 1.0 / parameters.hyperparameters.theta_r_2);

  l += log_beta_neg_odds(prior.p(t), parameters.hyperparameters.theta_p_1,
                         parameters.hyperparameters.theta_p_2);

  return l;
}

template <>
// TODO ensure no NaNs or infinities are generated
double Weights<Kind::Dirichlet>::log_likelihood_factor(const IMatrix &counts,
                                                       size_t t) const {
  // TODO
  assert(false);
  std::vector<Float> p(S);
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t s = 0; s < S; ++s)
    p[s] = theta(s, t);

  return log_dirichlet(p, prior.alpha);
}

/** sample theta */
template <>
template <typename Features>
void Weights<Kind::Gamma>::sample(const Features &features,
                                  const IMatrix &contributions_spot_type,
                                  const Vector &spot_scaling,
                                  const Vector &experiment_scaling_long) {
  LOG(info) << "Sampling Θ from Gamma distribution";

  const std::vector<Float> intensities = features.marginalize_genes();

#pragma omp parallel for if (DO_PARALLEL)
  for (size_t s = 0; s < S; ++s) {
    Float scale = spot_scaling[s];
    if (parameters.activate_experiment_scaling)
      scale *= experiment_scaling_long[s];
    for (size_t t = 0; t < T; ++t)
      // NOTE: std::gamma_distribution takes a shape and scale parameter
      theta(s, t) = std::max<Float>(
          std::numeric_limits<Float>::denorm_min(),
          std::gamma_distribution<Float>(
              prior.r[t] + contributions_spot_type(s, t),
              1.0 / (prior.p[t] + intensities[t] * scale))(EntropySource::rng));
  }
  if ((parameters.enforce_mean & ForceMean::Theta) != ForceMean::None)
#pragma omp parallel for if (DO_PARALLEL)
    for (size_t s = 0; s < S; ++s) {
      double z = 0;
      for (size_t t = 0; t < T; ++t)
        z += theta(s, t);
      for (size_t t = 0; t < T; ++t)
        theta(s, t) /= z;
    }
}

template <>
template <typename Features>
void Weights<Kind::Dirichlet>::sample(const Features &features,
                                      const IMatrix &contributions_spot_type,
                                      const Vector &spot_scaling,
                                      const Vector &experiment_scaling_long) {
  LOG(info) << "Sampling Θ from Dirichlet distribution";
  for (size_t s = 0; s < S; ++s) {
    std::vector<Float> a(T, parameters.hyperparameters.alpha);
#pragma omp parallel for if (DO_PARALLEL)
    for (size_t t = 0; t < T; ++t)
      a[t] += contributions_spot_type(s, t);
    auto theta_sample = sample_dirichlet<Float>(a);
    for (size_t t = 0; t < T; ++t)
      theta(s, t) = theta_sample[t];
  }
}
}
}

#endif
