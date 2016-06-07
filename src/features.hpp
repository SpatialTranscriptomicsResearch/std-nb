#ifndef FEATURES_HPP
#define FEATURES_HPP

#include "io.hpp"
#include "log.hpp"
#include "odds.hpp"
#include "pdist.hpp"
#include "priors.hpp"
#include "sampling.hpp"

namespace PoissonFactorization {
namespace Feature {

const Float phi_scaling = 1.0;

enum class Kind {
  Dirichlet,
  Gamma
  //, HierGamma
};

template <Kind kind>
struct Traits {};

template <>
struct Traits<Kind::Gamma> {
  typedef PRIOR::PHI::Gamma prior_type;
};

template <>
struct Traits<Kind::Dirichlet> {
  typedef PRIOR::PHI::Dirichlet prior_type;
};

template <Kind kind>
struct Features {
  typedef typename Traits<kind>::prior_type prior_type;
  Features(size_t G_, size_t S_, size_t T_, const Parameters &params)
      : G(G_),
        S(S_),
        T(T_),
        phi(G, T),
        parameters(params),
        prior(G, S, T, parameters) {
    initialize();
  };
  size_t G, S, T;
  Matrix phi;
  Parameters parameters;
  prior_type prior;

  void initialize_factor(size_t t);
  void initialize();
  void sample(const Matrix &theta, const IMatrix &contributions_gene_type,
              const Vector &spot_scaling,
              const Vector &experiment_scaling_long);
  void store(const std::string &prefix,
             const std::vector<std::string> &gene_names,
             const std::vector<std::string> &factor_names) const {
    write_matrix(phi, prefix + "phi.txt", gene_names, factor_names);
    prior.store(prefix, gene_names, factor_names);
  };

  double log_likelihood_factor(const IMatrix &counts, size_t t) const;
  std::vector<Float> marginalize_genes() const {
    std::vector<Float> intensities(T, 0);
#pragma omp parallel for if (DO_PARALLEL)
    for (size_t t = 0; t < T; ++t)
      for (size_t g = 0; g < G; ++g)
        intensities[t] += phi(g, t);
    return intensities;
  };

  void lift_sub_model(const Features<kind> &sub_model, size_t t1, size_t t2) {
    prior.lift_sub_model(sub_model.prior, t1, t2);
    for (size_t g = 0; g < G; ++g)
      phi(g, t1) = sub_model.phi(g, t2);
  }
};

template <>
void Features<Kind::Gamma>::initialize_factor(size_t t) {
  // initialize p of Φ
  LOG(debug) << "Initializing P of Φ";
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t g = 0; g < G; ++g)
    prior.p(g, t) = prob_to_neg_odds(sample_beta<Float>(
        parameters.hyperparameters.phi_p_1, parameters.hyperparameters.phi_p_2,
        EntropySource::rngs[omp_get_thread_num()]));

  // initialize r of Φ
  LOG(debug) << "Initializing R of Φ";
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t g = 0; g < G; ++g)
    // NOTE: std::gamma_distribution takes a shape and scale parameter
    prior.r(g, t) = std::gamma_distribution<Float>(
        parameters.hyperparameters.phi_r_1,
        1 / parameters.hyperparameters.phi_r_2)(
        EntropySource::rngs[omp_get_thread_num()]);

  // initialize Φ
  LOG(debug) << "Initializing Φ";
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t g = 0; g < G; ++g)
    // NOTE: std::gamma_distribution takes a shape and scale parameter
    phi(g, t)
        = std::gamma_distribution<Float>(prior.r(g, t), 1 / prior.p(g, t))(
            EntropySource::rngs[omp_get_thread_num()]);
}

template <>
void Features<Kind::Dirichlet>::initialize_factor(size_t t) {
  std::vector<double> a(G);
  for (size_t g = 0; g < G; ++g)
    a[g] = prior.alpha[g];
  auto x
      = sample_dirichlet<Float>(a, EntropySource::rngs[omp_get_thread_num()]);
  for (size_t g = 0; g < G; ++g)
    phi(g, t) = x[g];
}

template <>
void Features<Kind::Gamma>::initialize() {
  LOG(debug) << "Initializing Φ from Gamma distribution";
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t g = 0; g < G; ++g) {
    const size_t thread_num = omp_get_thread_num();
    for (size_t t = 0; t < T; ++t)
      // NOTE: gamma_distribution takes a shape and scale parameter
      phi(g, t) = std::gamma_distribution<Float>(
          prior.r(g, t), 1 / prior.p(g, t))(EntropySource::rngs[thread_num]);
  }
}

template <>
void Features<Kind::Dirichlet>::initialize() {
  LOG(debug) << "Initializing Φ from Dirichlet distribution";
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t t = 0; t < T; ++t)
    initialize_factor(t);
}

template <>
// TODO ensure no NaNs or infinities are generated
double Features<Kind::Gamma>::log_likelihood_factor(const IMatrix &counts,
    size_t t) const {
  double l = 0;

#pragma omp parallel for reduction(+ : l) if (DO_PARALLEL)
  for (size_t g = 0; g < G; ++g)
    // NOTE: log_gamma takes a shape and scale parameter
    l += log_gamma(phi(g, t), prior.r(g, t), 1.0 / prior.p(g, t));

#pragma omp parallel for reduction(+ : l) if (DO_PARALLEL)
  for (size_t g = 0; g < G; ++g)
    // NOTE: log_gamma takes a shape and scale parameter
    l += log_gamma(prior.r(g, t), parameters.hyperparameters.phi_r_1,
                   1.0 / parameters.hyperparameters.phi_r_2);

#pragma omp parallel for reduction(+ : l) if (DO_PARALLEL)
  for (size_t g = 0; g < G; ++g)
    l += log_beta_neg_odds(prior.p(g, t), parameters.hyperparameters.phi_p_1,
                           parameters.hyperparameters.phi_p_2);

  return l;
}

template <>
// TODO ensure no NaNs or infinities are generated
double Features<Kind::Dirichlet>::log_likelihood_factor(const IMatrix &counts,
                                                        size_t t) const {
  std::vector<Float> p(G);
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t g = 0; g < G; ++g)
    p[g] = phi(g, t);

  return log_dirichlet(p, prior.alpha);
}

/** sample phi */
template <>
void Features<Kind::Gamma>::sample(const Matrix &theta,
                                   const IMatrix &contributions_gene_type,
                                   const Vector &spot_scaling,
                                   const Vector &experiment_scaling_long) {
  LOG(info) << "Sampling Φ from Gamma distribution";

  // pre-computation
  Vector theta_t(T, arma::fill::zeros);
  for (size_t s = 0; s < S; ++s) {
    Float prod = spot_scaling[s];
    if (parameters.activate_experiment_scaling)
      prod *= experiment_scaling_long[s];
    for (size_t t = 0; t < T; ++t)
      theta_t[t] += theta(s, t) * prod;
  }

// main step: sampling
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t g = 0; g < G; ++g) {
    const size_t thread_num = omp_get_thread_num();
    for (size_t t = 0; t < T; ++t)
      // NOTE: gamma_distribution takes a shape and scale parameter
      phi(g, t) = std::gamma_distribution<Float>(
          prior.r(g, t) + contributions_gene_type(g, t),
          1.0 / (prior.p(g, t) + theta_t[t]))(EntropySource::rngs[thread_num]);
  }

  // enforce means if necessary
  if ((parameters.enforce_mean & ForceMean::Phi) != ForceMean::None)
    for (size_t t = 0; t < T; ++t) {
      double z = 0;
#pragma omp parallel for reduction(+ : z) if (DO_PARALLEL)
      for (size_t g = 0; g < G; ++g)
        z += phi(g, t);
#pragma omp parallel for if (DO_PARALLEL)
      for (size_t g = 0; g < G; ++g)
        phi(g, t) = phi(g, t) / z * phi_scaling;
    }
}

template <>
void Features<Kind::Dirichlet>::sample(const Matrix &theta,
                                       const IMatrix &contributions_gene_type,
                                       const Vector &spot_scaling,
                                       const Vector &experiment_scaling_long) {
  LOG(info) << "Sampling Φ from Dirichlet distribution";
  for (size_t t = 0; t < T; ++t) {
    std::vector<Float> a(G, parameters.hyperparameters.alpha);
#pragma omp parallel for if (DO_PARALLEL)
    for (size_t g = 0; g < G; ++g)
      a[g] += contributions_gene_type(g, t);
    auto phi_k = sample_dirichlet<Float>(a);
    for (size_t g = 0; g < G; ++g)
      phi(g, t) = phi_k[g];
  }
}
}
}

#endif
