#ifndef MIX_HPP
#define MIX_HPP

#include "io.hpp"
#include "pdist.hpp"
#include "priors.hpp"

#include "log.hpp"
#include "parallel.hpp"
#include "sampling.hpp"

namespace PoissonFactorization {
namespace Mix {

enum class Kind {
  Dirichlet,
  Gamma
  //, HierGamma
};

std::ostream &operator<<(std::ostream &os, Kind kind);
std::istream &operator>>(std::istream &is, Kind &kind);

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
void Weights<Kind::Gamma>::initialize_factor(size_t t);

template <>
void Weights<Kind::Dirichlet>::initialize_factor(size_t t);

template <>
void Weights<Kind::Gamma>::initialize();

template <>
void Weights<Kind::Dirichlet>::initialize();

template <>
double Weights<Kind::Gamma>::log_likelihood_factor(const IMatrix &counts,
                                                   size_t t) const;

template <>
double Weights<Kind::Dirichlet>::log_likelihood_factor(const IMatrix &counts,
                                                       size_t t) const;

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
