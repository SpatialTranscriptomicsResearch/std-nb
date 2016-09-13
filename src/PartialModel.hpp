#ifndef MIX_HPP
#define MIX_HPP

#include "aux.hpp"
#include "io.hpp"
#include "pdist.hpp"
#include "priors.hpp"

#include "log.hpp"
#include "parallel.hpp"
#include "sampling.hpp"

namespace PoissonFactorization {
namespace Partial {

const Float phi_scaling = 1.0;

enum class Variable { Feature, Mix, Spot, Experiment };

std::string to_string(Variable variable);
std::ostream &operator<<(std::ostream &os, Variable variable);
std::istream &operator>>(std::istream &is, Variable &variable);

enum class Kind { Constant, Dirichlet, Gamma, HierGamma };

std::string to_string(Kind kind);
std::ostream &operator<<(std::ostream &os, Kind kind);
std::istream &operator>>(std::istream &is, Kind &kind);

template <Variable variable, Kind kind>
struct Traits {};

template <>
struct Traits<Variable::Feature, Kind::Gamma> {
  typedef PRIOR::PHI::Gamma prior_type;
};

template <>
struct Traits<Variable::Feature, Kind::Dirichlet> {
  typedef PRIOR::PHI::Dirichlet prior_type;
};

template <>
struct Traits<Variable::Mix, Kind::HierGamma> {
  typedef PRIOR::THETA::Gamma prior_type;
};

template <>
struct Traits<Variable::Mix, Kind::Dirichlet> {
  typedef PRIOR::THETA::Dirichlet prior_type;
};

template <Variable variable, Kind kind>
struct Model {
  typedef typename Traits<variable, kind>::prior_type prior_type;
  Model(size_t G_, size_t S_, size_t T_, const Parameters &params)
      : G(G_),
        S(S_),
        T(T_),
        // different nr. of rows for features and mixing weights;
        // initialize construct this
        // matrix(X, T),
        parameters(params),
        prior(G, S, T, parameters) {
    initialize();
  };
  size_t G, S, T;
  Matrix matrix;
  Parameters parameters;
  prior_type prior;

  void initialize_factor(size_t t);
  void initialize();
  // template <Features::Kind feat_kind>
  template <typename M>
  void sample(const M &partmodel, const IMatrix &contributions,
              const Vector &spot, const Vector &experiment);

  std::string gen_path_stem(const std::string &prefix) const {
    return prefix + to_lower(to_string(variable) + "-" + to_string(kind));
  };

  void store(const std::string &prefix,
             const std::vector<std::string> &spot_names,
             const std::vector<std::string> &factor_names) const {
    const auto path = gen_path_stem(prefix);
    write_matrix(matrix, path + ".txt", spot_names, factor_names);
    prior.store(path, spot_names, factor_names);
  };

  double log_likelihood_factor(const IMatrix &counts, size_t t) const;
  double log_likelihood(const IMatrix &counts) const;
  // TODO

  void lift_sub_model(const Model<variable, kind> &sub_model, size_t t1,
                      size_t t2) {
    prior.lift_sub_model(sub_model.prior, t1, t2);
    for (size_t x = 0; x < matrix.n_rows; ++x)
      matrix(x, t1) = sub_model.matrix(x, t2);
  }
};

// Feature specializations

template <>
void Model<Variable::Feature, Kind::Gamma>::initialize_factor(size_t t);

template <>
void Model<Variable::Feature, Kind::Dirichlet>::initialize_factor(size_t t);

template <>
void Model<Variable::Feature, Kind::Gamma>::initialize();

template <>
void Model<Variable::Feature, Kind::Dirichlet>::initialize();

template <>
double Model<Variable::Feature, Kind::Gamma>::log_likelihood_factor(
    const IMatrix &counts, size_t t) const;

template <>
double Model<Variable::Feature, Kind::Dirichlet>::log_likelihood_factor(
    const IMatrix &counts, size_t t) const;

template <Kind kind>
Vector marginalize_spots(const Model<Variable::Mix, kind> &mix,
                         const Vector &spot, const Vector &experiment) {
  Vector theta_t(mix.T, arma::fill::zeros);
  // TODO improve parallelism
  for (size_t s = 0; s < mix.S; ++s) {
    Float prod = spot[s];
    if (mix.parameters.activate_experiment_scaling)
      prod *= experiment[s];
    for (size_t t = 0; t < mix.T; ++t)
      theta_t[t] += mix.matrix(s, t) * prod;
  }
  return theta_t;
}

/** sample phi */
template <>
template <typename M>
void Model<Variable::Feature, Kind::Gamma>::sample(
    const M &mix, const IMatrix &contributions_gene_type, const Vector &spot,
    const Vector &experiment) {
  LOG(info) << "Sampling Φ from Gamma distribution";

  // pre-computation
  Vector theta_t = marginalize_spots(mix, spot, experiment);

// main step: sampling
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t g = 0; g < G; ++g) {
    const size_t thread_num = omp_get_thread_num();
    for (size_t t = 0; t < T; ++t)
      // NOTE: gamma_distribution takes a shape and scale parameter
      matrix(g, t) = std::gamma_distribution<Float>(
          prior.r(g, t) + contributions_gene_type(g, t),
          1.0 / (prior.p(g, t) + theta_t[t]))(EntropySource::rngs[thread_num]);
  }

  // enforce means if necessary
  if ((parameters.enforce_mean & ForceMean::Phi) != ForceMean::None)
    for (size_t t = 0; t < T; ++t) {
      double z = 0;
#pragma omp parallel for reduction(+ : z) if (DO_PARALLEL)
      for (size_t g = 0; g < G; ++g)
        z += matrix(g, t);
#pragma omp parallel for if (DO_PARALLEL)
      for (size_t g = 0; g < G; ++g)
        matrix(g, t) = matrix(g, t) / z * phi_scaling;
    }
}

template <>
template <typename M>
void Model<Variable::Feature, Kind::Dirichlet>::sample(
    const M &mix, const IMatrix &contributions_gene_type, const Vector &spot,
    const Vector &experiment) {
  LOG(info) << "Sampling Φ from Dirichlet distribution";
  for (size_t t = 0; t < T; ++t) {
    std::vector<Float> a(G, 0);
#pragma omp parallel for if (DO_PARALLEL)
    for (size_t g = 0; g < G; ++g)
      a[g] = prior.alpha(g,t) + contributions_gene_type(g, t);
    auto phi_k = sample_dirichlet<Float>(begin(a), end(a));
    for (size_t g = 0; g < G; ++g)
      matrix(g, t) = phi_k[g];
  }
}

// Mixing specializations

template <>
void Model<Variable::Mix, Kind::HierGamma>::initialize_factor(size_t t);

template <>
void Model<Variable::Mix, Kind::Dirichlet>::initialize_factor(size_t t);

template <>
void Model<Variable::Mix, Kind::HierGamma>::initialize();

template <>
void Model<Variable::Mix, Kind::Dirichlet>::initialize();

template <>
double Model<Variable::Mix, Kind::HierGamma>::log_likelihood_factor(
    const IMatrix &counts, size_t t) const;

template <>
double Model<Variable::Mix, Kind::Dirichlet>::log_likelihood_factor(
    const IMatrix &counts, size_t t) const;

template <Kind kind>
std::vector<Float> marginalize_genes(
    const Model<Variable::Feature, kind> &features) {
  std::vector<Float> intensities(features.T, 0);
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t t = 0; t < features.T; ++t)
    for (size_t g = 0; g < features.G; ++g)
      intensities[t] += features.matrix(g, t);
  return intensities;
};

/** sample theta */
template <>
template <typename M>
void Model<Variable::Mix, Kind::HierGamma>::sample(
    const M &features, const IMatrix &contributions_spot_type,
    const Vector &spot, const Vector &experiment) {
  LOG(info) << "Sampling Θ from Gamma distribution";

  const std::vector<Float> intensities = marginalize_genes(features);

#pragma omp parallel for if (DO_PARALLEL)
  for (size_t s = 0; s < S; ++s) {
    Float scale = spot[s];
    if (parameters.activate_experiment_scaling)
      scale *= experiment[s];
    for (size_t t = 0; t < T; ++t)
      // NOTE: std::gamma_distribution takes a shape and scale parameter
      matrix(s, t) = std::max<Float>(
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
        z += matrix(s, t);
      for (size_t t = 0; t < T; ++t)
        matrix(s, t) /= z;
    }
}

template <>
template <typename M>
void Model<Variable::Mix, Kind::Dirichlet>::sample(
    const M &features, const IMatrix &contributions_spot_type,
    const Vector &spot, const Vector &experiment) {
  // TODO needs written-down proof; it's analogous to the case for the features
  LOG(info) << "Sampling Θ from Dirichlet distribution";
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t s = 0; s < S; ++s) {
    std::vector<Float> a(T, parameters.hyperparameters.alpha);
    for (size_t t = 0; t < T; ++t)
      a[t] += contributions_spot_type(s, t);
    auto theta_sample = sample_dirichlet<Float>(begin(a), end(a));
    for (size_t t = 0; t < T; ++t)
      matrix(s, t) = theta_sample[t];
  }
}
}
}

#endif
