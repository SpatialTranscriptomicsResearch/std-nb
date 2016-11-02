#ifndef MIX_HPP
#define MIX_HPP

#include "aux.hpp"
#include "io.hpp"
#include "log.hpp"
#include "parallel.hpp"
#include "pdist.hpp"
#include "priors.hpp"
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
  Model(size_t dim1_, size_t dim2_, const Parameters &params)
      : dim1(dim1_),
        dim2(dim2_),
        matrix(dim1, dim2),
        parameters(params),
        prior(dim1, dim2, parameters) {
    initialize();
  };
  size_t dim1, dim2;
  Matrix matrix;
  Parameters parameters;
  prior_type prior;

  void initialize_factor(size_t t);
  void initialize();

  template <typename Experiment, typename... Args>
  void sample(const Experiment &experiment, const Args &... args);

  template <typename Experiment, typename... Args>
  void sample_field(const Experiment &experiment, Matrix &field,
                    const Args &... args);

  std::string gen_path_stem(const std::string &prefix) const {
    return prefix + to_lower(to_string(variable) + "-" + to_string(kind));
  };

  void store(const std::string &prefix,
             const std::vector<std::string> &spot_names,
             const std::vector<std::string> &factor_names,
             const std::vector<size_t> &order) const {
    const auto path = gen_path_stem(prefix);
    write_matrix(matrix, path + FILENAME_ENDING, spot_names, factor_names, order);
    prior.store(path, spot_names, factor_names, order);
  };

  double log_likelihood_factor(size_t t) const;
  double log_likelihood() const;
};

template <typename Type, typename Res>
void perform_sampling(const Type &observed, const Type &explained, Res &m,
                      bool over_relax) {
#pragma omp parallel if (DO_PARALLEL)
  {
    const size_t thread_num = omp_get_thread_num();
    if (not over_relax) {
#pragma omp for
      for (size_t x = 0; x < observed.n_elem; ++x)
        // NOTE: gamma_distribution takes a shape and scale parameter
        m[x] = std::gamma_distribution<Float>(
            observed[x], 1.0 / explained[x])(EntropySource::rngs[thread_num]);
    } else {
#pragma omp for
      for (size_t x = 0; x < observed.n_elem; ++x) {
        // NOTE: gamma_cdf takes a shape and scale parameter
        double u = gamma_cdf(m[x], observed[x], 1.0 / explained[x]);
        const size_t K = 20;  // TODO introduce CLI switch
        double u_prime = u;
        size_t r = std::binomial_distribution<size_t>(
            K, u)(EntropySource::rngs[thread_num]);
        if (r > K - r)
          u_prime = u * sample_beta<Float>(K - r + 1, 2 * r - K,
                                           EntropySource::rngs[thread_num]);
        else if (r < K - r)
          u_prime
              = 1
                - (1 - u) * sample_beta<Float>(r + 1, K - 2 * r,
                                               EntropySource::rngs[thread_num]);
        u_prime = 0.5 + (u_prime - 0.5) * 0.95;
        /*
        double prev = m[x];
        std::cerr << prev << " " << observed[x] << " " << explained[x] << " "
                  << u << " " << u_prime << " -> " << std::flush;
                  */

        // NOTE: inverse_gamma_cdf takes a shape and scale parameter
        m[x] = inverse_gamma_cdf(u_prime, observed[x], 1.0 / explained[x]);
        /*
        std::cerr << m[x] << std::endl;
        */
      }
    }
  }
}

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
    size_t t) const;

template <>
double Model<Variable::Feature, Kind::Dirichlet>::log_likelihood_factor(
    size_t t) const;

/** sample phi */
template <>
template <typename Experiment, typename... Args>
void Model<Variable::Feature, Kind::Gamma>::sample(const Experiment &experiment,
                                                   const Args &... args) {
  LOG(verbose) << "Sampling Φ from Gamma distribution";

  Matrix observed = prior.r + experiment.contributions_gene_type;
  Matrix explained = prior.p + experiment.explained_gene_type(args...);

  perform_sampling(observed, explained, matrix, parameters.over_relax);
}

/* TODO reactivate
template <>
template <typename M>
void Model<Variable::Feature, Kind::Dirichlet>::sample(
    const M &mix, const Matrix &contributions_gene_type, const Vector &spot,
    const Vector &experiment, const Matrix &other) {
  LOG(verbose) << "Sampling Φ from Dirichlet distribution";
  for (size_t t = 0; t < dim2; ++t) {
    std::vector<Float> a(G, 0);
#pragma omp parallel for if (DO_PARALLEL)
    for (size_t g = 0; g < G; ++g)
      a[g] = prior.alpha(g, t) + contributions_gene_type(g, t);
    auto phi_k = sample_dirichlet<Float>(begin(a), end(a));
    for (size_t g = 0; g < G; ++g)
      matrix(g, t) = phi_k[g];
  }
}
*/

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
    size_t t) const;

template <>
double Model<Variable::Mix, Kind::Dirichlet>::log_likelihood_factor(
    size_t t) const;

/** sample theta */
template <>
template <typename Experiment, typename... Args>
void Model<Variable::Mix, Kind::HierGamma>::sample_field(
    const Experiment &experiment, Matrix &field, const Args &... args) {
  LOG(verbose) << "Sampling Θ from Gamma distribution";

  const bool convolve = parameters.targeted(Target::field);
  const double FACTOR = 100;

  const auto intensities = experiment.marginalize_genes(args...);

  Matrix observed = experiment.contributions_spot_type;
  Matrix explained = experiment.spot * intensities.t();
  if (convolve) {
    explained %= field;
    observed += FACTOR;
    explained += FACTOR;
  } else {
    observed.each_row() += prior.r.t();
    explained.each_row() += prior.p.t();
  }
  perform_sampling(observed, explained, matrix, parameters.over_relax);
  if (convolve)
    matrix %= field;

#pragma omp parallel for if (DO_PARALLEL)
  for (size_t s = 0; s < dim1; ++s)
    for (size_t t = 0; t < dim2; ++t)
      matrix(s, t) = std::max<Float>(std::numeric_limits<Float>::denorm_min(),
                                     matrix(s, t));
}

template <>
template <typename Experiment, typename... Args>
void Model<Variable::Mix, Kind::Dirichlet>::sample_field(
    const Experiment &experiment, Matrix &field __attribute__((unused)),
    const Args &... args __attribute__((unused))) {
  // TODO needs written-down proof; it's analogous to the case for the features
  LOG(verbose) << "Sampling Θ from Dirichlet distribution";
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t s = 0; s < dim1; ++s) {
    std::vector<Float> a(dim2, parameters.hyperparameters.alpha);
    for (size_t t = 0; t < dim2; ++t)
      a[t] += experiment.contributions_spot_type(s, t);
    auto theta_sample = sample_dirichlet<Float>(begin(a), end(a));
    for (size_t t = 0; t < dim2; ++t)
      matrix(s, t) = theta_sample[t];
  }
}
}
}

#endif
