#ifndef VARIANTMODEL_HPP
#define VARIANTMODEL_HPP

#include <random>
#include "counts.hpp"
#include "entropy.hpp"
#include "FactorAnalysis.hpp"
#include "compression.hpp"
#include "io.hpp"
#include "metropolis_hastings.hpp"
#include "pdist.hpp"
#include "odds.hpp"
#include "sampling.hpp"
#include "stats.hpp"
#include "timer.hpp"
#include "verbosity.hpp"

namespace FactorAnalysis {

#define DO_PARALLEL 1
#define DEFAULT_SEPARATOR "\t"
#define DEFAULT_LABEL ""
const Float phi_scaling = 1.0;
#define print_sub_model_cnt true

const size_t num_sub_gibbs = 50;
const bool consider_factor_likel = false;
const size_t sub_model_cnt = 10;
// static size_t sub_model_cnt; // TODO

bool gibbs_test(Float nextG, Float G, Verbosity verbosity,
                Float temperature = 50);
size_t num_lines(const std::string &path);

double compute_conditional_theta(const std::pair<Float, Float> &x,
                                 const std::vector<Int> &count_sums,
                                 const std::vector<Float> &weight_sums,
                                 const Hyperparameters &hyperparameters);
double compute_conditional(const std::pair<Float, Float> &x, Int count_sum,
                           Float weight_sum,
                           const Hyperparameters &hyperparameters);

enum class GibbsSample {
  empty = 0,
  contributions = 1 << 0,
  phi = 1 << 1,
  phi_r = 1 << 2,
  phi_p = 1 << 3,
  theta = 1 << 4,
  theta_r = 1 << 5,
  theta_p = 1 << 6,
  spot_scaling = 1 << 7,
  experiment_scaling = 1 << 8,
  merge_split = 1 << 9,
};

std::ostream &operator<<(std::ostream &os, const GibbsSample &which);
std::istream &operator>>(std::istream &is, GibbsSample &which);

inline constexpr GibbsSample operator&(GibbsSample a, GibbsSample b) {
  return static_cast<GibbsSample>(static_cast<int>(a) & static_cast<int>(b));
}

inline constexpr GibbsSample operator|(GibbsSample a, GibbsSample b) {
  return static_cast<GibbsSample>(static_cast<int>(a) | static_cast<int>(b));
}

inline constexpr GibbsSample operator^(GibbsSample a, GibbsSample b) {
  return static_cast<GibbsSample>(static_cast<int>(a) & static_cast<int>(b));
}

inline constexpr GibbsSample
operator~(GibbsSample a) {
  return static_cast<GibbsSample>((~static_cast<int>(a)) & ((1 << 10) - 1));
}

inline constexpr GibbsSample DefaultGibbs() {
  return GibbsSample::contributions | GibbsSample::phi | GibbsSample::phi_r
         | GibbsSample::phi_p | GibbsSample::theta | GibbsSample::theta_r
         | GibbsSample::theta_p | GibbsSample::spot_scaling
         | GibbsSample::experiment_scaling | GibbsSample::merge_split;
}

inline bool flagged(GibbsSample x) {
  return (GibbsSample::empty | x) != GibbsSample::empty;
}

struct Paths {
  Paths(const std::string &prefix, const std::string &suffix = "");
  std::string phi, theta, spot, experiment, r_phi, p_phi, r_theta, p_theta;
  std::string contributions_gene_type, contributions_spot_type,
      contributions_spot, contributions_experiment;
};

enum class Kind { Dirichlet, Gamma, HierGamma };

template <Kind kind>
struct KindTraits {};

struct DirichletPrior {
  double alpha;
  // or:
  // Vector alpha;
};

struct GammaPrior {
  Matrix r;
  Matrix p;
};

struct HierGammaPrior {
  Vector r;
  Vector p;
};

template <>
struct KindTraits<Kind::Dirichlet> {
  typedef DirichletPrior Prior;
};

template <>
struct KindTraits<Kind::Gamma> {
  typedef GammaPrior Prior;
};

template <>
struct KindTraits<Kind::HierGamma> {
  typedef HierGammaPrior Prior;
};

template <Kind kind = Kind::Gamma>
struct VariantModel {
  /** number of genes */
  size_t G;
  // const size_t G;
  /** number of samples */
  size_t S;
  // const size_t S;
  /** number of factors */
  size_t T;
  // const size_t T;
  /** number of experiments */
  size_t E;
  // const size_t E;

  Hyperparameters hyperparameters;
  Parameters parameters;

  /** hidden contributions to the count data due to the different factors */
  IMatrix contributions_gene_type, contributions_spot_type;
  IVector contributions_spot, contributions_experiment;

  /** Normalizing factor to translate Poisson rates \lambda_{xgst} to relative
   * frequencies \lambda_{gst} / z_{gs} for the multionomial distribution */
  Matrix lambda_gene_spot;

  /** factor loading matrix */
  Matrix phi;

  /** factor score matrix */
  Matrix theta;

  /** spot scaling vector */
  Vector spot_scaling;

  /** experiment scaling vector */
  Vector experiment_scaling;
  Vector experiment_scaling_long;

  /** shape parameter for the prior of the loading matrix */
  Matrix r_phi;
  /** scale parameter for the prior of the loading matrix */
  /* Stored as negative-odds */
  Matrix p_phi;

  /** shape parameter for the prior of the mixing matrix */
  Vector r_theta;
  /** scale parameter for the prior of the mixing matrix */
  /* Stored as negative-odds */
  Vector p_theta;

  Verbosity verbosity;

  VariantModel(const Counts &counts, const size_t T,
               const Hyperparameters &hyperparameters,
               const Parameters &parameters, Verbosity verbosity);

  VariantModel(const Counts &counts, const Paths &paths,
               const Hyperparameters &hyperparameters,
               const Parameters &parameters, Verbosity verbosity);

  void initialize_r_phi();
  void initialize_p_phi();
  void initialize_phi();
  void initialize_r_theta();
  void initialize_p_theta();
  void initialize_theta();

  void store(const Counts &counts, const std::string &prefix,
             bool mean_and_variance = false) const;

  Matrix weighted_theta() const;

  double log_likelihood(const IMatrix &counts) const;
  double log_likelihood_factor(const IMatrix &counts, size_t t) const;
  double log_likelihood_poisson_counts(const IMatrix &counts) const;

  /** sample count decomposition */
  void sample_contributions(const IMatrix &counts);
  void sample_contributions_sub(const IMatrix &counts, size_t g, size_t s,
                                RNG &rng, IMatrix &contrib_gene_type,
                                IMatrix &contrib_spot_type);

  /** sample phi */
  void sample_phi();
  Float sample_phi_sub(size_t g, size_t t, Float theta_t, RNG &rng) const;

  /** sample phi_p and phi_r */
  void sample_p_and_r();

  /** sample theta */
  std::vector<Float> compute_intensities_gene_type() const;
  void sample_theta();

  /** sample p_theta and r_theta */
  void sample_p_and_r_theta();

  /** sample spot scaling factors */
  void sample_spot_scaling();

  /** sample experiment scaling factors */
  void sample_experiment_scaling(const Counts &data);

  /** sample each of the variables from their conditional posterior */
  void gibbs_sample(const Counts &data, GibbsSample which, bool timing);

  void sample_split_merge(const Counts &data, GibbsSample which);
  void sample_merge(const Counts &data, size_t t1, size_t t2,
                    GibbsSample which);
  void sample_split(const Counts &data, size_t t, GibbsSample which);
  void lift_sub_model(const VariantModel &sub_model, size_t t1, size_t t2);

  VariantModel run_submodel(size_t t, size_t n, const Counts &counts,
                            GibbsSample which, const std::string &prefix,
                            const std::vector<size_t> &init_factors
                            = std::vector<size_t>());

  size_t find_weakest_factor() const;

  std::vector<Int> sample_reads(size_t g, size_t s, size_t n = 1) const;

  double posterior_expectation(size_t g, size_t s) const;
  double posterior_expectation_poisson(size_t g, size_t s) const;
  double posterior_variance(size_t g, size_t s) const;
  Matrix posterior_expectations() const;
  Matrix posterior_expectations_poisson() const;
  Matrix posterior_variances() const;

  /** check that parameter invariants are fulfilled */
  void check_model(const IMatrix &counts) const;

private:
  void update_experiment_scaling_long(const Counts &data);
};

template <Kind kind>
std::ostream &operator<<(std::ostream &os,
                         const FactorAnalysis::VariantModel<kind> &pfa);

template <Kind kind>
VariantModel<kind>::VariantModel(const Counts &c, const size_t T_,
                                 const Hyperparameters &hyperparameters_,
                                 const Parameters &parameters_,
                                 Verbosity verbosity_)
    : G(c.counts.n_rows),
      S(c.counts.n_cols),
      T(T_),
      E(c.experiment_names.size()),
      hyperparameters(hyperparameters_),
      parameters(parameters_),
      contributions_gene_type(G, T, arma::fill::zeros),
      contributions_spot_type(S, T, arma::fill::zeros),
      contributions_spot(S, arma::fill::zeros),
      contributions_experiment(E, arma::fill::zeros),
      lambda_gene_spot(G, S, arma::fill::zeros),
      phi(G, T),
      theta(S, T),
      spot_scaling(S, arma::fill::ones),
      experiment_scaling(E, arma::fill::ones),
      experiment_scaling_long(S, arma::fill::ones),
      r_phi(G, T),
      p_phi(G, T),
      r_theta(T),
      p_theta(T),
      verbosity(verbosity_) {
  initialize_p_phi();
  initialize_r_phi();
  initialize_phi();

  initialize_p_theta();
  initialize_r_theta();
  initialize_theta();

  // initialize:
  //  * contributions_gene_type
  //  * contributions_spot_type
  //  * lambda_gene_spot
  if (verbosity >= Verbosity::Debug)
    std::cout << "initializing contributions." << std::endl;
  sample_contributions(c.counts);

// initialize:
//  * contributions_spot
//  * contributions_experiment
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t s = 0; s < S; ++s)
    for (size_t g = 0; g < G; ++g) {
      contributions_spot(s) += c.counts(g, s);
      contributions_experiment(c.experiments[s]) += c.counts(g, s);
    }

  if (parameters.activate_experiment_scaling) {
    // initialize experiment scaling factors
    if (parameters.activate_experiment_scaling) {
      if (verbosity >= Verbosity::Debug)
        std::cout << "initializing experiment scaling." << std::endl;
      experiment_scaling = Vector(E, arma::fill::zeros);
      for (size_t s = 0; s < S; ++s)
        experiment_scaling(c.experiments[s]) += contributions_spot(s);
      Float z = 0;
      for (size_t e = 0; e < E; ++e)
        z += experiment_scaling(e);
      z /= E;
      for (size_t e = 0; e < E; ++e)
        experiment_scaling(e) /= z;
      // copy the experiment scaling parameters into the spot-indexed vector
      update_experiment_scaling_long(c);
    }
  }

  // initialize spot scaling factors
  {
    if (verbosity >= Verbosity::Debug)
      std::cout << "initializing spot scaling." << std::endl;
    Float z = 0;
    for (size_t s = 0; s < S; ++s) {
      if (verbosity >= Verbosity::Debug)
        std::cout << "z = " << z << " spot_scaling(s) = " << spot_scaling(s)
                  << " contributions_spot(s) = " << contributions_spot(s)
                  << " experiment_scaling_long(s) = "
                  << experiment_scaling_long(s);
      z += spot_scaling(s) = contributions_spot(s) / experiment_scaling_long(s);
      if (verbosity >= Verbosity::Debug)
        std::cout << " spot_scaling(s) = " << spot_scaling(s) << std::endl;
    }
    if (verbosity >= Verbosity::Debug)
      std::cout << "z = " << z << std::endl;
    z /= S;
    if (verbosity >= Verbosity::Debug)
      std::cout << "z = " << z << std::endl;
    for (size_t s = 0; s < S; ++s)
      spot_scaling(s) /= z;
  }
}

template <Kind kind>
void VariantModel<kind>::initialize_p_phi() {
  // initialize p_phi
  if (verbosity >= Verbosity::Debug)
    std::cout << "initializing p_phi." << std::endl;
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t g = 0; g < G; ++g) {
    const size_t thread_num = omp_get_thread_num();
    for (size_t t = 0; t < T; ++t)
      p_phi(g, t) = prob_to_neg_odds(
          sample_beta<Float>(hyperparameters.phi_p_1, hyperparameters.phi_p_2,
                             EntropySource::rngs[thread_num]));
  }
}

template <Kind kind>
void VariantModel<kind>::initialize_r_phi() {
  // initialize r_phi
  if (verbosity >= Verbosity::Debug)
    std::cout << "initializing r_phi." << std::endl;
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t g = 0; g < G; ++g) {
    const size_t thread_num = omp_get_thread_num();
    for (size_t t = 0; t < T; ++t)
      // NOTE: std::gamma_distribution takes a shape and scale parameter
      r_phi(g, t) = std::gamma_distribution<Float>(
          hyperparameters.phi_r_1,
          1 / hyperparameters.phi_r_2)(EntropySource::rngs[thread_num]);
  }
}

template <Kind kind>
void VariantModel<kind>::initialize_phi() {
  // initialize phi
  if (verbosity >= Verbosity::Debug)
    std::cout << "initializing phi." << std::endl;
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t g = 0; g < G; ++g) {
    const size_t thread_num = omp_get_thread_num();
    for (size_t t = 0; t < T; ++t)
      // NOTE: std::gamma_distribution takes a shape and scale parameter
      phi(g, t) = std::gamma_distribution<Float>(
          r_phi(g, t), 1 / p_phi(g, t))(EntropySource::rngs[thread_num]);
  }
}

template <Kind kind>
void VariantModel<kind>::initialize_p_theta() {
  // initialize p_theta
  if (verbosity >= Verbosity::Debug)
    std::cout << "initializing p_theta." << std::endl;
  for (size_t t = 0; t < T; ++t)
    if (false)  // TODO make this CLI-switchable
      p_theta[t] = prob_to_neg_odds(sample_beta<Float>(
          hyperparameters.theta_p_1, hyperparameters.theta_p_2));
    else
      p_theta[t] = 1;
}

template <Kind kind>
void VariantModel<kind>::initialize_r_theta() {
  // initialize r_theta
  if (verbosity >= Verbosity::Debug)
    std::cout << "initializing r_theta." << std::endl;
  for (size_t t = 0; t < T; ++t)
    // NOTE: std::gamma_distribution takes a shape and scale parameter
    r_theta[t] = std::gamma_distribution<Float>(
        hyperparameters.theta_r_1,
        1 / hyperparameters.theta_r_2)(EntropySource::rng);
}

template <Kind kind>
void VariantModel<kind>::initialize_theta() {
  // initialize theta
  if (verbosity >= Verbosity::Debug)
    std::cout << "initializing theta." << std::endl;
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t s = 0; s < S; ++s) {
    const size_t thread_num = omp_get_thread_num();
    for (size_t t = 0; t < T; ++t)
      // NOTE: std::gamma_distribution takes a shape and scale parameter
      theta(s, t) = std::gamma_distribution<Float>(
          r_theta(t), 1 / p_theta(t))(EntropySource::rngs[thread_num]);
  }
}

template <Kind kind>
VariantModel<kind>::VariantModel(const Counts &c, const Paths &paths,
                                 const Hyperparameters &hyperparameters_,
                                 const Parameters &parameters_,
                                 Verbosity verbosity_)
    : G(c.counts.n_rows),
      S(c.counts.n_cols),
      T(num_lines(paths.r_theta)),
      E(c.experiment_names.size()),
      hyperparameters(hyperparameters_),
      parameters(parameters_),
      contributions_gene_type(parse_file<IMatrix>(paths.contributions_gene_type, read_imatrix, DEFAULT_SEPARATOR, DEFAULT_LABEL)),
      contributions_spot_type(parse_file<IMatrix>(paths.contributions_spot_type, read_imatrix, DEFAULT_SEPARATOR, DEFAULT_LABEL)),
      contributions_spot(parse_file<IVector>(paths.contributions_spot, read_vector<IVector>, DEFAULT_SEPARATOR)),
      contributions_experiment(parse_file<IVector>(paths.contributions_experiment, read_vector<IVector>, DEFAULT_SEPARATOR)),
      phi(parse_file<Matrix>(paths.phi, read_matrix, DEFAULT_SEPARATOR, DEFAULT_LABEL)),
      theta(parse_file<Matrix>(paths.theta, read_matrix, DEFAULT_SEPARATOR, DEFAULT_LABEL)),
      spot_scaling(parse_file<Vector>(paths.spot, read_vector<Vector>, DEFAULT_SEPARATOR)),
      experiment_scaling(parse_file<Vector>(paths.experiment, read_vector<Vector>, DEFAULT_SEPARATOR)),
      experiment_scaling_long(S),
      r_phi(parse_file<Matrix>(paths.r_phi, read_matrix, DEFAULT_SEPARATOR, DEFAULT_LABEL)),
      p_phi(parse_file<Matrix>(paths.p_phi, read_matrix, DEFAULT_SEPARATOR, DEFAULT_LABEL)),
      r_theta(parse_file<Vector>(paths.r_theta, read_vector<Vector>, DEFAULT_SEPARATOR)),
      p_theta(parse_file<Vector>(paths.p_theta, read_vector<Vector>, DEFAULT_SEPARATOR)),
      verbosity(verbosity_) {
  update_experiment_scaling_long(c);

  if (verbosity >= Verbosity::Debug)
    std::cout << *this << std::endl;
}

template <Kind kind>
// TODO ensure no NaNs or infinities are generated
double VariantModel<kind>::log_likelihood_factor(const IMatrix &counts,
                                                 size_t t) const {
  double l = 0;

#pragma omp parallel for reduction(+ : l) if (DO_PARALLEL)
  for (size_t g = 0; g < G; ++g)
    // NOTE: log_gamma takes a shape and scale parameter
    l += log_gamma(phi(g, t), r_phi(g, t), 1.0 / p_phi(g, t));

  if (verbosity >= Verbosity::Debug)
    std::cout << "ll_phi = " << l << std::endl;

#pragma omp parallel for reduction(+ : l) if (DO_PARALLEL)
  for (size_t g = 0; g < G; ++g)
    // NOTE: log_gamma takes a shape and scale parameter
    l += log_gamma(r_phi(g, t), hyperparameters.phi_r_1,
                   1.0 / hyperparameters.phi_r_2);

  if (verbosity >= Verbosity::Debug)
    std::cout << "ll_r_phi = " << l << std::endl;

#pragma omp parallel for reduction(+ : l) if (DO_PARALLEL)
  for (size_t g = 0; g < G; ++g)
    l += log_beta_neg_odds(p_phi(g, t), hyperparameters.phi_p_1,
                           hyperparameters.phi_p_2);

  if (verbosity >= Verbosity::Debug)
    std::cout << "ll_p_phi = " << l << std::endl;

#pragma omp parallel for reduction(+ : l) if (DO_PARALLEL)
  for (size_t s = 0; s < S; ++s) {
    // NOTE: log_gamma takes a shape and scale parameter
    auto cur = log_gamma(theta(s, t), r_theta(t), 1.0 / p_theta(t));
    if (false and cur > 0)
      std::cout << "ll_cur > 0 for (s,t) = (" + std::to_string(s) + ", " + std::to_string(t) + "): " + std::to_string(cur)
        + " theta = " + std::to_string(theta(s,t))
        + " r = " + std::to_string(r_theta(t))
        + " p = " + std::to_string(p_theta(t))
        + " (r - 1) * log(theta) = " + std::to_string((r_theta(t)- 1) * log(theta(s,t)))
        + " - theta / 1/p = " + std::to_string(- theta(s,t) / 1/p_theta(t))
        + " - lgamma(r) = " + std::to_string(- lgamma(r_theta(t)))
        + " - r * log(1/p) = " + std::to_string(- r_theta(t) * log(1/p_theta(t)))
        + "\n" << std::flush;
    l += cur;
  }

  if (verbosity >= Verbosity::Debug)
    std::cout << "ll_theta = " << l << std::endl;

  // NOTE: log_gamma takes a shape and scale parameter
  l += log_gamma(r_theta(t), hyperparameters.theta_r_1,
                 1.0 / hyperparameters.theta_r_2);

  if (verbosity >= Verbosity::Debug)
    std::cout << "ll_r_theta = " << l << std::endl;

  l += log_beta_neg_odds(p_theta(t), hyperparameters.theta_p_1,
                         hyperparameters.theta_p_2);

  if (verbosity >= Verbosity::Debug)
    std::cout << "ll_p_theta = " << l << std::endl;

  if (std::isnan(l) or std::isinf(l))
    std::cout << "Warning: log likelihoood contribution of factor " << t
              << " = " << l << std::endl;

  if (verbosity >= Verbosity::Debug)
    std::cout << "ll_X = " << l << std::endl;

  return l;
}

template <Kind kind>
double VariantModel<kind>::log_likelihood_poisson_counts(
    const IMatrix &counts) const {
  double l = 0;
#pragma omp parallel for reduction(+ : l) if (DO_PARALLEL)
  for (size_t g = 0; g < G; ++g)
    for (size_t s = 0; s < S; ++s) {
      auto cur = log_poisson(counts(g, s),
                             lambda_gene_spot(g, s) * spot_scaling(s)
                                 * (parameters.activate_experiment_scaling
                                        ? experiment_scaling_long(s)
                                        : 1));
      if (std::isinf(cur) or std::isnan(cur))
        std::cout << "ll poisson(g=" + std::to_string(g) + ",s="
                         + std::to_string(s) + ") = " + std::to_string(cur)
                         + " counts = " + std::to_string(counts(g, s))
                         + " lambda = " + std::to_string(lambda_gene_spot(g, s))
                         + "\n"
                  << std::flush;
      l += cur;
    }
  return l;
}

template <Kind kind>
double VariantModel<kind>::log_likelihood(const IMatrix &counts) const {
  double l = 0;
  for (size_t t = 0; t < T; ++t)
    l += log_likelihood_factor(counts, t);

  for (size_t s = 0; s < S; ++s)
    l += log_gamma(spot_scaling(s), hyperparameters.spot_a,
                   1.0 / hyperparameters.spot_b);
  if (parameters.activate_experiment_scaling) {
    for (size_t e = 0; e < E; ++e)
      l += log_gamma(experiment_scaling(e), hyperparameters.experiment_a,
                     1.0 / hyperparameters.experiment_b);
  }

  l += log_likelihood_poisson_counts(counts);

  return l;
}

template <Kind kind>
Matrix VariantModel<kind>::weighted_theta() const {
  Matrix m = theta;
  for (size_t t = 0; t < T; ++t) {
    Float x = 0;
    for (size_t g = 0; g < G; ++g)
      x += phi(g, t);
    for (size_t s = 0; s < S; ++s) {
      m(s, t) *= x * spot_scaling(s);
      if (parameters.activate_experiment_scaling)
        m(s, t) *= experiment_scaling_long(s);
    }
  }
  return m;
}

template <Kind kind>
void VariantModel<kind>::store(const Counts &counts, const std::string &prefix,
                               bool mean_and_variance) const {
  std::vector<std::string> factor_names;
  for (size_t t = 1; t <= T; ++t)
    factor_names.push_back("Factor " + std::to_string(t));
  auto &gene_names = counts.row_names;
  auto &spot_names = counts.col_names;
  write_matrix(phi, prefix + "phi.txt", gene_names, factor_names);
  write_matrix(r_phi, prefix + "r_phi.txt", gene_names, factor_names);
  write_matrix(p_phi, prefix + "p_phi.txt", gene_names, factor_names);
  write_matrix(theta, prefix + "theta.txt", spot_names, factor_names);
  write_matrix(weighted_theta(), prefix + "weighted_theta.txt", spot_names, factor_names);
  write_vector(r_theta, prefix + "r_theta.txt", factor_names);
  write_vector(p_theta, prefix + "p_theta.txt", factor_names);
  write_vector(spot_scaling, prefix + "spot_scaling.txt", spot_names);
  write_vector(experiment_scaling, prefix + "experiment_scaling.txt", counts.experiment_names);
  write_matrix(contributions_gene_type, prefix + "contributions_gene_type.txt", gene_names, factor_names);
  write_matrix(contributions_spot_type, prefix + "contributions_spot_type.txt", spot_names, factor_names);
  write_vector(contributions_spot, prefix + "contributions_spot.txt", spot_names);
  write_vector(contributions_experiment, prefix + "contributions_experiment.txt", counts.experiment_names);
  if (mean_and_variance) {
    write_matrix(posterior_expectations(), prefix + "means.txt", gene_names, spot_names);
    write_matrix(posterior_expectations_poisson(), prefix + "means_poisson.txt", gene_names, spot_names);
    write_matrix(posterior_variances(), prefix + "variances.txt", gene_names, spot_names);
  }
}

template <Kind kind>
void VariantModel<kind>::sample_contributions_sub(const IMatrix &counts,
                                                  size_t g, size_t s, RNG &rng,
                                                  IMatrix &contrib_gene_type,
                                                  IMatrix &contrib_spot_type) {
  std::vector<double> rel_rate(T);
  double z = 0;
  // NOTE: in principle, lambda[g][s][t] is proportional to both
  // spot_scaling[s] and experiment_scaling[s]. However, these terms would
  // cancel. Thus, we do not multiply them in here.
  for (size_t t = 0; t < T; ++t)
    z += rel_rate[t] = phi(g, t) * theta(s, t);
  for (size_t t = 0; t < T; ++t)
    rel_rate[t] /= z;
  auto v = sample_multinomial<Int>(counts(g, s), rel_rate, rng);
  for (size_t t = 0; t < T; ++t) {
    contrib_gene_type(g, t) += v[t];
    contrib_spot_type(s, t) += v[t];
  }
  lambda_gene_spot(g, s) = z;
}

template <Kind kind>
/** sample count decomposition */
void VariantModel<kind>::sample_contributions(const IMatrix &counts) {
  if (verbosity >= Verbosity::Verbose)
    std::cout << "Sampling contributions" << std::endl;
  contributions_gene_type = IMatrix(G, T, arma::fill::zeros);
  contributions_spot_type = IMatrix(S, T, arma::fill::zeros);
#pragma omp parallel if (DO_PARALLEL)
  {
    IMatrix contrib_gene_type(G, T, arma::fill::zeros);
    IMatrix contrib_spot_type(S, T, arma::fill::zeros);
    const size_t thread_num = omp_get_thread_num();
#pragma omp for
    for (size_t g = 0; g < G; ++g)
      for (size_t s = 0; s < S; ++s)
        sample_contributions_sub(counts, g, s, EntropySource::rngs[thread_num],
                                 contrib_gene_type, contrib_spot_type);
#pragma omp critical
    {
      contributions_gene_type += contrib_gene_type;
      contributions_spot_type += contrib_spot_type;
    }
  }
}

template <Kind kind>
std::vector<Float> VariantModel<kind>::compute_intensities_gene_type() const {
  std::vector<Float> intensities(T, 0);
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t t = 0; t < T; ++t)
    for (size_t g = 0; g < G; ++g)
      intensities[t] += phi(g, t);
  return intensities;
}

template <Kind kind>
/** sample theta */
void VariantModel<kind>::sample_theta() {
  if (verbosity >= Verbosity::Verbose)
    std::cout << "Sampling Θ" << std::endl;
  const std::vector<Float> intensities = compute_intensities_gene_type();

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
              r_theta[t] + contributions_spot_type(s, t),
              1.0 / (p_theta[t] + intensities[t] * scale))(EntropySource::rng));
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

template <Kind kind>
/** sample p_theta and r_theta */
/* This is a simple Metropolis-Hastings sampling scheme */
void VariantModel<kind>::sample_p_and_r_theta() {
  if (verbosity >= Verbosity::Verbose)
    std::cout << "Sampling P_theta and R_theta" << std::endl;

  auto gen = [&](const std::pair<Float, Float> &x, std::mt19937 &rng) {
    std::normal_distribution<double> rnorm;
    const double f1 = exp(rnorm(rng));
    const double f2 = exp(rnorm(rng));
    return std::pair<Float, Float>(f1 * x.first, f2 * x.second);
  };

  for (size_t t = 0; t < T; ++t) {
    Float weight_sum = 0;
#pragma omp parallel for reduction(+ : weight_sum) if (DO_PARALLEL)
    for (size_t g = 0; g < G; ++g)
      weight_sum += phi(g, t);
    MetropolisHastings mh(parameters.temperature, parameters.prop_sd,
                          verbosity);

    std::vector<Int> count_sums(S, 0);
    std::vector<Float> weight_sums(S, 0);
#pragma omp parallel for if (DO_PARALLEL)
    for (size_t s = 0; s < S; ++s) {
      count_sums[s] = contributions_spot_type(s, t);
      weight_sums[s] = weight_sum * spot_scaling[s];
      if (parameters.activate_experiment_scaling)
        weight_sums[s] *= experiment_scaling_long[s];
    }
    auto res = mh.sample(std::pair<Float, Float>(r_theta[t], p_theta[t]),
                         parameters.n_iter, EntropySource::rng, gen,
                         compute_conditional_theta, count_sums, weight_sums,
                         hyperparameters);
    r_theta[t] = res.first;
    p_theta[t] = res.second;
  }
}

// herehere

/** sample p_phi and r_phi */
/* This is a simple Metropolis-Hastings sampling scheme */
template <Kind kind>
void VariantModel<kind>::sample_p_and_r() {
  if (verbosity >= Verbosity::Verbose)
    std::cout << "Sampling P and R" << std::endl;

  auto gen = [&](const std::pair<Float, Float> &x, std::mt19937 &rng) {
    std::normal_distribution<double> rnorm;
    const double f1 = exp(rnorm(rng));
    const double f2 = exp(rnorm(rng));
    return std::pair<Float, Float>(f1 * x.first, f2 * x.second);
  };

  for (size_t t = 0; t < T; ++t) {
    Float weight_sum = 0;
    for (size_t s = 0; s < S; ++s) {
      Float x = theta(s, t) * spot_scaling[s];
      if (parameters.activate_experiment_scaling)
        x *= experiment_scaling_long[s];
      weight_sum += x;
    }
    MetropolisHastings mh(parameters.temperature, parameters.prop_sd,
                          verbosity);

#pragma omp parallel for if (DO_PARALLEL)
    for (size_t g = 0; g < G; ++g) {
      const Int count_sum = contributions_gene_type(g, t);
      const size_t thread_num = omp_get_thread_num();
      auto res = mh.sample(std::pair<Float, Float>(r_phi(g, t), p_phi(g, t)),
                           parameters.n_iter, EntropySource::rngs[thread_num],
                           gen, compute_conditional, count_sum, weight_sum,
                           hyperparameters);
      r_phi(g, t) = res.first;
      p_phi(g, t) = res.second;
    }
  }
}

template <Kind kind>
Float VariantModel<kind>::sample_phi_sub(size_t g, size_t t, Float theta_t,
                                         RNG &rng) const {
  // NOTE: gamma_distribution takes a shape and scale parameter
  return std::gamma_distribution<Float>(
      r_phi(g, t) + contributions_gene_type(g, t),
      1.0 / (p_phi(g, t) + theta_t))(rng);
}

/** sample phi */
template <Kind kind>
void VariantModel<kind>::sample_phi() {
  if (verbosity >= Verbosity::Verbose)
    std::cout << "Sampling Φ" << std::endl;
  Vector theta_t(T, arma::fill::zeros);
  for (size_t s = 0; s < S; ++s) {
    Float prod = spot_scaling[s];
    if (parameters.activate_experiment_scaling)
      prod *= experiment_scaling_long[s];
    for (size_t t = 0; t < T; ++t)
      theta_t[t] += theta(s, t) * prod;
  }

#pragma omp parallel for if (DO_PARALLEL)
  for (size_t g = 0; g < G; ++g) {
    const size_t thread_num = omp_get_thread_num();
    for (size_t t = 0; t < T; ++t)
      phi(g, t)
          = sample_phi_sub(g, t, theta_t[t], EntropySource::rngs[thread_num]);
  }
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

/** sample spot scaling factors */
template <Kind kind>
void VariantModel<kind>::sample_spot_scaling() {
  if (verbosity >= Verbosity::Verbose)
    std::cout << "Sampling spot scaling factors" << std::endl;
  Vector phi_marginal(T, arma::fill::zeros);
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t t = 0; t < T; ++t)
    for (size_t g = 0; g < G; ++g)
      phi_marginal(t) += phi(g, t);
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t s = 0; s < S; ++s) {
    const Int summed_contribution = contributions_spot(s);

    Float intensity_sum = 0;
    for (size_t t = 0; t < T; ++t)
      intensity_sum += phi_marginal(t) * theta(s, t);
    if (parameters.activate_experiment_scaling)
      intensity_sum *= experiment_scaling_long[s];

    /*
    if (verbosity >= Verbosity::Debug)
    std::cout << "summed_contribution=" << summed_contribution
           << " intensity_sum=" << intensity_sum
           << " prev spot_scaling[" << s << "]=" << spot_scaling[s];
    */

    // NOTE: std::gamma_distribution takes a shape and scale parameter
    spot_scaling[s] = std::gamma_distribution<Float>(
        hyperparameters.spot_a + summed_contribution,
        1.0 / (hyperparameters.spot_b + intensity_sum))(EntropySource::rng);
    /*
    if (verbosity >= Verbosity::Debug)
    std::cout << "new spot_scaling[" << s << "]=" << spot_scaling[s] <<
    std::endl;
    */
  }

  if ((parameters.enforce_mean & ForceMean::Spot) != ForceMean::None) {
    double z = 0;
#pragma omp parallel for reduction(+ : z) if (DO_PARALLEL)
    for (size_t s = 0; s < S; ++s)
      z += spot_scaling[s];
    z /= S;
#pragma omp parallel for if (DO_PARALLEL)
    for (size_t s = 0; s < S; ++s)
      spot_scaling[s] /= z;
  }
}

/** sample experiment scaling factors */
template <Kind kind>
void VariantModel<kind>::sample_experiment_scaling(const Counts &data) {
  if (verbosity >= Verbosity::Verbose)
    std::cout << "Sampling experiment scaling factors" << std::endl;

  Vector phi_marginal(T, arma::fill::zeros);
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t t = 0; t < T; ++t)
    for (size_t g = 0; g < G; ++g)
      phi_marginal(t) += phi(g, t);
  std::vector<Float> intensity_sums(E, 0);
  // TODO: improve parallelism
  for (size_t s = 0; s < S; ++s) {
    double x = 0;
#pragma omp parallel for reduction(+ : x) if (DO_PARALLEL)
    for (size_t t = 0; t < T; ++t)
      x += phi_marginal(t) * theta(s, t);
    x *= spot_scaling[s];
    intensity_sums[data.experiments[s]] += x;
  }

  if (verbosity >= Verbosity::Debug)
    for (size_t e = 0; e < E; ++e)
      std::cout << "contributions_experiment[" << e
                << "]=" << contributions_experiment[e] << std::endl
                << "intensity_sum=" << intensity_sums[e] << std::endl
                << "prev experiment_scaling[" << e
                << "]=" << experiment_scaling[e] << std::endl;

  for (size_t e = 0; e < E; ++e) {
    // NOTE: std::gamma_distribution takes a shape and scale parameter
    experiment_scaling[e] = std::gamma_distribution<Float>(
        hyperparameters.experiment_a + contributions_experiment(e),
        1.0 / (hyperparameters.experiment_b + intensity_sums[e]))(
        EntropySource::rng);
    if (verbosity >= Verbosity::Debug)
      std::cout << "new experiment_scaling[" << e
                << "]=" << experiment_scaling[e] << std::endl;
  }

  // copy the experiment scaling parameters into the spot-indexed vector
  update_experiment_scaling_long(data);

  if ((parameters.enforce_mean & ForceMean::Experiment) != ForceMean::None) {
    double z = 0;
#pragma omp parallel for reduction(+ : z) if (DO_PARALLEL)
    for (size_t s = 0; s < S; ++s)
      z += experiment_scaling_long[s];
    z /= S;
#pragma omp parallel for if (DO_PARALLEL)
    for (size_t s = 0; s < S; ++s)
      experiment_scaling_long[s] /= z;

    for (size_t e = 0; e < E; ++e)
      experiment_scaling[e] /= z;
  }
}

/** copy the experiment scaling parameters into the spot-indexed vector */
template <Kind kind>
void VariantModel<kind>::update_experiment_scaling_long(const Counts &data) {
  for (size_t s = 0; s < S; ++s)
    experiment_scaling_long[s] = experiment_scaling[data.experiments[s]];
}

template <Kind kind>
void VariantModel<kind>::gibbs_sample(const Counts &data, GibbsSample which,
                                      bool timing) {
  check_model(data.counts);

  Timer timer;
  if (flagged(which & GibbsSample::contributions)) {
    sample_contributions(data.counts);
    if (timing and verbosity >= Verbosity::Info)
      std::cout << "This took " << timer.tock() << "μs." << std::endl;
    if (verbosity >= Verbosity::Everything)
      std::cout << "Log-likelihood = " << log_likelihood(data.counts)
                << std::endl;
    check_model(data.counts);
  }

  if (flagged(which & GibbsSample::merge_split)) {
    // NOTE: this has to be done right after the Gibbs step for the
    // contributions
    // because otherwise the lambda_gene_spot variables are not correct
    timer.tick();
    sample_split_merge(data, which);
    if (timing and verbosity >= Verbosity::Info)
      std::cout << "This took " << timer.tock() << "μs." << std::endl;
    if (verbosity >= Verbosity::Everything)
      std::cout << "Log-likelihood = " << log_likelihood(data.counts)
                << std::endl;
    check_model(data.counts);
  }

  if (flagged(which & GibbsSample::spot_scaling)) {
    timer.tick();
    sample_spot_scaling();
    if (timing and verbosity >= Verbosity::Info)
      std::cout << "This took " << timer.tock() << "μs." << std::endl;
    if (verbosity >= Verbosity::Everything)
      std::cout << "Log-likelihood = " << log_likelihood(data.counts)
                << std::endl;
    check_model(data.counts);
  }

  if (flagged(which & GibbsSample::experiment_scaling)) {
    if (E > 1 and parameters.activate_experiment_scaling) {
      timer.tick();
      sample_experiment_scaling(data);
      if (timing and verbosity >= Verbosity::Info)
        std::cout << "This took " << timer.tock() << "μs." << std::endl;
      if (verbosity >= Verbosity::Everything)
        std::cout << "Log-likelihood = " << log_likelihood(data.counts)
                  << std::endl;
      check_model(data.counts);
    }
  }

  if (flagged(which & (GibbsSample::phi_r | GibbsSample::phi_p))) {
    timer.tick();
    sample_p_and_r();
    if (timing and verbosity >= Verbosity::Info)
      std::cout << "This took " << timer.tock() << "μs." << std::endl;
    if (verbosity >= Verbosity::Everything)
      std::cout << "Log-likelihood = " << log_likelihood(data.counts)
                << std::endl;
    check_model(data.counts);
  }

  if (flagged(which & (GibbsSample::theta_r | GibbsSample::theta_p))) {
    timer.tick();
    sample_p_and_r_theta();
    if (timing and verbosity >= Verbosity::Info)
      std::cout << "This took " << timer.tock() << "μs." << std::endl;
    if (verbosity >= Verbosity::Everything)
      std::cout << "Log-likelihood = " << log_likelihood(data.counts)
                << std::endl;
    check_model(data.counts);
  }

  if (flagged(which & GibbsSample::phi)) {
    timer.tick();
    sample_phi();
    if (timing and verbosity >= Verbosity::Info)
      std::cout << "This took " << timer.tock() << "μs." << std::endl;
    if (verbosity >= Verbosity::Everything)
      std::cout << "Log-likelihood = " << log_likelihood(data.counts)
                << std::endl;
    check_model(data.counts);
  }

  if (flagged(which & GibbsSample::theta)) {
    timer.tick();
    sample_theta();
    if (timing and verbosity >= Verbosity::Info)
      std::cout << "This took " << timer.tock() << "μs." << std::endl;
    if (verbosity >= Verbosity::Everything)
      std::cout << "Log-likelihood = " << log_likelihood(data.counts)
                << std::endl;
    check_model(data.counts);
  }
}

template <Kind kind>
void VariantModel<kind>::sample_split_merge(const Counts &data,
                                            GibbsSample which) {
  if (T < 2)
    return;

  size_t s1 = std::uniform_int_distribution<Int>(0, S - 1)(EntropySource::rng);
  size_t s2 = std::uniform_int_distribution<Int>(0, S - 1)(EntropySource::rng);

  std::vector<Float> p1(T), p2(T);
  for (size_t t = 0; t < T; ++t) {
    p1[t] = theta(s1, t);
    p2[t] = theta(s2, t);
  }

  size_t t1
      = std::discrete_distribution<Int>(begin(p1), end(p1))(EntropySource::rng);
  size_t t2
      = std::discrete_distribution<Int>(begin(p2), end(p2))(EntropySource::rng);

  if (t1 != t2)
    sample_merge(data, t1, t2, which);
  else
    sample_split(data, t1, which);
}

template <Kind kind>
size_t VariantModel<kind>::find_weakest_factor() const {
  std::vector<Float> x(T, 0);
  std::cout << "Factor strengths: ";
  for (size_t t = 0; t < T; ++t) {
    Float y = 0;
    for (size_t g = 0; g < G; ++g)
      y += phi(g, t);
    for (size_t s = 0; s < S; ++s) {
      Float z = y * theta(s, t) * spot_scaling[s];
      if (parameters.activate_experiment_scaling)
        z *= experiment_scaling_long[s];
      x[t] += z;
    }
    std::cout << " " << x[t];
  }
  std::cout << std::endl;
  return std::distance(begin(x), min_element(begin(x), end(x)));
}

template <Kind kind>
VariantModel<kind> VariantModel<kind>::run_submodel(
    size_t t, size_t n, const Counts &counts, GibbsSample which,
    const std::string &prefix, const std::vector<size_t> &init_factors) {
  const bool show_timing = false;
  // TODO: use init_factors
  VariantModel<kind> sub_model(counts, t, hyperparameters, parameters,
                               Verbosity::Info);
  for (size_t s = 0; s < S; ++s) {
    sub_model.spot_scaling[s] = spot_scaling[s];
    sub_model.experiment_scaling_long[s] = experiment_scaling_long[s];
  }
  for (size_t e = 0; e < E; ++e)
    sub_model.experiment_scaling[e] = experiment_scaling[e];

  if (print_sub_model_cnt)
    sub_model.store(counts,
                    prefix + "submodel_init_" + std::to_string(sub_model_cnt));

  // keep spot and experiment scaling fixed
  // don't recurse into either merge or sample steps
  which = which
          & ~(GibbsSample::spot_scaling | GibbsSample::experiment_scaling
              | GibbsSample::merge_split);
  for (size_t i = 0; i < n; ++i)
    sub_model.gibbs_sample(counts, which, show_timing);

  if (print_sub_model_cnt)
    sub_model.store(counts,
                    prefix + "submodel_opti_" + std::to_string(sub_model_cnt));
  // sub_model_cnt++; // TODO

  return sub_model;
}

template <Kind kind>
void VariantModel<kind>::lift_sub_model(const VariantModel &sub_model,
                                        size_t t1, size_t t2) {
  for (size_t g = 0; g < G; ++g) {
    phi(g, t1) = sub_model.phi(g, t2);
    r_phi(g, t1) = sub_model.r_phi(g, t2);
    p_phi(g, t1) = sub_model.p_phi(g, t2);
    contributions_gene_type(g, t1) = sub_model.contributions_gene_type(g, t2);
  }

  for (size_t s = 0; s < S; ++s) {
    theta(s, t1) = sub_model.theta(s, t2);
    contributions_spot_type(s, t1) = sub_model.contributions_spot_type(s, t2);
  }
  r_theta(t1) = sub_model.r_theta(t2);
  p_theta(t1) = sub_model.p_theta(t2);
}

template <Kind kind>
void VariantModel<kind>::sample_split(const Counts &data, size_t t1,
                                      GibbsSample which) {
  size_t t2 = find_weakest_factor();
  if (verbosity >= Verbosity::Info)
    std::cout << "Performing a split step. Splitting " << t1 << " and " << t2
              << "." << std::endl;
  VariantModel previous(*this);

  double ll_previous = (consider_factor_likel
                            ? log_likelihood_factor(data.counts, t1)
                                  + log_likelihood_factor(data.counts, t2)
                            : 0)
                       + log_likelihood_poisson_counts(data.counts);

  Counts sub_counts = data;
  for (size_t g = 0; g < G; ++g)
    for (size_t s = 0; s < S; ++s) {
      Float lambda = phi(g, t1) * theta(s, t1) + phi(g, t2) * theta(s, t2);
      sub_counts.counts(g, s) = std::binomial_distribution<Int>(
          data.counts(g, s),
          lambda / lambda_gene_spot(g, s))(EntropySource::rng);
      // remove effect of current parameters
      lambda_gene_spot(g, s) -= lambda;
    }

  VariantModel sub_model
      = run_submodel(2, num_sub_gibbs, sub_counts, which, "splitmerge_split_");

  lift_sub_model(sub_model, t1, 0);
  lift_sub_model(sub_model, t2, 1);

  // add effect of updated parameters
  for (size_t g = 0; g < G; ++g)
    for (size_t s = 0; s < S; ++s)
      lambda_gene_spot(g, s)
          += phi(g, t1) * theta(s, t1) + phi(g, t2) * theta(s, t2);

  double ll_updated = (consider_factor_likel
                           ? log_likelihood_factor(data.counts, t1)
                                 + log_likelihood_factor(data.counts, t2)
                           : 0)
                      + log_likelihood_poisson_counts(data.counts);

  std::cout << "ll_split_previous = " << ll_previous << std::endl
            << "ll_split_updated = " << ll_updated << std::endl;
  if (gibbs_test(ll_updated, ll_previous, verbosity)) {
    std::cout << "ll_split_ACCEPT" << std::endl;
  } else {
    *this = previous;
    std::cout << "ll_split_REJECT" << std::endl;
  }
}

template <Kind kind>
void VariantModel<kind>::sample_merge(const Counts &data, size_t t1, size_t t2,
                                      GibbsSample which) {
  if (verbosity >= Verbosity::Info)
    std::cout << "Performing a merge step. Merging types " << t1 << " and "
              << t2 << "." << std::endl;
  VariantModel previous(*this);

  double ll_previous = (consider_factor_likel
                            ? log_likelihood_factor(data.counts, t1)
                                  + log_likelihood_factor(data.counts, t2)
                            : 0)
                       + log_likelihood_poisson_counts(data.counts);

  Counts sub_counts = data;
  for (size_t g = 0; g < G; ++g)
    for (size_t s = 0; s < S; ++s) {
      Float lambda = phi(g, t1) * theta(s, t1) + phi(g, t2) * theta(s, t2);
      sub_counts.counts(g, s) = std::binomial_distribution<Int>(
          data.counts(g, s),
          lambda / lambda_gene_spot(g, s))(EntropySource::rng);
      // remove effect of current parameters
      lambda_gene_spot(g, s) -= lambda;
    }

  VariantModel sub_model
      = run_submodel(1, num_sub_gibbs, sub_counts, which, "splitmerge_merge_");

  lift_sub_model(sub_model, t1, 0);

  // add effect of updated parameters
  for (size_t g = 0; g < G; ++g)
    for (size_t s = 0; s < S; ++s)
      lambda_gene_spot(g, s) += phi(g, t1) * theta(s, t1);

  // initialize p_phi
  for (size_t g = 0; g < G; ++g)
    p_phi(g, t2) = prob_to_neg_odds(
        sample_beta<Float>(hyperparameters.phi_p_1, hyperparameters.phi_p_2,
                           EntropySource::rngs[omp_get_thread_num()]));

  // initialize r_phi
  if (verbosity >= Verbosity::Debug)
    std::cout << "initializing r_phi." << std::endl;
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t g = 0; g < G; ++g)
    // NOTE: std::gamma_distribution takes a shape and scale parameter
    r_phi(g, t2) = std::gamma_distribution<Float>(
        hyperparameters.phi_r_1,
        1 / hyperparameters.phi_r_2)(EntropySource::rngs[omp_get_thread_num()]);

  // initialize phi
  if (verbosity >= Verbosity::Debug)
    std::cout << "initializing phi." << std::endl;
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t g = 0; g < G; ++g)
    // NOTE: std::gamma_distribution takes a shape and scale parameter
    phi(g, t2) = std::gamma_distribution<Float>(r_phi(g, t2), 1 / p_phi(g, t2))(
        EntropySource::rngs[omp_get_thread_num()]);

  // randomly initialize p_theta
  if (verbosity >= Verbosity::Debug)
    std::cout << "initializing p of theta." << std::endl;
  if (true)  // TODO make this CLI-switchable
    p_theta[t2] = prob_to_neg_odds(sample_beta<Float>(
        hyperparameters.theta_p_1, hyperparameters.theta_p_2));
  else
    p_theta[t2] = 1;

  // initialize r_theta
  if (verbosity >= Verbosity::Debug)
    std::cout << "initializing r of theta." << std::endl;
  // NOTE: std::gamma_distribution takes a shape and scale parameter
  r_theta[t2] = std::gamma_distribution<Float>(
      hyperparameters.theta_r_1,
      1 / hyperparameters.theta_r_2)(EntropySource::rng);

  // initialize theta
  if (verbosity >= Verbosity::Debug)
    std::cout << "initializing theta." << std::endl;
  for (size_t s = 0; s < S; ++s)
    // NOTE: std::gamma_distribution takes a shape and scale parameter
    theta(s, t2) = std::gamma_distribution<Float>(
        r_theta(t2), 1 / p_theta(t2))(EntropySource::rng);

  // add effect of updated parameters
  for (size_t g = 0; g < G; ++g)
    for (size_t s = 0; s < S; ++s)
      lambda_gene_spot(g, s) += phi(g, t2) * theta(s, t2);

  for (size_t g = 0; g < G; ++g)
    contributions_gene_type(g, t2) = 0;
  for (size_t s = 0; s < S; ++s)
    contributions_spot_type(s, t2) = 0;
  for (size_t g = 0; g < G; ++g)
    for (size_t s = 0; s < S; ++s) {
      Float lambda = phi(g, t2) * theta(s, t2);
      Int count = std::binomial_distribution<Int>(
          data.counts(g, s),
          lambda / lambda_gene_spot(g, s))(EntropySource::rng);
      contributions_gene_type(g, t2) += count;
      contributions_spot_type(s, t2) += count;
    }

  double ll_updated = (consider_factor_likel
                           ? log_likelihood_factor(data.counts, t1)
                                 + log_likelihood_factor(data.counts, t2)
                           : 0)
                      + log_likelihood_poisson_counts(data.counts);

  std::cout << "ll_merge_previous = " << ll_previous << std::endl
            << "ll_merge_updated = " << ll_updated << std::endl;
  if (gibbs_test(ll_updated, ll_previous, verbosity)) {
    std::cout << "ll_merge_ACCEPT" << std::endl;
  } else {
    *this = previous;
    std::cout << "ll_merge_REJECT" << std::endl;
  }
}

template <Kind kind>
std::vector<Int> VariantModel<kind>::sample_reads(size_t g, size_t s,
                                                  size_t n) const {
  std::vector<Float> prods(T);
  for (size_t t = 0; t < T; ++t) {
    prods[t] = theta(s, t) * spot_scaling[s];
    if (parameters.activate_experiment_scaling)
      prods[t] *= experiment_scaling_long[s];
  }

  std::vector<Int> v(n, 0);
  // TODO parallelize
  // #pragma omp parallel for if (DO_PARALLEL)
  for (size_t i = 0; i < n; ++i)
    for (size_t t = 0; t < T; ++t)
      v[i] += sample_negative_binomial(
          r_phi(g, t), prods[t] / (prods[t] + p_phi(g, t)), EntropySource::rng);
  return v;
}

template <Kind kind>
double VariantModel<kind>::posterior_expectation(size_t g, size_t s) const {
  double x = 0;
  for (size_t t = 0; t < T; ++t)
    x += r_phi(g, t) / p_phi(g, t) * theta(s, t);
  x *= spot_scaling[s];
  if (parameters.activate_experiment_scaling)
    x *= experiment_scaling_long[s];
  return x;
}

template <Kind kind>
double VariantModel<kind>::posterior_expectation_poisson(size_t g,
                                                         size_t s) const {
  double x = 0;
  for (size_t t = 0; t < T; ++t)
    x += phi(g, t) * theta(s, t);
  x *= spot_scaling[s];
  if (parameters.activate_experiment_scaling)
    x *= experiment_scaling_long[s];
  return x;
}

template <Kind kind>
double VariantModel<kind>::posterior_variance(size_t g, size_t s) const {
  double x = 0;
  double prod_ = spot_scaling[s];
  if (parameters.activate_experiment_scaling)
    prod_ *= experiment_scaling_long[s];
  for (size_t t = 0; t < T; ++t) {
    double prod = theta(s, t) * prod_;
    x += r_phi(g, t) * prod / (prod + p_phi(g, t)) / p_phi(g, t) / p_phi(g, t);
  }
  return x;
}

template <Kind kind>
Matrix VariantModel<kind>::posterior_expectations() const {
  Matrix m(G, S);
  for (size_t g = 0; g < G; ++g)
    for (size_t s = 0; s < S; ++s)
      m(g, s) = posterior_expectation(g, s);
  return m;
}

template <Kind kind>
Matrix VariantModel<kind>::posterior_expectations_poisson() const {
  Matrix m(G, S);
  for (size_t g = 0; g < G; ++g)
    for (size_t s = 0; s < S; ++s)
      m(g, s) = posterior_expectation_poisson(g, s);
  return m;
}

template <Kind kind>
Matrix VariantModel<kind>::posterior_variances() const {
  Matrix m(G, S);
  for (size_t g = 0; g < G; ++g)
    for (size_t s = 0; s < S; ++s)
      m(g, s) = posterior_variance(g, s);
  return m;
}

template <Kind kind>
void VariantModel<kind>::check_model(const IMatrix &counts) const {
  return;
  // check that the contributions add up to the observations
  /*
  for (size_t g = 0; g < G; ++g)
    for (size_t s = 0; s < S; ++s) {
      Int z = 0;
      for (size_t t = 0; t < T; ++t) z += contributions(g, s, t);
      if (z != counts(g, s))
        throw(std::runtime_error(
            "Contributions do not add up to observations for gene " +
            std::to_string(g) + " in spot " + std::to_string(s) + "."));
    }
  */

  // check that phi is positive
  for (size_t g = 0; g < G; ++g)
    for (size_t t = 0; t < T; ++t) {
      /*
      if (phi[g][t] == 0)
        throw(std::runtime_error("Phi is zero for gene " + std::to_string(g) +
                            " in factor " + std::to_string(t) + "."));
                            */
      if (phi(g, t) < 0)
        throw(std::runtime_error("Phi is negative for gene " + std::to_string(g)
                                 + " in factor " + std::to_string(t) + "."));
    }

  // check that theta is positive
  for (size_t s = 0; s < S; ++s)
    for (size_t t = 0; t < T; ++t) {
      if (theta(s, t) == 0)
        throw(std::runtime_error("Theta is zero for spot " + std::to_string(s)
                                 + " in factor " + std::to_string(t) + "."));
      if (theta(s, t) < 0)
        throw(std::runtime_error("Theta is negative for spot "
                                 + std::to_string(s) + " in factor "
                                 + std::to_string(t) + "."));
    }

  // check that r_phi and p_phi are positive, and that p is < 1
  for (size_t g = 0; g < G; ++g)
    for (size_t t = 0; t < T; ++t) {
      if (p_phi(g, t) < 0)
        throw(std::runtime_error("P[" + std::to_string(g) + "]["
                                 + std::to_string(t) + "] is smaller zero: p="
                                 + std::to_string(p_phi(g, t)) + "."));
      if (p_phi(g, t) == 0)
        throw(std::runtime_error("P is zero for gene " + std::to_string(g)
                                 + " in factor " + std::to_string(t) + "."));

      if (r_phi(g, t) < 0)
        throw(std::runtime_error("R[" + std::to_string(g) + "]["
                                 + std::to_string(t) + "] is smaller zero: r="
                                 + std::to_string(r_phi(g, t)) + "."));
      if (r_phi(g, t) == 0)
        throw(std::runtime_error("R is zero for gene " + std::to_string(g)
                                 + " in factor " + std::to_string(t) + "."));
    }

  // check hyperparameters
  if (hyperparameters.phi_r_1 == 0)
    throw(std::runtime_error("The prior phi_r_1 is zero."));
  if (hyperparameters.phi_r_2 == 0)
    throw(std::runtime_error("The prior phi_r_2 is zero."));
  if (hyperparameters.phi_p_1 == 0)
    throw(std::runtime_error("The prior phi_p_1 is zero."));
  if (hyperparameters.phi_p_2 == 0)
    throw(std::runtime_error("The prior phi_p_2 is zero."));
  if (hyperparameters.alpha == 0)
    throw(std::runtime_error("The prior alpha is zero."));
}

template <Kind kind>
std::ostream &operator<<(std::ostream &os,
                         const FactorAnalysis::VariantModel<kind> &pfa) {
  os << "Variant Poisson Factor Analysis "
     << "S = " << pfa.S << " "
     << "G = " << pfa.G << " "
     << "T = " << pfa.T << std::endl;

  if (pfa.verbosity >= Verbosity::Verbose) {
    os << "Φ" << std::endl;
    for (size_t g = 0; g < std::min<size_t>(pfa.G, 10); ++g) {
      for (size_t t = 0; t < pfa.T; ++t)
        os << (t > 0 ? "\t" : "") << pfa.phi(g, t);
      os << std::endl;
    }

    size_t phi_zeros = 0;
    os << "Φ factor sums" << std::endl;
    for (size_t t = 0; t < pfa.T; ++t) {
      double sum = 0;
      for (size_t g = 0; g < pfa.G; ++g) {
        if (pfa.phi(g, t) == 0)
          phi_zeros++;
        sum += pfa.phi(g, t);
      }
      os << (t > 0 ? "\t" : "") << sum;
    }
    os << std::endl;
    os << "There are " << phi_zeros << " zeros in Φ. This corresponds to "
       << (100.0 * phi_zeros / pfa.T / pfa.G) << "%." << std::endl;

    os << "Θ" << std::endl;
    for (size_t s = 0; s < std::min<size_t>(pfa.S, 10); ++s) {
      for (size_t t = 0; t < pfa.T; ++t)
        os << (t > 0 ? "\t" : "") << pfa.theta(s, t);
      os << std::endl;
    }

    size_t theta_zeros = 0;
    os << "Θ factor sums" << std::endl;
    for (size_t t = 0; t < pfa.T; ++t) {
      double sum = 0;
      for (size_t s = 0; s < pfa.S; ++s) {
        if (pfa.theta(s, t) == 0)
          theta_zeros++;
        sum += pfa.theta(s, t);
      }
      os << (t > 0 ? "\t" : "") << sum;
    }
    os << std::endl;
    os << "There are " << theta_zeros << " zeros in Θ." << std::endl;

    os << "R" << std::endl;
    for (size_t g = 0; g < std::min<size_t>(pfa.G, 10); ++g) {
      for (size_t t = 0; t < pfa.T; ++t)
        os << (t > 0 ? "\t" : "") << pfa.r_phi(g, t);
      os << std::endl;
    }

    size_t r_zeros = 0;
    for (size_t g = 0; g < pfa.G; ++g)
      for (size_t t = 0; t < pfa.T; ++t)
        if (pfa.r_phi(g, t) == 0)
          r_zeros++;
    os << "There are " << r_zeros << " zeros in r_phi. This corresponds to "
       << (100.0 * r_zeros / pfa.G / pfa.T) << "%." << std::endl;

    os << "P" << std::endl;
    for (size_t g = 0; g < std::min<size_t>(pfa.G, 10); ++g) {
      for (size_t t = 0; t < pfa.T; ++t)
        os << (t > 0 ? "\t" : "") << pfa.p_phi(g, t);
      os << std::endl;
    }

    size_t p_zeros = 0;
    for (size_t g = 0; g < pfa.G; ++g)
      for (size_t t = 0; t < pfa.T; ++t)
        if (pfa.p_phi(g, t) == 0)
          p_zeros++;
    os << "There are " << p_zeros << " zeros in p. This corresponds to "
       << (100.0 * p_zeros / pfa.G / pfa.T) << "%." << std::endl;

    os << "Spot scaling factors" << std::endl;
    for (size_t s = 0; s < pfa.S; ++s)
      os << (s > 0 ? "\t" : "") << pfa.spot_scaling[s];
    os << std::endl;
    size_t spot_scaling_zeros = 0;
    for (size_t s = 0; s < pfa.S; ++s)
      if (pfa.spot_scaling[s] == 0)
        spot_scaling_zeros++;
    os << "There are " << spot_scaling_zeros << " zeros in spot_scaling."
       << std::endl;
    os << Stats::summary(pfa.spot_scaling) << std::endl;

    if (pfa.parameters.activate_experiment_scaling) {
      os << "Experiment scaling factors" << std::endl;
      for (size_t e = 0; e < pfa.E; ++e)
        os << (e > 0 ? "\t" : "") << pfa.experiment_scaling[e];
      os << std::endl;
      size_t experiment_scaling_zeros = 0;
      for (size_t e = 0; e < pfa.E; ++e)
        if (pfa.experiment_scaling[e] == 0)
          spot_scaling_zeros++;
      os << "There are " << experiment_scaling_zeros
         << " zeros in experiment_scaling." << std::endl;
      os << Stats::summary(pfa.experiment_scaling) << std::endl;
    }
  }

  os << "R_theta factors" << std::endl;
  for (size_t t = 0; t < pfa.T; ++t)
    os << (t > 0 ? "\t" : "") << pfa.r_theta[t];
  os << std::endl;
  size_t r_theta_zeros = 0;
  for (size_t t = 0; t < pfa.T; ++t)
    if (pfa.r_theta[t] == 0)
      r_theta_zeros++;
  os << "There are " << r_theta_zeros << " zeros in R_theta." << std::endl;
  os << Stats::summary(pfa.r_theta) << std::endl;

  os << "P_theta factors" << std::endl;
  for (size_t t = 0; t < pfa.T; ++t)
    os << (t > 0 ? "\t" : "") << pfa.p_theta[t];
  os << std::endl;
  size_t p_theta_zeros = 0;
  for (size_t t = 0; t < pfa.T; ++t)
    if (pfa.p_theta[t] == 0)
      p_theta_zeros++;
  os << "There are " << p_theta_zeros << " zeros in P_theta." << std::endl;
  os << Stats::summary(pfa.p_theta) << std::endl;

  return os;
}
}

#endif
