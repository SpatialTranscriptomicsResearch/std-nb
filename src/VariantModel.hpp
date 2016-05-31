#ifndef VARIANTMODEL_HPP
#define VARIANTMODEL_HPP

#include <random>
#include <omp.h>
#include "counts.hpp"
#include "entropy.hpp"
#include "parallel.hpp"
#include "parameters.hpp"
#include "compression.hpp"
#include "features.hpp"
#include "io.hpp"
#include "metropolis_hastings.hpp"
#include "pdist.hpp"
#include "odds.hpp"
#include "priors.hpp"
#include "sampling.hpp"
#include "stats.hpp"
#include "target.hpp"
#include "timer.hpp"
#include "verbosity.hpp"

namespace PoissonFactorization {

#define DEFAULT_SEPARATOR "\t"
#define DEFAULT_LABEL ""
#define print_sub_model_cnt true

const size_t num_sub_gibbs_split = 50;
const size_t num_sub_gibbs_merge = 10;
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

struct Paths {
  Paths(const std::string &prefix, const std::string &suffix = "");
  std::string phi, theta, spot, experiment, r_phi, p_phi, r_theta, p_theta;
  std::string contributions_gene_type, contributions_spot_type,
      contributions_spot, contributions_experiment;
};

template <Kind kind = Kind::Gamma>
struct Model {
  typedef Features<kind> features_t;
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

  Parameters parameters;

  /** hidden contributions to the count data due to the different factors */
  IMatrix contributions_gene_type, contributions_spot_type;
  IVector contributions_spot, contributions_experiment;

  /** Normalizing factor to translate Poisson rates \lambda_{xgst} to relative
   * frequencies \lambda_{gst} / z_{gs} for the multionomial distribution */
  Matrix lambda_gene_spot;

  /** factor loading matrix */
  features_t features;
  inline Float &phi(size_t x, size_t y) { return features.phi(x, y); };
  inline Float phi(size_t x, size_t y) const { return features.phi(x, y); };

  /** factor score matrix */
  Matrix theta;

  /** spot scaling vector */
  Vector spot_scaling;

  /** experiment scaling vector */
  Vector experiment_scaling;
  Vector experiment_scaling_long;

  /** shape parameter for the prior of the mixing matrix */
  Vector r_theta;
  /** scale parameter for the prior of the mixing matrix */
  /* Stored as negative-odds */
  Vector p_theta;

  Verbosity verbosity;

  Model(const Counts &counts, const size_t T, const Parameters &parameters,
        Verbosity verbosity);

  Model(const Counts &counts, const Paths &paths, const Parameters &parameters,
        Verbosity verbosity);

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

  /** sample theta */
  void sample_theta();

  /** sample p_theta and r_theta */
  void sample_p_and_r_theta();

  /** sample spot scaling factors */
  void sample_spot_scaling();

  /** sample experiment scaling factors */
  void sample_experiment_scaling(const Counts &data);

  /** sample each of the variables from their conditional posterior */
  void gibbs_sample(const Counts &data, Target which, bool timing);

  void sample_split_merge(const Counts &data, Target which);
  void sample_merge(const Counts &data, size_t t1, size_t t2, Target which);
  void sample_split(const Counts &data, size_t t, Target which);
  void lift_sub_model(const Model &sub_model, size_t t1, size_t t2);

  Model run_submodel(size_t t, size_t n, const Counts &counts, Target which,
                     const std::string &prefix,
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
                         const PoissonFactorization::Model<kind> &pfa);

template <Kind kind>
Model<kind>::Model(const Counts &c, const size_t T_,
                   const Parameters &parameters_, Verbosity verbosity_)
    : G(c.counts.n_rows),
      S(c.counts.n_cols),
      T(T_),
      E(c.experiment_names.size()),
      parameters(parameters_),
      contributions_gene_type(G, T, arma::fill::zeros),
      contributions_spot_type(S, T, arma::fill::zeros),
      contributions_spot(S, arma::fill::zeros),
      contributions_experiment(E, arma::fill::zeros),
      lambda_gene_spot(G, S, arma::fill::zeros),
      features(G, S, T, parameters),
      theta(S, T),
      spot_scaling(S, arma::fill::ones),
      experiment_scaling(E, arma::fill::ones),
      experiment_scaling_long(S, arma::fill::ones),
      r_theta(T),
      p_theta(T),
      verbosity(verbosity_) {
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
void Model<kind>::initialize_p_theta() {
  // initialize p_theta
  if (verbosity >= Verbosity::Debug)
    std::cout << "initializing p_theta." << std::endl;
  for (size_t t = 0; t < T; ++t)
    if (false)  // TODO make this CLI-switchable
      p_theta[t] = prob_to_neg_odds(sample_beta<Float>(
          parameters.hyperparameters.theta_p_1, parameters.hyperparameters.theta_p_2));
    else
      p_theta[t] = 1;
}

template <Kind kind>
void Model<kind>::initialize_r_theta() {
  // initialize r_theta
  if (verbosity >= Verbosity::Debug)
    std::cout << "initializing r_theta." << std::endl;
  for (size_t t = 0; t < T; ++t)
    // NOTE: std::gamma_distribution takes a shape and scale parameter
    r_theta[t] = std::gamma_distribution<Float>(
        parameters.hyperparameters.theta_r_1,
        1 / parameters.hyperparameters.theta_r_2)(EntropySource::rng);
}

template <Kind kind>
void Model<kind>::initialize_theta() {
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
Model<kind>::Model(const Counts &c, const Paths &paths,
                   const Parameters &parameters_, Verbosity verbosity_)
    : G(c.counts.n_rows),
      S(c.counts.n_cols),
      T(num_lines(paths.r_theta)),
      E(c.experiment_names.size()),
      parameters(parameters_),
      contributions_gene_type(parse_file<IMatrix>(paths.contributions_gene_type, read_imatrix, DEFAULT_SEPARATOR, DEFAULT_LABEL)),
      contributions_spot_type(parse_file<IMatrix>(paths.contributions_spot_type, read_imatrix, DEFAULT_SEPARATOR, DEFAULT_LABEL)),
      contributions_spot(parse_file<IVector>(paths.contributions_spot, read_vector<IVector>, DEFAULT_SEPARATOR)),
      contributions_experiment(parse_file<IVector>(paths.contributions_experiment, read_vector<IVector>, DEFAULT_SEPARATOR)),
      // features(parse_file<Matrix>(paths.features, read_matrix, DEFAULT_SEPARATOR, DEFAULT_LABEL)), // TODO reactivate
      features(G, S, T, parameters), // TODO deactivate
      theta(parse_file<Matrix>(paths.theta, read_matrix, DEFAULT_SEPARATOR, DEFAULT_LABEL)),
      spot_scaling(parse_file<Vector>(paths.spot, read_vector<Vector>, DEFAULT_SEPARATOR)),
      experiment_scaling(parse_file<Vector>(paths.experiment, read_vector<Vector>, DEFAULT_SEPARATOR)),
      experiment_scaling_long(S),
      r_theta(parse_file<Vector>(paths.r_theta, read_vector<Vector>, DEFAULT_SEPARATOR)),
      p_theta(parse_file<Vector>(paths.p_theta, read_vector<Vector>, DEFAULT_SEPARATOR)),
      verbosity(verbosity_) {
  update_experiment_scaling_long(c);

  if (verbosity >= Verbosity::Debug)
    std::cout << *this << std::endl;
}

template <Kind kind>
// TODO ensure no NaNs or infinities are generated
double Model<kind>::log_likelihood_factor(const IMatrix &counts,
                                          size_t t) const {
  double l = features.log_likelihood_factor(counts, t);

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
  l += log_gamma(r_theta(t), parameters.hyperparameters.theta_r_1,
                 1.0 / parameters.hyperparameters.theta_r_2);

  if (verbosity >= Verbosity::Debug)
    std::cout << "ll_r_theta = " << l << std::endl;

  l += log_beta_neg_odds(p_theta(t), parameters.hyperparameters.theta_p_1,
                         parameters.hyperparameters.theta_p_2);

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
double Model<kind>::log_likelihood_poisson_counts(
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
double Model<kind>::log_likelihood(const IMatrix &counts) const {
  double l = 0;
  for (size_t t = 0; t < T; ++t)
    l += log_likelihood_factor(counts, t);

  for (size_t s = 0; s < S; ++s)
    l += log_gamma(spot_scaling(s), parameters.hyperparameters.spot_a,
                   1.0 / parameters.hyperparameters.spot_b);
  if (parameters.activate_experiment_scaling) {
    for (size_t e = 0; e < E; ++e)
      l += log_gamma(experiment_scaling(e), parameters.hyperparameters.experiment_a,
                     1.0 / parameters.hyperparameters.experiment_b);
  }

  l += log_likelihood_poisson_counts(counts);

  return l;
}

template <Kind kind>
Matrix Model<kind>::weighted_theta() const {
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
void Model<kind>::store(const Counts &counts, const std::string &prefix,
                               bool mean_and_variance) const {
  std::vector<std::string> factor_names;
  for (size_t t = 1; t <= T; ++t)
    factor_names.push_back("Factor " + std::to_string(t));
  auto &gene_names = counts.row_names;
  auto &spot_names = counts.col_names;
  features.store(prefix, gene_names, factor_names);
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
  if (false and mean_and_variance) { // TODO reactivate
    write_matrix(posterior_expectations(), prefix + "means.txt", gene_names, spot_names);
    write_matrix(posterior_expectations_poisson(), prefix + "means_poisson.txt", gene_names, spot_names);
    write_matrix(posterior_variances(), prefix + "variances.txt", gene_names, spot_names);
  }
}

template <Kind kind>
void Model<kind>::sample_contributions_sub(const IMatrix &counts,
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
void Model<kind>::sample_contributions(const IMatrix &counts) {
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
/** sample theta */
void Model<kind>::sample_theta() {
  if (verbosity >= Verbosity::Verbose)
    std::cout << "Sampling Θ" << std::endl;
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
void Model<kind>::sample_p_and_r_theta() {
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
                         parameters.hyperparameters);
    r_theta[t] = res.first;
    p_theta[t] = res.second;
  }
}

/** sample spot scaling factors */
template <Kind kind>
void Model<kind>::sample_spot_scaling() {
  if (verbosity >= Verbosity::Verbose)
    std::cout << "Sampling spot scaling factors" << std::endl;
  auto phi_marginal = features.marginalize_genes();
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t s = 0; s < S; ++s) {
    const Int summed_contribution = contributions_spot(s);

    Float intensity_sum = 0;
    for (size_t t = 0; t < T; ++t)
      intensity_sum += phi_marginal[t] * theta(s, t);
    if (parameters.activate_experiment_scaling)
      intensity_sum *= experiment_scaling_long[s];

    // NOTE: std::gamma_distribution takes a shape and scale parameter
    spot_scaling[s] = std::gamma_distribution<Float>(
        parameters.hyperparameters.spot_a + summed_contribution,
        1.0 / (parameters.hyperparameters.spot_b + intensity_sum))(EntropySource::rng);
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
void Model<kind>::sample_experiment_scaling(const Counts &data) {
  if (verbosity >= Verbosity::Verbose)
    std::cout << "Sampling experiment scaling factors" << std::endl;

  auto phi_marginal = features.marginalize_genes();
  std::vector<Float> intensity_sums(E, 0);
  // TODO: improve parallelism
  for (size_t s = 0; s < S; ++s) {
    double x = 0;
#pragma omp parallel for reduction(+ : x) if (DO_PARALLEL)
    for (size_t t = 0; t < T; ++t)
      x += phi_marginal[t] * theta(s, t);
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
        parameters.hyperparameters.experiment_a + contributions_experiment(e),
        1.0 / (parameters.hyperparameters.experiment_b + intensity_sums[e]))(
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
void Model<kind>::update_experiment_scaling_long(const Counts &data) {
  for (size_t s = 0; s < S; ++s)
    experiment_scaling_long[s] = experiment_scaling[data.experiments[s]];
}

template <Kind kind>
void Model<kind>::gibbs_sample(const Counts &data, Target which, bool timing) {
  check_model(data.counts);

  Timer timer;
  if (flagged(which & Target::contributions)) {
    sample_contributions(data.counts);
    if (timing and verbosity >= Verbosity::Info)
      std::cout << "This took " << timer.tock() << "μs." << std::endl;
    if (verbosity >= Verbosity::Everything)
      std::cout << "Log-likelihood = " << log_likelihood(data.counts)
                << std::endl;
    check_model(data.counts);
  }

  if (flagged(which & Target::merge_split)) {
    // NOTE: this has to be done right after the Gibbs step for the
    // contributions because otherwise lambda_gene_spot is inconsistent
    timer.tick();
    sample_split_merge(data, which);
    if (timing and verbosity >= Verbosity::Info)
      std::cout << "This took " << timer.tock() << "μs." << std::endl;
    if (verbosity >= Verbosity::Everything)
      std::cout << "Log-likelihood = " << log_likelihood(data.counts)
                << std::endl;
    check_model(data.counts);
  }

  if (flagged(which & Target::spot_scaling)) {
    timer.tick();
    sample_spot_scaling();
    if (timing and verbosity >= Verbosity::Info)
      std::cout << "This took " << timer.tock() << "μs." << std::endl;
    if (verbosity >= Verbosity::Everything)
      std::cout << "Log-likelihood = " << log_likelihood(data.counts)
                << std::endl;
    check_model(data.counts);
  }

  if (flagged(which & Target::experiment_scaling)) {
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

  if (flagged(which & (Target::phi_r | Target::phi_p))) {
    timer.tick();
    features.prior.sample(theta, contributions_gene_type, spot_scaling,
                          experiment_scaling_long);
    if (timing and verbosity >= Verbosity::Info)
      std::cout << "This took " << timer.tock() << "μs." << std::endl;
    if (verbosity >= Verbosity::Everything)
      std::cout << "Log-likelihood = " << log_likelihood(data.counts)
                << std::endl;
    check_model(data.counts);
  }

  if (flagged(which & (Target::theta_r | Target::theta_p))) {
    timer.tick();
    sample_p_and_r_theta();
    if (timing and verbosity >= Verbosity::Info)
      std::cout << "This took " << timer.tock() << "μs." << std::endl;
    if (verbosity >= Verbosity::Everything)
      std::cout << "Log-likelihood = " << log_likelihood(data.counts)
                << std::endl;
    check_model(data.counts);
  }

  if (flagged(which & Target::phi)) {
    timer.tick();
    features.sample(theta, contributions_gene_type, spot_scaling,
                    experiment_scaling_long);
    if (timing and verbosity >= Verbosity::Info)
      std::cout << "This took " << timer.tock() << "μs." << std::endl;
    if (verbosity >= Verbosity::Everything)
      std::cout << "Log-likelihood = " << log_likelihood(data.counts)
                << std::endl;
    check_model(data.counts);
  }

  if (flagged(which & Target::theta)) {
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
void Model<kind>::sample_split_merge(const Counts &data, Target which) {
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
size_t Model<kind>::find_weakest_factor() const {
  std::vector<Float> x(T, 0);
  std::cout << "Factor strengths: ";
  auto phi_marginal = features.marginalize_genes();
  for (size_t t = 0; t < T; ++t) {
    for (size_t s = 0; s < S; ++s) {
      Float z = phi_marginal[t] * theta(s, t) * spot_scaling[s];
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
Model<kind> Model<kind>::run_submodel(size_t t, size_t n, const Counts &counts,
                                      Target which, const std::string &prefix,
                                      const std::vector<size_t> &init_factors) {
  const bool show_timing = false;
  // TODO: use init_factors
  Model<kind> sub_model(counts, t, parameters, Verbosity::Info);
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
          & ~(Target::spot_scaling | Target::experiment_scaling
              | Target::merge_split);
  for (size_t i = 0; i < n; ++i)
    sub_model.gibbs_sample(counts, which, show_timing);

  if (print_sub_model_cnt)
    sub_model.store(counts,
                    prefix + "submodel_opti_" + std::to_string(sub_model_cnt));
  // sub_model_cnt++; // TODO

  return sub_model;
}

template <Kind kind>
void Model<kind>::lift_sub_model(const Model &sub_model,
                                        size_t t1, size_t t2) {
  features.prior.lift_sub_model(sub_model.features.prior, t1, t2);
  for (size_t g = 0; g < G; ++g) {
    features.phi(g, t1) = sub_model.features.phi(g, t2);
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
void Model<kind>::sample_split(const Counts &data, size_t t1, Target which) {
  size_t t2 = find_weakest_factor();
  if (verbosity >= Verbosity::Info)
    std::cout << "Performing a split step. Splitting " << t1 << " and " << t2
              << "." << std::endl;
  Model previous(*this);

  double ll_previous = (consider_factor_likel
                            ? log_likelihood_factor(data.counts, t1)
                                  + log_likelihood_factor(data.counts, t2)
                            : 0)
                       + log_likelihood_poisson_counts(data.counts);

  Counts sub_counts = data;
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t g = 0; g < G; ++g)
    for (size_t s = 0; s < S; ++s) {
      Float lambda = phi(g, t1) * theta(s, t1) + phi(g, t2) * theta(s, t2);
      sub_counts.counts(g, s) = std::binomial_distribution<Int>(
          data.counts(g, s),
          lambda / lambda_gene_spot(g, s))(EntropySource::rng);
      // remove effect of current parameters
      lambda_gene_spot(g, s) -= lambda;
    }

  Model sub_model
      = run_submodel(2, num_sub_gibbs_split, sub_counts, which, "splitmerge_split_");

  lift_sub_model(sub_model, t1, 0);
  lift_sub_model(sub_model, t2, 1);

  // add effect of updated parameters
#pragma omp parallel for if (DO_PARALLEL)
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
void Model<kind>::sample_merge(const Counts &data, size_t t1, size_t t2,
                               Target which) {
  if (verbosity >= Verbosity::Info)
    std::cout << "Performing a merge step. Merging types " << t1 << " and "
              << t2 << "." << std::endl;
  Model previous(*this);

  double ll_previous = (consider_factor_likel
                            ? log_likelihood_factor(data.counts, t1)
                                  + log_likelihood_factor(data.counts, t2)
                            : 0)
                       + log_likelihood_poisson_counts(data.counts);

  Counts sub_counts = data;
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t g = 0; g < G; ++g)
    for (size_t s = 0; s < S; ++s) {
      Float lambda = phi(g, t1) * theta(s, t1) + phi(g, t2) * theta(s, t2);
      sub_counts.counts(g, s) = std::binomial_distribution<Int>(
          data.counts(g, s),
          lambda / lambda_gene_spot(g, s))(EntropySource::rng);
      // remove effect of current parameters
      lambda_gene_spot(g, s) -= lambda;
    }

  Model sub_model
      = run_submodel(1, num_sub_gibbs_merge, sub_counts, which, "splitmerge_merge_");

  lift_sub_model(sub_model, t1, 0);

  // add effect of updated parameters
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t g = 0; g < G; ++g)
    for (size_t s = 0; s < S; ++s)
      lambda_gene_spot(g, s) += phi(g, t1) * theta(s, t1);

  features.initialize_factor(t2);

  // randomly initialize p_theta
  if (verbosity >= Verbosity::Debug)
    std::cout << "initializing p of theta." << std::endl;
  if (true)  // TODO make this CLI-switchable
    p_theta[t2] = prob_to_neg_odds(sample_beta<Float>(
        parameters.hyperparameters.theta_p_1, parameters.hyperparameters.theta_p_2));
  else
    p_theta[t2] = 1;

  // initialize r_theta
  if (verbosity >= Verbosity::Debug)
    std::cout << "initializing r of theta." << std::endl;
  // NOTE: std::gamma_distribution takes a shape and scale parameter
  r_theta[t2] = std::gamma_distribution<Float>(
      parameters.hyperparameters.theta_r_1,
      1 / parameters.hyperparameters.theta_r_2)(EntropySource::rng);

  // initialize theta
  if (verbosity >= Verbosity::Debug)
    std::cout << "initializing theta." << std::endl;
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t s = 0; s < S; ++s)
    // NOTE: std::gamma_distribution takes a shape and scale parameter
    theta(s, t2) = std::gamma_distribution<Float>(
        r_theta(t2), 1 / p_theta(t2))(EntropySource::rng);

  // add effect of updated parameters
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t g = 0; g < G; ++g)
    for (size_t s = 0; s < S; ++s)
      lambda_gene_spot(g, s) += phi(g, t2) * theta(s, t2);

  for (size_t g = 0; g < G; ++g)
    contributions_gene_type(g, t2) = 0;
  for (size_t s = 0; s < S; ++s)
    contributions_spot_type(s, t2) = 0;
#pragma omp parallel for if (DO_PARALLEL)
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
std::vector<Int> Model<kind>::sample_reads(size_t g, size_t s, size_t n) const {
  std::cout << "Error: not implemented: sampling reads." << std::endl;
  exit(-1);
}

template <>
std::vector<Int> Model<Kind::Gamma>::sample_reads(size_t g, size_t s,
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
          features.prior.r(g, t),
          prods[t] / (prods[t] + features.prior.p(g, t)), EntropySource::rng);
  return v;
}

template <Kind kind>
double Model<kind>::posterior_expectation(size_t g, size_t s) const {
  std::cout << "Error: not implemented: computing posterior expectations."
            << std::endl;
  exit(-1);
}

template <>
double Model<Kind::Gamma>::posterior_expectation(size_t g, size_t s) const {
  double x = 0;
  for (size_t t = 0; t < T; ++t)
    x += features.prior.r(g, t) / features.prior.p(g, t) * theta(s, t);
  x *= spot_scaling[s];
  if (parameters.activate_experiment_scaling)
    x *= experiment_scaling_long[s];
  return x;
}

template <Kind kind>
double Model<kind>::posterior_expectation_poisson(size_t g,
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
double Model<kind>::posterior_variance(size_t g, size_t s) const {
  std::cout << "Error: not implemented: computing posterior variance."
            << std::endl;
  exit(-1);
}

template <>
double Model<Kind::Gamma>::posterior_variance(size_t g, size_t s) const {
  double x = 0;
  double prod_ = spot_scaling[s];
  if (parameters.activate_experiment_scaling)
    prod_ *= experiment_scaling_long[s];
  for (size_t t = 0; t < T; ++t) {
    double prod = theta(s, t) * prod_;
    x += features.prior.r(g, t) * prod / (prod + features.prior.p(g, t)) /
  features.prior.p(g, t) / features.prior.p(g, t);
  }
  return x;
}

template <Kind kind>
Matrix Model<kind>::posterior_expectations() const {
  Matrix m(G, S);
  for (size_t g = 0; g < G; ++g)
    for (size_t s = 0; s < S; ++s)
      m(g, s) = posterior_expectation(g, s);
  return m;
}

template <Kind kind>
Matrix Model<kind>::posterior_expectations_poisson() const {
  Matrix m(G, S);
  for (size_t g = 0; g < G; ++g)
    for (size_t s = 0; s < S; ++s)
      m(g, s) = posterior_expectation_poisson(g, s);
  return m;
}

template <Kind kind>
Matrix Model<kind>::posterior_variances() const {
  Matrix m(G, S);
  for (size_t g = 0; g < G; ++g)
    for (size_t s = 0; s < S; ++s)
      m(g, s) = posterior_variance(g, s);
  return m;
}

template <Kind kind>
void Model<kind>::check_model(const IMatrix &counts) const {
  return;
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
      /*
      if (features.prior.p(g, t) < 0)
        throw(std::runtime_error("P[" + std::to_string(g) + "]["
                                 + std::to_string(t) + "] is smaller zero: p="
                                 + std::to_string(features.prior.p(g, t)) + "."));
      if (features.prior.p(g, t) == 0)
        throw(std::runtime_error("P is zero for gene " + std::to_string(g)
                                 + " in factor " + std::to_string(t) + "."));

      if (features.prior.r(g, t) < 0)
        throw(std::runtime_error("R[" + std::to_string(g) + "]["
                                 + std::to_string(t) + "] is smaller zero: r="
                                 + std::to_string(features.prior.r(g, t)) + "."));
      if (features.prior.r(g, t) == 0)
        throw(std::runtime_error("R is zero for gene " + std::to_string(g)
                                 + " in factor " + std::to_string(t) + "."));
      */
    }

  // check hyper-parameters
  if (parameters.hyperparameters.phi_r_1 == 0)
    throw(std::runtime_error("The prior phi_r_1 is zero."));
  if (parameters.hyperparameters.phi_r_2 == 0)
    throw(std::runtime_error("The prior phi_r_2 is zero."));
  if (parameters.hyperparameters.phi_p_1 == 0)
    throw(std::runtime_error("The prior phi_p_1 is zero."));
  if (parameters.hyperparameters.phi_p_2 == 0)
    throw(std::runtime_error("The prior phi_p_2 is zero."));
  if (parameters.hyperparameters.alpha == 0)
    throw(std::runtime_error("The prior alpha is zero."));
}

template <Kind kind>
std::ostream &operator<<(std::ostream &os,
                         const PoissonFactorization::Model<kind> &pfa) {
  os << "Poisson Factorization "
     << "G = " << pfa.G << " "
     << "S = " << pfa.S << " "
     << "T = " << pfa.T << std::endl;

  if (pfa.verbosity >= Verbosity::Verbose) {
    print_matrix_head(os, pfa.features.phi, "Φ");
    print_matrix_head(os, pfa.theta, "Θ");
    os << pfa.features.prior;

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
  }

  return os;
}
}

#endif
