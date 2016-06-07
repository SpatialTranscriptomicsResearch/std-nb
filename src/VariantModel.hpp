#ifndef VARIANTMODEL_HPP
#define VARIANTMODEL_HPP

#include <random>
#include <omp.h>
#include "counts.hpp"
#include "entropy.hpp"
#include "parallel.hpp"
#include "parameters.hpp"
#include "compression.hpp"
#include "log.hpp"
#include "features.hpp"
#include "mix.hpp"
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
#define print_sub_model_cnt false  // TODO make configurable

const size_t num_sub_gibbs_split = 50;
const size_t num_sub_gibbs_merge = 10;
const bool consider_factor_likel = false;
const size_t sub_model_cnt = 10;
// static size_t sub_model_cnt; // TODO

bool gibbs_test(Float nextG, Float G, Verbosity verbosity,
                Float temperature = 50);
size_t num_lines(const std::string &path);

struct Paths {
  Paths(const std::string &prefix, const std::string &suffix = "");
  std::string phi, theta, spot, experiment, r_phi, p_phi, r_theta, p_theta;
  std::string contributions_gene_type, contributions_spot_type,
      contributions_spot, contributions_experiment;
};

template <Feature::Kind feat_kind = Feature::Kind::Gamma,
          Mix::Kind mix_kind = Mix::Kind::Gamma>
struct Model {
  typedef Feature::Features<feat_kind> features_t;
  typedef Mix::Weights<mix_kind> weights_t;
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
  /** factor score matrix */
  weights_t weights;

  inline Float &phi(size_t x, size_t y) { return features.phi(x, y); };
  inline Float phi(size_t x, size_t y) const { return features.phi(x, y); };

  inline Float &theta(size_t x, size_t y) { return weights.theta(x, y); };
  inline Float theta(size_t x, size_t y) const { return weights.theta(x, y); };

  /** spot scaling vector */
  Vector spot_scaling;

  /** experiment scaling vector */
  Vector experiment_scaling;
  Vector experiment_scaling_long;

  Verbosity verbosity;

  Model(const Counts &counts, const size_t T, const Parameters &parameters,
        Verbosity verbosity);

  Model(const Counts &counts, const Paths &paths, const Parameters &parameters,
        Verbosity verbosity);

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

template <Feature::Kind feat_kind, Mix::Kind mix_kind>
std::ostream &operator<<(
    std::ostream &os,
    const PoissonFactorization::Model<feat_kind, mix_kind> &pfa);

template <Feature::Kind feat_kind, Mix::Kind mix_kind>
Model<feat_kind, mix_kind>::Model(const Counts &c, const size_t T_,
                                  const Parameters &parameters_,
                                  Verbosity verbosity_)
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
      weights(G, S, T, parameters),
      spot_scaling(S, arma::fill::ones),
      experiment_scaling(E, arma::fill::ones),
      experiment_scaling_long(S, arma::fill::ones),
      verbosity(verbosity_) {
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
      LOG(debug) << "Initializing experiment scaling.";
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
    LOG(debug) << "Initializing spot scaling.";
    Float z = 0;
    for (size_t s = 0; s < S; ++s)
      z += spot_scaling(s) = contributions_spot(s) / experiment_scaling_long(s);
    z /= S;
    for (size_t s = 0; s < S; ++s)
      spot_scaling(s) /= z;
  }
}

template <Feature::Kind feat_kind, Mix::Kind mix_kind>
Model<feat_kind,mix_kind>::Model(const Counts &c, const Paths &paths,
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
      features(G, S, T, parameters), // TODO deactivate
      weights(G, S, T, parameters), // TODO deactivate
      // features(parse_file<Matrix>(paths.features, read_matrix, DEFAULT_SEPARATOR, DEFAULT_LABEL)), // TODO reactivate
      // theta(parse_file<Matrix>(paths.theta, read_matrix, DEFAULT_SEPARATOR, DEFAULT_LABEL)), // TODO reactivate
      // r_theta(parse_file<Vector>(paths.r_theta, read_vector<Vector>, DEFAULT_SEPARATOR)),
      // p_theta(parse_file<Vector>(paths.p_theta, read_vector<Vector>, DEFAULT_SEPARATOR)),
      spot_scaling(parse_file<Vector>(paths.spot, read_vector<Vector>, DEFAULT_SEPARATOR)),
      experiment_scaling(parse_file<Vector>(paths.experiment, read_vector<Vector>, DEFAULT_SEPARATOR)),
      experiment_scaling_long(S),
      verbosity(verbosity_) {
  update_experiment_scaling_long(c);

  LOG(debug) << *this;
}

template <Feature::Kind feat_kind, Mix::Kind mix_kind>
// TODO ensure no NaNs or infinities are generated
double Model<feat_kind, mix_kind>::log_likelihood_factor(const IMatrix &counts,
                                                         size_t t) const {
  double l = features.log_likelihood_factor(counts, t)
             + weights.log_likelihood_factor(counts, t);

  if (std::isnan(l) or std::isinf(l))
    LOG(warning) << "Warning: log likelihoood contribution of factor " << t
              << " = " << l;

  LOG(debug) << "ll_X = " << l;

  return l;
}

template <Feature::Kind feat_kind, Mix::Kind mix_kind>
double Model<feat_kind, mix_kind>::log_likelihood_poisson_counts(
    const IMatrix &counts) const {
  double l = 0;
#pragma omp parallel for reduction(+ : l) if (DO_PARALLEL)
  for (size_t g = 0; g < G; ++g)
    for (size_t s = 0; s < S; ++s) {
      double rate = lambda_gene_spot(g, s) * spot_scaling(s);
      if (parameters.activate_experiment_scaling)
        rate *= experiment_scaling_long(s);
      auto cur = log_poisson(counts(g, s), rate);
      if (std::isinf(cur) or std::isnan(cur))
        LOG(warning) << "ll poisson(g=" << g << ",s=" << s << ") = " << cur
                     << " counts = " << counts(g, s)
                     << " lambda = " << lambda_gene_spot(g, s)
                     << " rate = " << rate;
      l += cur;
    }
  return l;
}

template <Feature::Kind feat_kind, Mix::Kind mix_kind>
double Model<feat_kind, mix_kind>::log_likelihood(const IMatrix &counts) const {
  double l = 0;
  for (size_t t = 0; t < T; ++t)
    l += log_likelihood_factor(counts, t);

  for (size_t s = 0; s < S; ++s)
    l += log_gamma(spot_scaling(s), parameters.hyperparameters.spot_a,
                   1.0 / parameters.hyperparameters.spot_b);
  if (parameters.activate_experiment_scaling) {
    for (size_t e = 0; e < E; ++e)
      l += log_gamma(experiment_scaling(e),
                     parameters.hyperparameters.experiment_a,
                     1.0 / parameters.hyperparameters.experiment_b);
  }

  l += log_likelihood_poisson_counts(counts);

  return l;
}

template <Feature::Kind feat_kind, Mix::Kind mix_kind>
Matrix Model<feat_kind, mix_kind>::weighted_theta() const {
  Matrix m = weights.theta;
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

template <Feature::Kind feat_kind, Mix::Kind mix_kind>
void Model<feat_kind, mix_kind>::store(const Counts &counts,
                                       const std::string &prefix,
                                       bool mean_and_variance) const {
  std::vector<std::string> factor_names;
  for (size_t t = 1; t <= T; ++t)
    factor_names.push_back("Factor " + std::to_string(t));
  auto &gene_names = counts.row_names;
  auto &spot_names = counts.col_names;
  features.store(prefix, gene_names, factor_names);
  weights.store(prefix, spot_names, factor_names);
  write_matrix(weighted_theta(), prefix + "weighted_theta.txt", spot_names, factor_names);
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

template <Feature::Kind feat_kind, Mix::Kind mix_kind>
void Model<feat_kind, mix_kind>::sample_contributions_sub(
    const IMatrix &counts, size_t g, size_t s, RNG &rng,
    IMatrix &contrib_gene_type, IMatrix &contrib_spot_type) {
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

template <Feature::Kind feat_kind, Mix::Kind mix_kind>
/** sample count decomposition */
void Model<feat_kind, mix_kind>::sample_contributions(const IMatrix &counts) {
  LOG(info) << "Sampling contributions";
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

/** sample spot scaling factors */
template <Feature::Kind feat_kind, Mix::Kind mix_kind>
void Model<feat_kind, mix_kind>::sample_spot_scaling() {
  LOG(info) << "Sampling spot scaling factors";
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
        1.0 / (parameters.hyperparameters.spot_b + intensity_sum))(
        EntropySource::rng);
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
template <Feature::Kind feat_kind, Mix::Kind mix_kind>
void Model<feat_kind, mix_kind>::sample_experiment_scaling(const Counts &data) {
  LOG(info) << "Sampling experiment scaling factors";

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
    for (size_t e = 0; e < E; ++e) {
      LOG(debug) << "contributions_experiment[" << e
                 << "]=" << contributions_experiment[e];
      LOG(debug) << "intensity_sum=" << intensity_sums[e];
      LOG(debug) << "prev experiment_scaling[" << e
                 << "]=" << experiment_scaling[e];
    }

  for (size_t e = 0; e < E; ++e) {
    // NOTE: std::gamma_distribution takes a shape and scale parameter
    experiment_scaling[e] = std::gamma_distribution<Float>(
        parameters.hyperparameters.experiment_a + contributions_experiment(e),
        1.0 / (parameters.hyperparameters.experiment_b + intensity_sums[e]))(
        EntropySource::rng);
    LOG(debug) << "new experiment_scaling[" << e
               << "]=" << experiment_scaling[e];
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
template <Feature::Kind feat_kind, Mix::Kind mix_kind>
void Model<feat_kind, mix_kind>::update_experiment_scaling_long(
    const Counts &data) {
  for (size_t s = 0; s < S; ++s)
    experiment_scaling_long[s] = experiment_scaling[data.experiments[s]];
}

template <Feature::Kind feat_kind, Mix::Kind mix_kind>
void Model<feat_kind, mix_kind>::gibbs_sample(const Counts &data, Target which,
                                              bool timing) {
  check_model(data.counts);

  Timer timer;
  if (flagged(which & Target::contributions)) {
    sample_contributions(data.counts);
    if (timing)
      LOG(info) << "This took " << timer.tock() << "μs.";
    LOG(debug) << "Log-likelihood = " << log_likelihood(data.counts);
    check_model(data.counts);
  }

  if (flagged(which & Target::merge_split)) {
    // NOTE: this has to be done right after the Gibbs step for the
    // contributions because otherwise lambda_gene_spot is inconsistent
    timer.tick();
    sample_split_merge(data, which);
    if (timing)
      LOG(info) << "This took " << timer.tock() << "μs.";
    LOG(debug) << "Log-likelihood = " << log_likelihood(data.counts);
    check_model(data.counts);
  }

  if (flagged(which & Target::spot_scaling)) {
    timer.tick();
    sample_spot_scaling();
    if (timing)
      LOG(info) << "This took " << timer.tock() << "μs.";
    LOG(debug) << "Log-likelihood = " << log_likelihood(data.counts);
    check_model(data.counts);
  }

  if (flagged(which & Target::experiment_scaling)) {
    if (E > 1 and parameters.activate_experiment_scaling) {
      timer.tick();
      sample_experiment_scaling(data);
      if (timing)
        LOG(info) << "This took " << timer.tock() << "μs.";
      LOG(debug) << "Log-likelihood = " << log_likelihood(data.counts);
      check_model(data.counts);
    }
  }

  if (flagged(which & (Target::phi_r | Target::phi_p))) {
    timer.tick();
    features.prior.sample(weights.theta, contributions_gene_type, spot_scaling,
                          experiment_scaling_long);
    if (timing)
      LOG(info) << "This took " << timer.tock() << "μs.";
    LOG(debug) << "Log-likelihood = " << log_likelihood(data.counts);
    check_model(data.counts);
  }

  if (flagged(which & (Target::theta_r | Target::theta_p))) {
    timer.tick();
    weights.prior.sample(features.phi, contributions_spot_type, spot_scaling,
                         experiment_scaling_long);
    if (timing)
      LOG(info) << "This took " << timer.tock() << "μs.";
    LOG(debug) << "Log-likelihood = " << log_likelihood(data.counts);
    check_model(data.counts);
  }

  if (flagged(which & Target::phi)) {
    timer.tick();
    features.sample(weights.theta, contributions_gene_type, spot_scaling,
                    experiment_scaling_long);
    if (timing)
      LOG(info) << "This took " << timer.tock() << "μs.";
    LOG(debug) << "Log-likelihood = " << log_likelihood(data.counts);
    check_model(data.counts);
  }

  if (flagged(which & Target::theta)) {
    timer.tick();
    weights.sample(features, contributions_spot_type, spot_scaling,
                   experiment_scaling_long);
    if (timing)
      LOG(info) << "This took " << timer.tock() << "μs.";
    LOG(debug) << "Log-likelihood = " << log_likelihood(data.counts);
    check_model(data.counts);
  }
}

template <Feature::Kind feat_kind, Mix::Kind mix_kind>
void Model<feat_kind, mix_kind>::sample_split_merge(const Counts &data,
                                                    Target which) {
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

template <Feature::Kind feat_kind, Mix::Kind mix_kind>
size_t Model<feat_kind, mix_kind>::find_weakest_factor() const {
  std::vector<Float> x(T, 0);
  auto phi_marginal = features.marginalize_genes();
  for (size_t t = 0; t < T; ++t)
    for (size_t s = 0; s < S; ++s) {
      Float z = phi_marginal[t] * theta(s, t) * spot_scaling[s];
      if (parameters.activate_experiment_scaling)
        z *= experiment_scaling_long[s];
      x[t] += z;
    }
  return std::distance(begin(x), min_element(begin(x), end(x)));
}

template <Feature::Kind feat_kind, Mix::Kind mix_kind>
Model<feat_kind, mix_kind> Model<feat_kind, mix_kind>::run_submodel(
    size_t t, size_t n, const Counts &counts, Target which,
    const std::string &prefix, const std::vector<size_t> &init_factors) {
  const bool show_timing = false;
  // TODO: use init_factors
  Model<feat_kind, mix_kind> sub_model(counts, t, parameters, Verbosity::Info);
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

  bool prev_logging = boost::log::core::get()->set_logging_enabled(false);

  for (size_t i = 0; i < n; ++i) {
    sub_model.gibbs_sample(counts, which, show_timing);
    LOG(info) << "sub model log likelihood = "
              << sub_model.log_likelihood_poisson_counts(counts.counts);
  }

  boost::log::core::get()->set_logging_enabled(prev_logging);

  if (print_sub_model_cnt)
    sub_model.store(counts,
                    prefix + "submodel_opti_" + std::to_string(sub_model_cnt));
  // sub_model_cnt++; // TODO

  return sub_model;
}

template <Feature::Kind feat_kind, Mix::Kind mix_kind>
void Model<feat_kind, mix_kind>::lift_sub_model(const Model &sub_model,
                                                size_t t1, size_t t2) {
  features.lift_sub_model(sub_model.features, t1, t2);
  weights.lift_sub_model(sub_model.weights, t1, t2);

  for (size_t g = 0; g < G; ++g)
    contributions_gene_type(g, t1) = sub_model.contributions_gene_type(g, t2);

  for (size_t s = 0; s < S; ++s)
    contributions_spot_type(s, t1) = sub_model.contributions_spot_type(s, t2);
}

template <Feature::Kind feat_kind, Mix::Kind mix_kind>
void Model<feat_kind, mix_kind>::sample_split(const Counts &data, size_t t1,
                                              Target which) {
  size_t t2 = find_weakest_factor();
  LOG(info) << "Performing a split step: " << t1 << " and " << t2 << ".";
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

  Model sub_model = run_submodel(2, num_sub_gibbs_split, sub_counts, which,
                                 "splitmerge_split_");

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

  LOG(debug) << "ll_split_previous = " << ll_previous
            << " ll_split_updated = " << ll_updated;
  if (gibbs_test(ll_updated, ll_previous, verbosity)) {
    LOG(info) << "Split step accecpted";
  } else {
    *this = previous;
    LOG(info) << "Split step rejected";
  }
}

template <Feature::Kind feat_kind, Mix::Kind mix_kind>
void Model<feat_kind, mix_kind>::sample_merge(const Counts &data, size_t t1,
                                              size_t t2, Target which) {
  LOG(info) << "Performing a merge step: " << t1 << " and " << t2 << ".";
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

  Model sub_model = run_submodel(1, num_sub_gibbs_merge, sub_counts, which,
                                 "splitmerge_merge_");

  lift_sub_model(sub_model, t1, 0);

  // add effect of updated parameters
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t g = 0; g < G; ++g)
    for (size_t s = 0; s < S; ++s)
      lambda_gene_spot(g, s) += phi(g, t1) * theta(s, t1);

  features.initialize_factor(t2);

  // randomly initialize p_theta
  LOG(debug) << "Initializing P of Θ";
  if (true)  // TODO make this CLI-switchable
    weights.prior.p[t2] = prob_to_neg_odds(
        sample_beta<Float>(parameters.hyperparameters.theta_p_1,
                           parameters.hyperparameters.theta_p_2));
  else
    weights.prior.p[t2] = 1;

  // initialize r_theta
  LOG(debug) << "Initializing R of Θ";
  // NOTE: std::gamma_distribution takes a shape and scale parameter
  weights.prior.r[t2] = std::gamma_distribution<Float>(
      parameters.hyperparameters.theta_r_1,
      1 / parameters.hyperparameters.theta_r_2)(EntropySource::rng);

  // initialize theta
  LOG(debug) << "Initializing Θ";
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t s = 0; s < S; ++s)
    // NOTE: std::gamma_distribution takes a shape and scale parameter
    theta(s, t2) = std::gamma_distribution<Float>(
        weights.prior.r(t2), 1 / weights.prior.p(t2))(EntropySource::rng);

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

  LOG(debug) << "ll_merge_previous = " << ll_previous
            << " ll_merge_updated = " << ll_updated;
  if (gibbs_test(ll_updated, ll_previous, verbosity)) {
    LOG(info) << "Merge step accepted";
  } else {
    *this = previous;
    LOG(info) << "Merge step rejected";
  }
}

template <Feature::Kind feat_kind, Mix::Kind mix_kind>
std::vector<Int> Model<feat_kind, mix_kind>::sample_reads(size_t g, size_t s,
                                                          size_t n) const {
  LOG(fatal) << "Error: not implemented: sampling reads.";
  exit(-1);
}

/*
 TODO reactivate
template <>
template <Mix::Kind mix_kind>
std::vector<Int> Model<Feature::Kind::Gamma, mix_kind>::sample_reads(size_t g, size_t s,
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
*/

template <Feature::Kind feat_kind, Mix::Kind mix_kind>
double Model<feat_kind, mix_kind>::posterior_expectation(size_t g,
                                                         size_t s) const {
  LOG(fatal) << "Error: not implemented: computing posterior expectations.";
  exit(-1);
}

/*
 TODO reactivate
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
*/

template <Feature::Kind feat_kind, Mix::Kind mix_kind>
double Model<feat_kind, mix_kind>::posterior_expectation_poisson(
    size_t g, size_t s) const {
  double x = 0;
  for (size_t t = 0; t < T; ++t)
    x += phi(g, t) * theta(s, t);
  x *= spot_scaling[s];
  if (parameters.activate_experiment_scaling)
    x *= experiment_scaling_long[s];
  return x;
}

template <Feature::Kind feat_kind, Mix::Kind mix_kind>
double Model<feat_kind, mix_kind>::posterior_variance(size_t g,
                                                      size_t s) const {
  LOG(fatal) << "Error: not implemented: computing posterior variance.";
  exit(-1);
}

/*
 TODO reactivate
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
*/

template <Feature::Kind feat_kind, Mix::Kind mix_kind>
Matrix Model<feat_kind, mix_kind>::posterior_expectations() const {
  Matrix m(G, S);
  for (size_t g = 0; g < G; ++g)
    for (size_t s = 0; s < S; ++s)
      m(g, s) = posterior_expectation(g, s);
  return m;
}

template <Feature::Kind feat_kind, Mix::Kind mix_kind>
Matrix Model<feat_kind, mix_kind>::posterior_expectations_poisson() const {
  Matrix m(G, S);
  for (size_t g = 0; g < G; ++g)
    for (size_t s = 0; s < S; ++s)
      m(g, s) = posterior_expectation_poisson(g, s);
  return m;
}

template <Feature::Kind feat_kind, Mix::Kind mix_kind>
Matrix Model<feat_kind, mix_kind>::posterior_variances() const {
  Matrix m(G, S);
  for (size_t g = 0; g < G; ++g)
    for (size_t s = 0; s < S; ++s)
      m(g, s) = posterior_variance(g, s);
  return m;
}

template <Feature::Kind feat_kind, Mix::Kind mix_kind>
void Model<feat_kind, mix_kind>::check_model(const IMatrix &counts) const {
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

template <Feature::Kind feat_kind, Mix::Kind mix_kind>
std::ostream &operator<<(
    std::ostream &os,
    const PoissonFactorization::Model<feat_kind, mix_kind> &pfa) {
  os << "Poisson Factorization "
     << "G = " << pfa.G << " "
     << "S = " << pfa.S << " "
     << "T = " << pfa.T << std::endl;

  if (pfa.verbosity >= Verbosity::Verbose) {
    print_matrix_head(os, pfa.features.phi, "Φ");
    print_matrix_head(os, pfa.weights.theta, "Θ");
    os << pfa.features.prior;
    os << pfa.weights.prior;

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

    /*
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
    */
  }

  return os;
}
}

#endif
