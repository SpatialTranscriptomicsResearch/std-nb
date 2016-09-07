#ifndef MODEL_HPP
#define MODEL_HPP

#include <random>
#include "counts.hpp"
#include "entropy.hpp"
#include "parallel.hpp"
#include "parameters.hpp"
#include "compression.hpp"
#include "log.hpp"
#include "PartialModel.hpp"
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

const size_t num_sub_gibbs_split = 10;
// TODO consider lowering the number of Gibbs steps when merging with Dirichlet factors
const size_t num_sub_gibbs_merge = 10;
const bool consider_factor_likel = false;
const size_t sub_model_cnt = 10;
// static size_t sub_model_cnt; // TODO

bool gibbs_test(Float nextG, Float G, Float temperature = 50);
size_t num_lines(const std::string &path);

struct Paths {
  Paths(const std::string &prefix, const std::string &suffix = "");
  std::string phi, theta, spot, experiment, r_phi, p_phi, r_theta, p_theta;
  std::string contributions_gene_type, contributions_spot_type,
      contributions_gene, contributions_spot, contributions_experiment;
};

template <Partial::Kind feat_kind = Partial::Kind::Gamma,
          Partial::Kind mix_kind = Partial::Kind::HierGamma>
struct Model {
  typedef Partial::Model<Partial::Variable::Feature, feat_kind> features_t;
  typedef Partial::Model<Partial::Variable::Mix, mix_kind> weights_t;
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
  IVector contributions_gene, contributions_spot, contributions_experiment;

  /** Normalizing factor to translate Poisson rates \lambda_{xgst} to relative
   * frequencies \lambda_{gst} / z_{gs} for the multionomial distribution */
  Matrix lambda_gene_spot;

  /** factor loading matrix */
  features_t features;
  /** factor score matrix */
  weights_t weights;

  inline Float &phi(size_t x, size_t y) { return features.matrix(x, y); };
  inline Float phi(size_t x, size_t y) const { return features.matrix(x, y); };

  inline Float &theta(size_t x, size_t y) { return weights.matrix(x, y); };
  inline Float theta(size_t x, size_t y) const { return weights.matrix(x, y); };

  /** spot scaling vector */
  Vector spot;

  /** experiment scaling vector */
  Vector experiment_scaling;
  Vector experiment_scaling_long;

  Model(const Counts &counts, const size_t T, const Parameters &parameters);
  Model(const Counts &counts, const Paths &paths, const Parameters &parameters);

  void store(const Counts &counts, const std::string &prefix,
             bool mean_and_variance = false) const;

  Matrix weighted_theta() const;

  double log_likelihood(const IMatrix &counts) const;
  double log_likelihood_factor(const IMatrix &counts, size_t t) const;
  double log_likelihood_poisson_counts(const IMatrix &counts) const;

  /** sample count decomposition */
  void sample_contributions(const IMatrix &counts);
  void sample_contributions_variational(const IMatrix &counts);
  void sample_contributions_sub(const IMatrix &counts, size_t g, size_t s,
                                RNG &rng, IMatrix &contrib_gene_type,
                                IMatrix &contrib_spot_type);

  /** sample spot scaling factors */
  void sample_spot();

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

  double posterior_expectation_poisson(size_t g, size_t s) const;
  Matrix posterior_expectations_poisson() const;

  /** check that parameter invariants are fulfilled */
  void check_model(const IMatrix &counts) const;

private:
  void update_experiment_scaling_long(const Counts &data);
};

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
Model<feat_kind, mix_kind> operator*(const Model<feat_kind, mix_kind> &a,
                                     const Model<feat_kind, mix_kind> &b);

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
Model<feat_kind, mix_kind> operator+(const Model<feat_kind, mix_kind> &a,
                                     const Model<feat_kind, mix_kind> &b);

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
Model<feat_kind, mix_kind> operator-(const Model<feat_kind, mix_kind> &a,
                                     const Model<feat_kind, mix_kind> &b);

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
Model<feat_kind, mix_kind> operator*(const Model<feat_kind, mix_kind> &a,
                                     double x);

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
Model<feat_kind, mix_kind> operator/(const Model<feat_kind, mix_kind> &a,
                                     double x);

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
std::ostream &operator<<(
    std::ostream &os,
    const Model<feat_kind, mix_kind> &pfa);

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
Model<feat_kind, mix_kind>::Model(const Counts &c, const size_t T_,
                                  const Parameters &parameters_)
    : G(c.counts.n_rows),
      S(c.counts.n_cols),
      T(T_),
      E(c.experiment_names.size()),
      parameters(parameters_),
      contributions_gene_type(G, T, arma::fill::zeros),
      contributions_spot_type(S, T, arma::fill::zeros),
      contributions_gene(G, arma::fill::zeros),
      contributions_spot(S, arma::fill::zeros),
      contributions_experiment(E, arma::fill::zeros),
      lambda_gene_spot(G, S, arma::fill::zeros),
      features(G, S, T, parameters),
      weights(G, S, T, parameters),
      spot(S, arma::fill::ones),
      experiment_scaling(E, arma::fill::ones),
      experiment_scaling_long(S, arma::fill::ones) {
  if (false) {
    // initialize:
    //  * contributions_gene_type
    //  * contributions_spot_type
    //  * lambda_gene_spot
    LOG(debug) << "Initializing contributions.";
    sample_contributions(c.counts);
  }

// initialize:
//  * contributions_spot
//  * contributions_experiment
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t s = 0; s < S; ++s)
    for (size_t g = 0; g < G; ++g) {
      contributions_spot(s) += c.counts(g, s);
      contributions_experiment(c.experiments[s]) += c.counts(g, s);
    }
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t g = 0; g < G; ++g)
    for (size_t s = 0; s < S; ++s)
      contributions_gene(g) += c.counts(g, s);

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
      z += spot(s) = contributions_spot(s) / experiment_scaling_long(s);
    z /= S;
    for (size_t s = 0; s < S; ++s)
      spot(s) /= z;
  }
}

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
Model<feat_kind, mix_kind>::Model(const Counts &c, const Paths &paths,
                                  const Parameters &parameters_)
    : G(c.counts.n_rows),
      S(c.counts.n_cols),
      T(num_lines(paths.r_theta)),
      E(c.experiment_names.size()),
      parameters(parameters_),
      contributions_gene_type(parse_file<IMatrix>(paths.contributions_gene_type, read_imatrix, DEFAULT_SEPARATOR, DEFAULT_LABEL)),
      contributions_spot_type(parse_file<IMatrix>(paths.contributions_spot_type, read_imatrix, DEFAULT_SEPARATOR, DEFAULT_LABEL)),
      contributions_gene(parse_file<IVector>(paths.contributions_gene, read_vector<IVector>, DEFAULT_SEPARATOR)),
      contributions_spot(parse_file<IVector>(paths.contributions_spot, read_vector<IVector>, DEFAULT_SEPARATOR)),
      contributions_experiment(parse_file<IVector>(paths.contributions_experiment, read_vector<IVector>, DEFAULT_SEPARATOR)),
      features(G, S, T, parameters), // TODO deactivate
      weights(G, S, T, parameters), // TODO deactivate
      // features(parse_file<Matrix>(paths.features, read_matrix, DEFAULT_SEPARATOR, DEFAULT_LABEL)), // TODO reactivate
      // theta(parse_file<Matrix>(paths.theta, read_matrix, DEFAULT_SEPARATOR, DEFAULT_LABEL)), // TODO reactivate
      // r_theta(parse_file<Vector>(paths.r_theta, read_vector<Vector>, DEFAULT_SEPARATOR)),
      // p_theta(parse_file<Vector>(paths.p_theta, read_vector<Vector>, DEFAULT_SEPARATOR)),
      spot(parse_file<Vector>(paths.spot, read_vector<Vector>, DEFAULT_SEPARATOR)),
      experiment_scaling(parse_file<Vector>(paths.experiment, read_vector<Vector>, DEFAULT_SEPARATOR)),
      experiment_scaling_long(S) {
  update_experiment_scaling_long(c);

  LOG(debug) << *this;
}

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
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

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
double Model<feat_kind, mix_kind>::log_likelihood_poisson_counts(
    const IMatrix &counts) const {
  double l = 0;
#pragma omp parallel for reduction(+ : l) if (DO_PARALLEL)
  for (size_t g = 0; g < G; ++g)
    for (size_t s = 0; s < S; ++s) {
      double rate = lambda_gene_spot(g, s) * spot(s);
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

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
double Model<feat_kind, mix_kind>::log_likelihood(const IMatrix &counts) const {
  double l_features = features.log_likelihood(contributions_gene_type);
  double l_mix = weights.log_likelihood(contributions_spot_type);
  double l = l_features + l_mix;

  for (size_t s = 0; s < S; ++s)
    l += log_gamma(spot(s), parameters.hyperparameters.spot_a,
                   1.0 / parameters.hyperparameters.spot_b);
  if (parameters.activate_experiment_scaling) {
    for (size_t e = 0; e < E; ++e)
      l += log_gamma(experiment_scaling(e),
                     parameters.hyperparameters.experiment_a,
                     1.0 / parameters.hyperparameters.experiment_b);
  }

  double poisson_logl = log_likelihood_poisson_counts(counts);
  l += poisson_logl;

  return l;
}

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
Matrix Model<feat_kind, mix_kind>::weighted_theta() const {
  Matrix m = weights.matrix;
  for (size_t t = 0; t < T; ++t) {
    Float x = 0;
    for (size_t g = 0; g < G; ++g)
      x += phi(g, t);
    for (size_t s = 0; s < S; ++s) {
      m(s, t) *= x * spot(s);
      if (parameters.activate_experiment_scaling)
        m(s, t) *= experiment_scaling_long(s);
    }
  }
  return m;
}

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
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
  write_matrix(weighted_theta(), prefix + "weighted-mix.txt", spot_names, factor_names);
  write_vector(spot, prefix + "spot-scaling.txt", spot_names);
  write_vector(experiment_scaling, prefix + "experiment-scaling.txt", counts.experiment_names);
  // write_matrix(lambda_gene_spot, prefix + "lambda_gene_spot.txt", gene_names, spot_names);
  write_matrix(contributions_gene_type, prefix + "contributions_gene_type.txt", gene_names, factor_names);
  write_matrix(contributions_spot_type, prefix + "contributions_spot_type.txt", spot_names, factor_names);
  write_vector(contributions_gene, prefix + "contributions_gene.txt", gene_names);
  write_vector(contributions_spot, prefix + "contributions_spot.txt", spot_names);
  write_vector(contributions_experiment, prefix + "contributions_experiment.txt", counts.experiment_names);
  if (false and mean_and_variance) {
    write_matrix(posterior_expectations_poisson(), prefix + "means_poisson.txt", gene_names, spot_names);
    /* TODO reactivate
    write_matrix(posterior_expectations(), prefix + "means.txt", gene_names, spot_names);
    write_matrix(posterior_variances(), prefix + "variances.txt", gene_names, spot_names);
    */
  }
}

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
void Model<feat_kind, mix_kind>::sample_contributions_sub(
    const IMatrix &counts, size_t g, size_t s, RNG &rng,
    IMatrix &contrib_gene_type, IMatrix &contrib_spot_type) {
  std::vector<double> rel_rate(T);
  double z = 0;
  // NOTE: in principle, lambda[g][s][t] is proportional to both
  // spot[s] and experiment_scaling[s]. However, these terms would
  // cancel. Thus, we do not multiply them in here.
  for (size_t t = 0; t < T; ++t)
    z += rel_rate[t] = phi(g, t) * theta(s, t);
  for (size_t t = 0; t < T; ++t)
    rel_rate[t] /= z;
  lambda_gene_spot(g, s) = z;
  if (counts(g, s) > 0) {
    auto v = sample_multinomial<Int>(counts(g, s), begin(rel_rate), end(rel_rate), rng);
    for (size_t t = 0; t < T; ++t) {
      contrib_gene_type(g, t) += v[t];
      contrib_spot_type(s, t) += v[t];
    }
  }
}

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
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

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
/** sample count decomposition */
void Model<feat_kind, mix_kind>::sample_contributions_variational(
    const IMatrix &counts) {
  LOG(info) << "Sampling contribution marginals";
  contributions_gene_type = IMatrix(G, T, arma::fill::zeros);
  contributions_spot_type = IMatrix(S, T, arma::fill::zeros);
  lambda_gene_spot = Matrix(G, S, arma::fill::zeros);
  Matrix rate_gene_type(G, T, arma::fill::zeros);
  Matrix rate_spot_type(S, T, arma::fill::zeros);

#pragma omp parallel if (DO_PARALLEL)
  {
    Matrix rate_spot_type_(S, T, arma::fill::zeros);
    Vector lambda(T);
#pragma omp for
    for (size_t g = 0; g < G; ++g)
      for (size_t s = 0; s < S; ++s) {
        double factor = spot(s);
        if (parameters.activate_experiment_scaling)
          factor *= experiment_scaling_long[s];
        double z = 0;
        for (size_t t = 0; t < T; ++t)
          z += lambda[t] = phi(g, t) * theta(s, t) * factor;
        lambda_gene_spot(g, s) = z;
        if (counts(g, s) > 0)
          for (size_t t = 0; t < T; ++t) {
            const double x = lambda[t] / z * counts(g, s);
            rate_gene_type(g, t) += x;
            rate_spot_type_(s, t) += x;
          }
      }
#pragma omp critical
    { rate_spot_type += rate_spot_type_; }
  }

#pragma omp parallel if (DO_PARALLEL)
  for (size_t g = 0; g < G; ++g) {
    const size_t thread_num = omp_get_thread_num();
    std::vector<Float> a(T);
    double z = 0;
    for (size_t t = 0; t < T; ++t)
      z += a[t] = rate_gene_type(g, t);
    for (size_t t = 0; t < T; ++t)
      a[t] /= z;

    // LOG(info) << "contributions_gene(" << g << ") = " << contributions_gene[g];
    auto v = sample_multinomial<Int>(contributions_gene(g), begin(a), end(a),
                                     EntropySource::rngs[thread_num]);
    for (size_t t = 0; t < T; ++t)
      contributions_gene_type(g, t) = v[t];
  }

#pragma omp parallel if (DO_PARALLEL)
  for (size_t s = 0; s < S; ++s) {
    const size_t thread_num = omp_get_thread_num();
    std::vector<Float> a(T);
    double z = 0;
    for (size_t t = 0; t < T; ++t)
      z += a[t] = rate_spot_type(s, t);
    for (size_t t = 0; t < T; ++t)
      a[t] /= z;

    auto v = sample_multinomial<Int>(contributions_spot(s), begin(a), end(a),
                                     EntropySource::rngs[thread_num]);
    for (size_t t = 0; t < T; ++t)
      contributions_spot_type(s, t) = v[t];
  }
}

/** sample spot scaling factors */
template <Partial::Kind feat_kind, Partial::Kind mix_kind>
void Model<feat_kind, mix_kind>::sample_spot() {
  LOG(info) << "Sampling spot scaling factors";
  auto phi_marginal = marginalize_genes(features);
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t s = 0; s < S; ++s) {
    const Int summed_contribution = contributions_spot(s);

    Float intensity_sum = 0;
    for (size_t t = 0; t < T; ++t)
      intensity_sum += phi_marginal[t] * theta(s, t);
    if (parameters.activate_experiment_scaling)
      intensity_sum *= experiment_scaling_long[s];

    // NOTE: std::gamma_distribution takes a shape and scale parameter
    spot[s] = std::gamma_distribution<Float>(
        parameters.hyperparameters.spot_a + summed_contribution,
        1.0 / (parameters.hyperparameters.spot_b + intensity_sum))(
        EntropySource::rng);
  }

  if ((parameters.enforce_mean & ForceMean::Spot) != ForceMean::None) {
    double z = 0;
#pragma omp parallel for reduction(+ : z) if (DO_PARALLEL)
    for (size_t s = 0; s < S; ++s)
      z += spot[s];
    z /= S;
#pragma omp parallel for if (DO_PARALLEL)
    for (size_t s = 0; s < S; ++s)
      spot[s] /= z;
  }
}

/** sample experiment scaling factors */
template <Partial::Kind feat_kind, Partial::Kind mix_kind>
void Model<feat_kind, mix_kind>::sample_experiment_scaling(const Counts &data) {
  LOG(info) << "Sampling experiment scaling factors";

  auto phi_marginal = marginalize_genes(features);
  std::vector<Float> intensity_sums(E, 0);
  // TODO: improve parallelism
  for (size_t s = 0; s < S; ++s) {
    double x = 0;
#pragma omp parallel for reduction(+ : x) if (DO_PARALLEL)
    for (size_t t = 0; t < T; ++t)
      x += phi_marginal[t] * theta(s, t);
    x *= spot[s];
    intensity_sums[data.experiments[s]] += x;
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
template <Partial::Kind feat_kind, Partial::Kind mix_kind>
void Model<feat_kind, mix_kind>::update_experiment_scaling_long(
    const Counts &data) {
  for (size_t s = 0; s < S; ++s)
    experiment_scaling_long[s] = experiment_scaling[data.experiments[s]];
}

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
void Model<feat_kind, mix_kind>::gibbs_sample(const Counts &data, Target which,
                                              bool timing) {
  Timer timer;
  if (flagged(which & Target::contributions)) {
    if(parameters.variational)
      sample_contributions_variational(data.counts);
    else
      sample_contributions(data.counts);
    if (timing)
      LOG(info) << "This took " << timer.tock() << "μs.";
    LOG(debug) << "Log-likelihood = " << log_likelihood(data.counts);
  }

  if (flagged(which & Target::merge_split)) {
    // NOTE: this has to be done right after the Gibbs step for the
    // contributions because otherwise lambda_gene_spot is inconsistent
    timer.tick();
    sample_split_merge(data, which);
    if (timing)
      LOG(info) << "This took " << timer.tock() << "μs.";
    LOG(debug) << "Log-likelihood = " << log_likelihood(data.counts);
  }

  if (flagged(which & Target::spot)) {
    timer.tick();
    sample_spot();
    if (timing)
      LOG(info) << "This took " << timer.tock() << "μs.";
    LOG(debug) << "Log-likelihood = " << log_likelihood(data.counts);
  }

  if (flagged(which & Target::experiment)) {
    if (E > 1 and parameters.activate_experiment_scaling) {
      timer.tick();
      sample_experiment_scaling(data);
      if (timing)
        LOG(info) << "This took " << timer.tock() << "μs.";
      LOG(debug) << "Log-likelihood = " << log_likelihood(data.counts);
    }
  }

  if (flagged(which & (Target::phi_r | Target::phi_p))) {
    timer.tick();
    features.prior.sample(weights.matrix, contributions_gene_type, spot,
                          experiment_scaling_long);
    if (timing)
      LOG(info) << "This took " << timer.tock() << "μs.";
    LOG(debug) << "Log-likelihood = " << log_likelihood(data.counts);
  }

  if (flagged(which & (Target::theta_r | Target::theta_p))) {
    timer.tick();
    weights.prior.sample(features.matrix, contributions_spot_type, spot,
                         experiment_scaling_long);
    if (timing)
      LOG(info) << "This took " << timer.tock() << "μs.";
    LOG(debug) << "Log-likelihood = " << log_likelihood(data.counts);
  }

  if (flagged(which & Target::phi)) {
    timer.tick();
    features.sample(weights, contributions_gene_type, spot,
                    experiment_scaling_long);
    if (timing)
      LOG(info) << "This took " << timer.tock() << "μs.";
    LOG(debug) << "Log-likelihood = " << log_likelihood(data.counts);
  }

  if (flagged(which & Target::theta)) {
    timer.tick();
    weights.sample(features, contributions_spot_type, spot,
                   experiment_scaling_long);
    if (timing)
      LOG(info) << "This took " << timer.tock() << "μs.";
    LOG(debug) << "Log-likelihood = " << log_likelihood(data.counts);
  }
}

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
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

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
size_t Model<feat_kind, mix_kind>::find_weakest_factor() const {
  std::vector<Float> x(T, 0);
  auto phi_marginal = marginalize_genes(features);
  for (size_t t = 0; t < T; ++t)
    for (size_t s = 0; s < S; ++s) {
      Float z = phi_marginal[t] * theta(s, t) * spot[s];
      if (parameters.activate_experiment_scaling)
        z *= experiment_scaling_long[s];
      x[t] += z;
    }
  return std::distance(begin(x), min_element(begin(x), end(x)));
}

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
Model<feat_kind, mix_kind> Model<feat_kind, mix_kind>::run_submodel(
    size_t t, size_t n, const Counts &counts, Target which,
    const std::string &prefix, const std::vector<size_t> &init_factors) {
  const bool show_timing = false;
  // TODO: use init_factors
  Model<feat_kind, mix_kind> sub_model(counts, t, parameters);
  sub_model.spot = spot;
  sub_model.experiment_scaling_long = experiment_scaling_long;
  sub_model.experiment_scaling = experiment_scaling;

  if (print_sub_model_cnt)
    sub_model.store(counts,
                    prefix + "submodel_init_" + std::to_string(sub_model_cnt));

  // keep spot and experiment scaling fixed
  // don't recurse into either merge or sample steps
  which = which & ~(Target::spot | Target::experiment | Target::merge_split);

  // deactivate logging during Gibbs sampling for sub model
  bool prev_logging = boost::log::core::get()->set_logging_enabled(false);

  for (size_t i = 0; i < n; ++i) {
    sub_model.gibbs_sample(counts, which, show_timing);
    LOG(info) << "sub model log likelihood = "
              << sub_model.log_likelihood_poisson_counts(counts.counts);
  }

  // re-activate logging after Gibbs sampling for sub model
  boost::log::core::get()->set_logging_enabled(prev_logging);

  if (print_sub_model_cnt)
    sub_model.store(counts,
                    prefix + "submodel_opti_" + std::to_string(sub_model_cnt));
  // sub_model_cnt++; // TODO

  return sub_model;
}

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
void Model<feat_kind, mix_kind>::lift_sub_model(const Model &sub_model,
                                                size_t t1, size_t t2) {
  features.lift_sub_model(sub_model.features, t1, t2);
  weights.lift_sub_model(sub_model.weights, t1, t2);

  for (size_t g = 0; g < G; ++g)
    contributions_gene_type(g, t1) = sub_model.contributions_gene_type(g, t2);

  for (size_t s = 0; s < S; ++s)
    contributions_spot_type(s, t1) = sub_model.contributions_spot_type(s, t2);
}

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
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
  if (gibbs_test(ll_updated, ll_previous)) {
    LOG(info) << "Split step accecpted";
  } else {
    *this = previous;
    LOG(info) << "Split step rejected";
  }
}

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
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

  features.initialize_factor(t2);
  weights.initialize_factor(t2);

// add effect of updated parameters
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t g = 0; g < G; ++g)
    for (size_t s = 0; s < S; ++s)
      lambda_gene_spot(g, s)
          += phi(g, t1) * theta(s, t1) + phi(g, t2) * theta(s, t2);

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
  if (gibbs_test(ll_updated, ll_previous)) {
    LOG(info) << "Merge step accepted";
  } else {
    *this = previous;
    LOG(info) << "Merge step rejected";
  }
}

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
double Model<feat_kind, mix_kind>::posterior_expectation_poisson(
    size_t g, size_t s) const {
  double x = 0;
  for (size_t t = 0; t < T; ++t)
    x += phi(g, t) * theta(s, t);
  x *= spot[s];
  if (parameters.activate_experiment_scaling)
    x *= experiment_scaling_long[s];
  return x;
}

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
Matrix Model<feat_kind, mix_kind>::posterior_expectations_poisson() const {
  Matrix m(G, S);
  for (size_t g = 0; g < G; ++g)
    for (size_t s = 0; s < S; ++s)
      m(g, s) = posterior_expectation_poisson(g, s);
  return m;
}

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
void Model<feat_kind, mix_kind>::check_model(const IMatrix &counts) const {
  // check that phi is positive
  for (size_t g = 0; g < G; ++g)
    for (size_t t = 0; t < T; ++t) {
      if (phi[g][t] == 0)
        throw(std::runtime_error("Phi is zero for gene " + std::to_string(g)
                                 + " in factor " + std::to_string(t) + "."));
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
}

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
std::ostream &operator<<(
    std::ostream &os,
    const Model<feat_kind, mix_kind> &pfa) {
  os << "Poisson Factorization "
     << "G = " << pfa.G << " "
     << "S = " << pfa.S << " "
     << "T = " << pfa.T << std::endl;

  if (verbosity >= Verbosity::verbose) {
    print_matrix_head(os, pfa.features.matrix, "Φ");
    print_matrix_head(os, pfa.weights.matrix, "Θ");
    os << pfa.features.prior;
    os << pfa.weights.prior;

    print_vector_head(os, pfa.spot, "Spot scaling factors");
    if (pfa.parameters.activate_experiment_scaling)
      print_vector_head(os, pfa.experiment_scaling,
                        "Experiment scaling factors");
  }

  return os;
}

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
Model<feat_kind, mix_kind> operator*(const Model<feat_kind, mix_kind> &a,
                                     const Model<feat_kind, mix_kind> &b) {
  Model<feat_kind, mix_kind> model = a;

  model.contributions_gene_type %= b.contributions_gene_type;
  model.contributions_spot_type %= b.contributions_spot_type;
  model.contributions_gene %= b.contributions_gene;
  model.contributions_spot %= b.contributions_spot;
  model.contributions_experiment %= b.contributions_experiment;

  model.spot %= b.spot;
  model.experiment_scaling %= b.experiment_scaling;
  model.experiment_scaling_long %= b.experiment_scaling_long;

  model.features.matrix %= b.features.matrix;
  model.weights.matrix %= b.weights.matrix;

  return model;
}

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
Model<feat_kind, mix_kind> operator+(const Model<feat_kind, mix_kind> &a,
                                     const Model<feat_kind, mix_kind> &b) {
  Model<feat_kind, mix_kind> model = a;

  model.contributions_gene_type += b.contributions_gene_type;
  model.contributions_spot_type += b.contributions_spot_type;
  model.contributions_gene += b.contributions_gene;
  model.contributions_spot += b.contributions_spot;
  model.contributions_experiment += b.contributions_experiment;

  model.spot += b.spot;
  model.experiment_scaling += b.experiment_scaling;
  model.experiment_scaling_long += b.experiment_scaling_long;

  model.features.matrix += b.features.matrix;
  model.weights.matrix += b.weights.matrix;

  return model;
}

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
Model<feat_kind, mix_kind> operator-(const Model<feat_kind, mix_kind> &a,
                                     const Model<feat_kind, mix_kind> &b) {
  Model<feat_kind, mix_kind> model = a;

  model.contributions_gene_type -= b.contributions_gene_type;
  model.contributions_spot_type -= b.contributions_spot_type;
  model.contributions_gene -= b.contributions_gene;
  model.contributions_spot -= b.contributions_spot;
  model.contributions_experiment -= b.contributions_experiment;

  model.spot -= b.spot;
  model.experiment_scaling -= b.experiment_scaling;
  model.experiment_scaling_long -= b.experiment_scaling_long;

  model.features.matrix -= b.features.matrix;
  model.weights.matrix -= b.weights.matrix;

  return model;
}

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
Model<feat_kind, mix_kind> operator*(const Model<feat_kind, mix_kind> &a,
                                     double x) {
  Model<feat_kind, mix_kind> model = a;

  model.contributions_gene_type *= x;
  model.contributions_spot_type *= x;
  model.contributions_gene *= x;
  model.contributions_spot *= x;
  model.contributions_experiment *= x;

  model.spot *= x;
  model.experiment_scaling *= x;
  model.experiment_scaling_long *= x;

  model.features.matrix *= x;
  model.weights.matrix *= x;

  return model;
}

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
Model<feat_kind, mix_kind> operator/(const Model<feat_kind, mix_kind> &a,
                                     double x) {
  Model<feat_kind, mix_kind> model = a;

  model.contributions_gene_type /= x;
  model.contributions_spot_type /= x;
  model.contributions_gene /= x;
  model.contributions_spot /= x;
  model.contributions_experiment /= x;

  model.spot /= x;
  model.experiment_scaling /= x;
  model.experiment_scaling_long /= x;

  model.features.matrix /= x;
  model.weights.matrix /= x;

  return model;
}
}

#endif
