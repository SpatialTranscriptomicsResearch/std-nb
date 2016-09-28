#ifndef MODEL_HPP
#define MODEL_HPP

#include <random>
#include "PartialModel.hpp"
#include "compression.hpp"
#include "counts.hpp"
#include "entropy.hpp"
#include "io.hpp"
#include "Experiment.hpp"
#include "log.hpp"
#include "metropolis_hastings.hpp"
#include "odds.hpp"
#include "parallel.hpp"
#include "parameters.hpp"
#include "pdist.hpp"
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
const double local_phi_scaling_factor = 50;

size_t num_lines(const std::string &path);

template <Partial::Kind feat_kind = Partial::Kind::Gamma,
          Partial::Kind mix_kind = Partial::Kind::HierGamma>
struct Model {
  using features_t = Partial::Model<Partial::Variable::Feature, feat_kind>;
  using weights_t = Partial::Model<Partial::Variable::Mix, mix_kind>;
  using experiment_t = Experiment<feat_kind, mix_kind>;

  // computes a matrix M(g,t)
  // with M(g,t) = prior.p(g,t) + \sum_e var_phi(e)(g,t) sum_{s \in S_e}
  // theta(s,t) sigma(s)
  Matrix expected_gene_type() const;

  // TODO consider const
  /** number of genes */
  size_t G;
  /** number of factors */
  size_t T;
  /** number of experiments */
  size_t E;

  std::vector<experiment_t> experiments;

  Parameters parameters;

  /** hidden contributions to the count data due to the different factors */
  IMatrix contributions_gene_type;
  IVector contributions_gene;

  /** factor loading matrix */
  features_t features;
  // TODO consider weights_t global_weights;

  void update_contributions();

  inline Float &phi(size_t g, size_t t) { return features.matrix(g, t); };
  inline Float phi(size_t g, size_t t) const { return features.matrix(g, t); };

  Model(const std::vector<Counts> &data, const size_t T,
        const Parameters &parameters);
  // Model(const Counts &counts, const Paths &paths, const Parameters
  // &parameters);
  //
  void add_experiment(const Counts &data);

  void store(const std::string &prefix) const;

  /* TODO reactivate
  double log_likelihood(const IMatrix &counts) const;
  double log_likelihood_factor(const IMatrix &counts, size_t t) const;
  double log_likelihood_poisson_counts(const IMatrix &counts) const;
  */

  /** sample each of the variables from their conditional posterior */
  void gibbs_sample(Target which);

  /* TODO reactivate
  double posterior_expectation_poisson(size_t g, size_t s) const;
  Matrix posterior_expectations_poisson() const;
  */
};

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
Matrix Model<feat_kind, mix_kind>::expected_gene_type() const {
  // TODO use a matrix valued expression
  Matrix expected = features.prior.p;
  for (auto &experiment : experiments) {
    Vector theta_t = experiment.marginalize_spots();
    for (size_t t = 0; t < T; ++t)
      for (size_t g = 0; g < G; ++g)
        expected(g, t) += experiment.phi(g, t) * theta_t(t);
  }
  return expected;
}

/*
template <Partial::Kind feat_kind, Partial::Kind mix_kind>
Matrix Model<feat_kind, mix_kind>::marginalize_genes() const {
  Matrix intensities(features.E, features.T, arma::fill::zeros);
  for (size_t e = 0; e < E; ++e) {
    auto col = experiments[e].marginalize_genes(features);
    for (size_t g = 0; g < features.G; ++g)
      intensities(e, g) = col(g);
  }
  return intensities;
};
*/

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
std::ostream &operator<<(std::ostream &os,
                         const Model<feat_kind, mix_kind> &pfa);

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
void Model<feat_kind, mix_kind>::update_contributions() {
  contributions_gene_type.fill(0);
  contributions_gene.fill(0);
  for (auto &experiment : experiments)
#pragma omp parallel for if (DO_PARALLEL)
    for (size_t g = 0; g < G; ++g) {
      contributions_gene(g) += experiment.contributions_gene(g);
      for (size_t t = 0; t < T; ++t)
        contributions_gene_type(g, t)
            += experiment.contributions_gene_type(g, t);
    }
}

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
void Model<feat_kind, mix_kind>::add_experiment(const Counts &counts) {
  Parameters experiment_parameters = parameters;
  parameters.hyperparameters.phi_p_1 *= local_phi_scaling_factor;
  parameters.hyperparameters.phi_r_1 *= local_phi_scaling_factor;
  parameters.hyperparameters.phi_p_2 *= local_phi_scaling_factor;
  parameters.hyperparameters.phi_r_2 *= local_phi_scaling_factor;
  experiments.push_back({counts, T, experiment_parameters});
  experiments.rbegin()->features.matrix.fill(1);
  experiments.rbegin()->features.prior.r.fill(local_phi_scaling_factor);
  experiments.rbegin()->features.prior.p.fill(local_phi_scaling_factor);
}

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
Model<feat_kind, mix_kind>::Model(const std::vector<Counts> &c, const size_t T_,
                                  const Parameters &parameters_)
    : G(c.begin()->counts.n_rows),  // TODO FIXME c could be empty
      T(T_),
      E(c.size()),
      experiments(),
      parameters(parameters_),
      contributions_gene_type(G, T, arma::fill::zeros),
      contributions_gene(G, arma::fill::zeros),
      features(G, T, parameters) {
  LOG(info) << "G = " << G << " T = " << T << " E = " << E;
  for (auto &counts : c)
    add_experiment(counts);
  // TODO FIXME
  features.matrix.fill(1);
  features.prior.r.fill(1);
  features.prior.p.fill(1);
  update_contributions();

  /* TODO reactivate
  if (parameters.activate_experiment_scaling) {
    // initialize experiment scaling factors
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
  */
}

/* TODO reactivate
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
*/

/* TODO reactivate
template <Partial::Kind feat_kind, Partial::Kind mix_kind>
// TODO ensure no NaNs or infinities are generated
double Model<feat_kind, mix_kind>::Experiment::log_likelihood_factor(
    size_t t, Model<feat_kind, mix_kind>::features_t global_features) const {
  // TODO use global features
  assert(0);
  double l = features.log_likelihood_factor(counts, t)
             + weights.log_likelihood_factor(counts, t);

  if (std::isnan(l) or std::isinf(l))
    LOG(warning) << "Warning: log likelihoood contribution of factor " << t
                 << " = " << l;

  LOG(debug) << "ll_X = " << l;

  return l;
}

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
// TODO ensure no NaNs or infinities are generated
double Model<feat_kind, mix_kind>::log_likelihood_factor(size_t t) const {
  double l = 0;
  for (auto &experiment : experiments)
    l += experiment.log_likelihood_factor(t, features);
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

  double l_experiment_features = 0;
  for(size_t e = 0; e < E; ++e)
    l_experiment_features +=
features[e].log_likelihood(experiments_contributions_gene_type);

  double l = l_features + l_experiment_features + l_mix;

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
*/

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
void Model<feat_kind, mix_kind>::store(const std::string &prefix) const {
  std::vector<std::string> factor_names;
  for (size_t t = 1; t <= T; ++t)
    factor_names.push_back("Factor " + std::to_string(t));
  auto &gene_names = experiments.begin()->data.row_names;
  features.store(prefix, gene_names, factor_names);
  write_matrix(contributions_gene_type, prefix + "contributions_gene_type.txt", gene_names, factor_names);
  write_vector(contributions_gene, prefix + "contributions_gene.txt", gene_names);
  for (size_t e = 0; e < E; ++e) {
    std::string exp_prefix = prefix + "experiment" + std::to_string(e) + "-";
    experiments[e].store(exp_prefix);
  }
}

/** sample experiment scaling factors */
/* TODO reactivate
template <Partial::Kind feat_kind, Partial::Kind mix_kind>
void Model<feat_kind, mix_kind>::sample_experiment_scaling(const Counts &data) {
  LOG(info) << "Sampling experiment scaling factors";

  auto phi_marginal = marginalize_genes(features, experiment_features);
  std::vector<Float> intensity_sums(E, 0);
  // TODO: improve parallelism
  for (size_t s = 0; s < S; ++s) {
    double x = 0;
#pragma omp parallel for reduction(+ : x) if (DO_PARALLEL)
    for (size_t t = 0; t < T; ++t)
      x += phi_marginal(experiment_of(s), t) * theta(s, t);
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
*/

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
void Model<feat_kind, mix_kind>::gibbs_sample(Target which) {
  for (auto &experiment : experiments)
    if (flagged(which & Target::contributions))
      experiment.sample_contributions(features.matrix);
  update_contributions();

  // for (auto &experiment : experiments)
  //   experiment.gibbs_sample(features.matrix, which);
  // update_contributions();

  // TODO FIXME implement
  if (sample_local_phi_priors)
    if (flagged(which & (Target::phi_r | Target::phi_p)))
      features.prior.sample(*this);

  if (flagged(which & Target::phi))
    features.sample(*this);

  for (auto &experiment : experiments)
    experiment.gibbs_sample(features.matrix, which);
}

/* TODO reactivate
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
*/

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
std::ostream &operator<<(std::ostream &os,
                         const Model<feat_kind, mix_kind> &model) {
  os << "Poisson Factorization "
     << "G = " << model.G << " "
     << "T = " << model.T << " "
     << "E = " << model.E << std::endl;

  if (verbosity >= Verbosity::verbose) {
    print_matrix_head(os, model.features.matrix, "Φ");
    os << model.features.prior;
    /* TODO reactivate
    if (model.parameters.activate_experiment_scaling)
      print_vector_head(os, model.experiment_scaling,
                        "Experiment scaling factors");
    */
  }
  for (auto &experiment : model.experiments)
    os << experiment;

  return os;
}

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
Model<feat_kind, mix_kind> operator*(const Model<feat_kind, mix_kind> &a,
                                     const Model<feat_kind, mix_kind> &b) {
  Model<feat_kind, mix_kind> model = a;

  model.contributions_gene_type %= b.contributions_gene_type;
  model.contributions_gene %= b.contributions_gene;
  model.features.matrix %= b.features.matrix;
  for (size_t e = 0; e < model.E; ++e)
    model.experiments[e] = model.experiments[e] * b.experiments[e];

  return model;
}

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
Model<feat_kind, mix_kind> operator+(const Model<feat_kind, mix_kind> &a,
                                     const Model<feat_kind, mix_kind> &b) {
  Model<feat_kind, mix_kind> model = a;

  model.contributions_gene_type += b.contributions_gene_type;
  model.contributions_gene += b.contributions_gene;
  model.features.matrix += b.features.matrix;
  for (size_t e = 0; e < model.E; ++e)
    model.experiments[e] = model.experiments[e] + b.experiments[e];

  return model;
}

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
Model<feat_kind, mix_kind> operator-(const Model<feat_kind, mix_kind> &a,
                                     const Model<feat_kind, mix_kind> &b) {
  Model<feat_kind, mix_kind> model = a;

  model.contributions_gene_type -= b.contributions_gene_type;
  model.contributions_gene -= b.contributions_gene;
  model.features.matrix -= b.features.matrix;
  for (size_t e = 0; e < model.E; ++e)
    model.experiments[e] = model.experiments[e] - b.experiments[e];

  return model;
}

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
Model<feat_kind, mix_kind> operator*(const Model<feat_kind, mix_kind> &a,
                                     double x) {
  Model<feat_kind, mix_kind> model = a;

  model.contributions_gene_type *= x;
  model.contributions_gene *= x;
  model.features.matrix *= x;
  for (auto &experiment : model.experiments)
    experiment = experiment * x;

  return model;
}

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
Model<feat_kind, mix_kind> operator/(const Model<feat_kind, mix_kind> &a,
                                     double x) {
  Model<feat_kind, mix_kind> model = a;

  model.contributions_gene_type /= x;
  model.contributions_gene /= x;
  model.features.matrix /= x;
  for (auto &experiment : model.experiments)
    experiment = experiment / x;

  return model;
}
}

#endif
