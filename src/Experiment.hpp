#ifndef EXPERIMENT_HPP
#define EXPERIMENT_HPP

#include <random>
#include "PartialModel.hpp"
#include "compression.hpp"
#include "counts.hpp"
#include "entropy.hpp"
#include "io.hpp"
#include "log.hpp"
#include "metropolis_hastings.hpp"
#include "odds.hpp"
#include "parallel.hpp"
#include "parameters.hpp"
#include "Paths.hpp"
#include "pdist.hpp"
#include "priors.hpp"
#include "sampling.hpp"
#include "stats.hpp"
#include "target.hpp"
#include "timer.hpp"
#include "verbosity.hpp"

namespace PoissonFactorization {

const bool sample_global_phi_priors = true;
const bool sample_local_phi_priors = false;

bool gibbs_test(Float nextG, Float G, Float temperature = 50);
size_t num_lines(const std::string &path);

template <Partial::Kind feat_kind = Partial::Kind::Gamma,
          Partial::Kind mix_kind = Partial::Kind::HierGamma>
struct Experiment {
  using features_t = Partial::Model<Partial::Variable::Feature, feat_kind>;
  using weights_t = Partial::Model<Partial::Variable::Mix, mix_kind>;

  Experiment(const Counts &counts, const size_t T,
             const Parameters &parameters);
// TODO implement loading of Experiment

  Counts data;

  /** number of genes */
  size_t G;
  /** number of samples */
  size_t S;
  /** number of factors */
  size_t T;

  Parameters parameters;

  /** hidden contributions to the count data due to the different factors */
  IMatrix contributions_gene_type, contributions_spot_type;
  IVector contributions_gene, contributions_spot;
  Int contributions_experiment;

  /** factor loading matrix */
  features_t features;

  /** factor score matrix */
  weights_t weights;

  /** Normalizing factor to translate Poisson rates \lambda_{xgst} to relative
   * frequencies \lambda_{gst} / z_{gs} for the multionomial distribution */
  Matrix lambda_gene_spot;

  /** spot scaling vector */
  Vector spot;

  /** experiment scaling vector */
  Float experiment_scaling;

  inline Float &phi(size_t g, size_t t) { return features.matrix(g, t); };
  inline Float phi(size_t g, size_t t) const { return features.matrix(g, t); };

  inline Float &theta(size_t s, size_t t) { return weights.matrix(s, t); };
  inline Float theta(size_t s, size_t t) const { return weights.matrix(s, t); };

  Matrix weighted_theta() const;

  void gibbs_sample(const Matrix &global_phi, Target which);

  /** sample spot scaling factors */
  void sample_spot(const Matrix &var_phi);

  /** sample count decomposition */
  void sample_contributions(const Matrix &var_phi);
  void sample_contributions_sub(const Matrix &var_phi, size_t g, size_t s,
                                RNG &rng, IMatrix &contrib_gene_type,
                                IMatrix &contrib_spot_type);

  Vector marginalize_genes(const Matrix &var_phi) const;
  Vector marginalize_spots() const;
  void store(const std::string &prefix) const;

  // computes a matrix M(g,t)
  // with M(g,t) = prior.p(g,t) + var_phi(g,t) sum_s theta(s,t) sigma(s)
  Matrix expected_gene_type(const Matrix &var_phi) const;
};

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
Matrix Experiment<feat_kind, mix_kind>::expected_gene_type(
    const Matrix &var_phi) const {
  Vector theta_t = marginalize_spots();
  // TODO use a matrix valued expression
  Matrix expected(G, T, arma::fill::zeros);
  for (size_t g = 0; g < G; ++g)
    for (size_t t = 0; t < T; ++t)
      expected(g, t) = var_phi(g, t) * theta_t(t);
  return expected;
};

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
Vector Experiment<feat_kind, mix_kind>::marginalize_genes(
    const Matrix &var_phi) const {
  Vector intensities(T, arma::fill::zeros);
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t t = 0; t < T; ++t)
    for (size_t g = 0; g < G; ++g)
      intensities(t) += phi(g, t) * var_phi(g, t);
  return intensities;
};

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
Vector Experiment<feat_kind, mix_kind>::marginalize_spots() const {
  Vector intensities(T, arma::fill::zeros);
  // TODO improve parallelism
  for (size_t s = 0; s < S; ++s) {
    Float prod = spot[s];
    if (parameters.activate_experiment_scaling)
      prod *= experiment_scaling;
    for (size_t t = 0; t < T; ++t)
      intensities[t] += theta(s, t) * prod;
  }
  return intensities;
}

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
Experiment<feat_kind, mix_kind>::Experiment(
    const Counts &data_, const size_t T_, const Parameters &parameters_)
    : data(data_),
      G(data.counts.n_rows),
      S(data.counts.n_cols),
      T(T_),
      parameters(parameters_),
      contributions_gene_type(G, T, arma::fill::zeros),
      contributions_spot_type(S, T, arma::fill::zeros),
      contributions_gene(G, arma::fill::zeros),
      contributions_spot(S, arma::fill::zeros),
      contributions_experiment(0),
      features(G, T, parameters),
      weights(S, T, parameters),
      lambda_gene_spot(G, S, arma::fill::zeros),
      spot(S, arma::fill::ones),
      experiment_scaling(1) {
  LOG(info) << "G = " << G << " S = " << S << " T = " << T;
/* TODO consider reactivate
if (false) {
  // initialize:
  //  * contributions_gene_type
  //  * contributions_spot_type
  //  * lambda_gene_spot
  LOG(debug) << "Initializing contributions.";
  sample_contributions(c.counts);
}
*/

// initialize:
//  * contributions_spot
//  * contributions_experiment
//  TODO use initializer list together with a sums and a colSums function
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t s = 0; s < S; ++s)
    for (size_t g = 0; g < G; ++g) {
      contributions_spot(s) += data.counts(g, s);
      contributions_experiment += data.counts(g, s);
    }

//  TODO use initializer list together with a rowSums function
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t g = 0; g < G; ++g)
    for (size_t s = 0; s < S; ++s)
      contributions_gene(g) += data.counts(g, s);

  /*
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
  }
  */

  // initialize spot scaling factors
  {
    LOG(debug) << "Initializing spot scaling.";
    Float z = 0;
    for (size_t s = 0; s < S; ++s)
      z += spot(s) = contributions_spot(s) / experiment_scaling;
    z /= S;
    for (size_t s = 0; s < S; ++s)
      spot(s) /= z;
  }
}

// TODO implement loading of Experiment

/* TODO reactivate
template <Partial::Kind feat_kind, Partial::Kind mix_kind>
// TODO ensure no NaNs or infinities are generated
double Experiment<feat_kind, mix_kind>::log_likelihood_factor(
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
Matrix Experiment<feat_kind, mix_kind>::weighted_theta() const {
  Matrix m = weights.matrix;
  for (size_t t = 0; t < T; ++t) {
    Float x = 0;
    for (size_t g = 0; g < G; ++g)
      x += phi(g, t);
    for (size_t s = 0; s < S; ++s) {
      m(s, t) *= x * spot(s);
      if (parameters.activate_experiment_scaling)
        m(s, t) *= experiment_scaling;
    }
  }
  return m;
}

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
void Experiment<feat_kind, mix_kind>::store(
    const std::string &prefix) const {
  std::vector<std::string> factor_names;
  for (size_t t = 1; t <= T; ++t)
    factor_names.push_back("Factor " + std::to_string(t));
  auto &gene_names = data.row_names;
  auto &spot_names = data.col_names;
  features.store(prefix, gene_names, factor_names);
  weights.store(prefix, spot_names, factor_names);
  write_vector(spot, prefix + "spot-scaling.txt", spot_names);
  write_matrix(weighted_theta(), prefix + "weighted-mix.txt", spot_names, factor_names);
  /* TODO reactivate
  write_vector(experiment_scaling, prefix + "experiment-scaling.txt", counts.experiment_names);
  */
  if (parameters.store_lambda)
    write_matrix(lambda_gene_spot, prefix + "lambda_gene_spot.txt", gene_names, spot_names);
  write_matrix(contributions_gene_type, prefix + "contributions_gene_type.txt", gene_names, factor_names);
  write_matrix(contributions_spot_type, prefix + "contributions_spot_type.txt", spot_names, factor_names);
  write_vector(contributions_gene, prefix + "contributions_gene.txt", gene_names);
  write_vector(contributions_spot, prefix + "contributions_spot.txt", spot_names);
  /* TODO reactivate
  write_vector(contributions_experiment, prefix + "contributions_experiment.txt", counts.experiment_names);
  */
  /* TODO reactivate
  if (false and mean_and_variance) {
    write_matrix(posterior_expectations_poisson(), prefix + "means_poisson.txt", gene_names, spot_names);
    // TODO reactivate
    // write_matrix(posterior_expectations(), prefix + "means.txt", gene_names, spot_names);
    // write_matrix(posterior_variances(), prefix + "variances.txt", gene_names, spot_names);
  }
  */
}

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
void Experiment<feat_kind, mix_kind>::sample_contributions_sub(
    const Matrix &var_phi, size_t g, size_t s, RNG &rng,
    IMatrix &contrib_gene_type, IMatrix &contrib_spot_type) {
  std::vector<double> rel_rate(T);
  double z = 0;
  // NOTE: in principle, lambda[g][s][t] is proportional to both
  // spot[s] and experiment_scaling[s]. However, these terms would
  // cancel. Thus, we do not multiply them in here.
  for (size_t t = 0; t < T; ++t)
    z += rel_rate[t] = phi(g, t) * var_phi(g, t) * theta(s, t);
  for (size_t t = 0; t < T; ++t)
    rel_rate[t] /= z;
  lambda_gene_spot(g, s) = z;
  if (data.counts(g, s) > 0) {
    auto v = sample_multinomial<Int>(data.counts(g, s), begin(rel_rate),
                                     end(rel_rate), rng);
    /*
    LOG(debug) << "contribution_sub g = " << g << " s = " << s << " n = " <<
    data.counts(g,s);
    for(size_t t = 0; t < T; ++t)
      LOG(debug) << "theta[" << t << "] = " << theta(s,t);
    for(size_t t = 0; t < T; ++t)
      LOG(debug) << "phi[" << t << "] = " << phi(g,t);
    for(size_t t = 0; t < T; ++t)
      LOG(debug) << "var_phi.matrix[" << t << "] = " << var_phi(g,t);
    for(size_t t = 0; t < T; ++t)
      LOG(debug) << " p[" << t << "] = " << rel_rate[t];
    for(size_t t = 0; t < T; ++t)
      LOG(debug) << " v[" << t << "] = " << v[t];
    */
    for (size_t t = 0; t < T; ++t) {
      contrib_gene_type(g, t) += v[t];
      contrib_spot_type(s, t) += v[t];
    }
  }
}

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
/** sample count decomposition */
void Experiment<feat_kind, mix_kind>::sample_contributions(
    const Matrix &var_phi) {
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
        sample_contributions_sub(var_phi, g, s, EntropySource::rngs[thread_num],
                                 contrib_gene_type, contrib_spot_type);
#pragma omp critical
    {
      contributions_gene_type += contrib_gene_type;
      contributions_spot_type += contrib_spot_type;
    }
  }
}

/** sample spot scaling factors */
template <Partial::Kind feat_kind, Partial::Kind mix_kind>
void Experiment<feat_kind, mix_kind>::sample_spot(
    const Matrix &var_phi) {
  LOG(info) << "Sampling spot scaling factors";
  auto phi_marginal = marginalize_genes(var_phi);
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t s = 0; s < S; ++s) {
    Float intensity_sum = 0;
    for (size_t t = 0; t < T; ++t)
      intensity_sum += phi_marginal(t) * theta(s, t);
    if (parameters.activate_experiment_scaling)
      intensity_sum *= experiment_scaling;

    // NOTE: std::gamma_distribution takes a shape and scale parameter
    spot[s] = std::gamma_distribution<Float>(
        parameters.hyperparameters.spot_a + contributions_spot(s),
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
void Experiment<feat_kind, mix_kind>::gibbs_sample(const Matrix &global_phi,
                                                          Target which) {
  // TODO reactivate
  if (false)
    if (flagged(which & Target::contributions))
      sample_contributions(global_phi);

  /* TODO reactivate
  if (flagged(which & Target::experiment))
    if (E > 1 and parameters.activate_experiment_scaling)
      sample_experiment_scaling(data);
  */

  if (flagged(which & (Target::theta_r | Target::theta_p)))
    weights.prior.sample(features.matrix % global_phi, contributions_spot_type,
                         spot, experiment_scaling);

  if (flagged(which & Target::theta))
    weights.sample(*this, global_phi);

  if (sample_local_phi_priors)
    if (flagged(which & (Target::phi_r | Target::phi_p)))
      // TODO FIXME make this work!
      features.prior.sample(*this, global_phi);

  if (flagged(which & Target::phi))
    features.sample(*this, global_phi);

  if (flagged(which & Target::spot))
    sample_spot(global_phi);
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
Experiment<feat_kind, mix_kind> operator*(
    const Experiment<feat_kind, mix_kind> &a,
    const Experiment<feat_kind, mix_kind> &b) {
  Experiment<feat_kind, mix_kind> experiment = a;

  experiment.contributions_gene_type %= b.contributions_gene_type;
  experiment.contributions_spot_type %= b.contributions_spot_type;
  experiment.contributions_gene %= b.contributions_gene;
  experiment.contributions_spot %= b.contributions_spot;
  experiment.contributions_experiment %= b.contributions_experiment;

  experiment.spot %= b.spot;
  experiment.experiment_scaling *= b.experiment_scaling;

  experiment.features.matrix %= b.features.matrix;
  experiment.weights.matrix %= b.weights.matrix;

  return experiment;
}

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
Experiment<feat_kind, mix_kind> operator+(
    const Experiment<feat_kind, mix_kind> &a,
    const Experiment<feat_kind, mix_kind> &b) {
  Experiment<feat_kind, mix_kind> experiment = a;

  experiment.contributions_gene_type += b.contributions_gene_type;
  experiment.contributions_spot_type += b.contributions_spot_type;
  experiment.contributions_gene += b.contributions_gene;
  experiment.contributions_spot += b.contributions_spot;
  experiment.contributions_experiment += b.contributions_experiment;

  experiment.spot += b.spot;
  experiment.experiment_scaling += b.experiment_scaling;

  experiment.features.matrix += b.features.matrix;
  experiment.weights.matrix += b.weights.matrix;

  return experiment;
}

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
Experiment<feat_kind, mix_kind> operator-(
    const Experiment<feat_kind, mix_kind> &a,
    const Experiment<feat_kind, mix_kind> &b) {
  Experiment<feat_kind, mix_kind> experiment = a;

  experiment.contributions_gene_type -= b.contributions_gene_type;
  experiment.contributions_spot_type -= b.contributions_spot_type;
  experiment.contributions_gene -= b.contributions_gene;
  experiment.contributions_spot -= b.contributions_spot;
  experiment.contributions_experiment -= b.contributions_experiment;

  experiment.spot -= b.spot;
  experiment.experiment_scaling -= b.experiment_scaling;

  experiment.features.matrix -= b.features.matrix;
  experiment.weights.matrix -= b.weights.matrix;

  return experiment;
}

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
Experiment<feat_kind, mix_kind> operator*(
    const Experiment<feat_kind, mix_kind> &a, double x) {
  Experiment<feat_kind, mix_kind> experiment = a;

  experiment.contributions_gene_type *= x;
  experiment.contributions_spot_type *= x;
  experiment.contributions_gene *= x;
  experiment.contributions_spot *= x;
  experiment.contributions_experiment *= x;

  experiment.spot *= x;
  experiment.experiment_scaling *= x;

  experiment.features.matrix *= x;
  experiment.weights.matrix *= x;

  return experiment;
}

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
Experiment<feat_kind, mix_kind> operator/(
    const Experiment<feat_kind, mix_kind> &a, double x) {
  Experiment<feat_kind, mix_kind> experiment = a;

  experiment.contributions_gene_type /= x;
  experiment.contributions_spot_type /= x;
  experiment.contributions_gene /= x;
  experiment.contributions_spot /= x;
  experiment.contributions_experiment /= x;

  experiment.spot /= x;
  experiment.experiment_scaling /= x;

  experiment.features.matrix /= x;
  experiment.weights.matrix /= x;

  return experiment;
}

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
Experiment<feat_kind, mix_kind> operator-(
    const Experiment<feat_kind, mix_kind> &a, double x) {
  Experiment<feat_kind, mix_kind> experiment = a;

  experiment.contributions_gene_type -= x;
  experiment.contributions_spot_type -= x;
  experiment.contributions_gene -= x;
  experiment.contributions_spot -= x;
  experiment.contributions_experiment -= x;

  experiment.spot -= x;
  experiment.experiment_scaling -= x;

  experiment.features.matrix -= x;
  experiment.weights.matrix -= x;

  return experiment;
}

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
std::ostream &operator<<(std::ostream &os,
                         const Experiment<feat_kind, mix_kind> &experiment) {
  os << "Experiment "
     << "G = " << experiment.G << " "
     << "S = " << experiment.S << " "
     << "T = " << experiment.T << std::endl;

  if (verbosity >= Verbosity::verbose) {
    print_matrix_head(os, experiment.features.matrix, "Φ");
    print_matrix_head(os, experiment.weights.matrix, "Θ");
    /* TODO reactivate
    os << experiment.features.prior;
    os << experiment.weights.prior;

    print_vector_head(os, experiment.spot, "Spot scaling factors");
    if (experiment.parameters.activate_experiment_scaling)
      print_vector_head(os, experiment.experiment_scaling,
                        "Experiment scaling factors");
    */
  }

  return os;
}
}

#endif
