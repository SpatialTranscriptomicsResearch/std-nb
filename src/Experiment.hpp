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

template <Partial::Kind feat_kind = Partial::Kind::Gamma,
          Partial::Kind mix_kind = Partial::Kind::HierGamma>
struct Experiment {
  using features_t = Partial::Model<Partial::Variable::Feature, feat_kind>;
  using weights_t = Partial::Model<Partial::Variable::Mix, mix_kind>;

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

  /** factor loading matrix */
  features_t features;

  /** factor score matrix */
  weights_t weights;

  /** Normalizing factor to translate Poisson rates \lambda_{xgst} to relative
   * frequencies \lambda_{gst} / z_{gs} for the multionomial distribution */
  Matrix lambda_gene_spot;

  /** spot scaling vector */
  Vector spot;

  Experiment(const Counts &counts, const size_t T,
             const Parameters &parameters);
// TODO implement loading of Experiment

  void store(const std::string &prefix, const features_t &global_features) const;

  void gibbs_sample(const Matrix &global_phi, Target which);

  double log_likelihood() const;
  double log_likelihood_poisson_counts() const;

  Matrix posterior_expectations_poisson() const;
  Matrix posterior_expectations_negative_multinomial(const features_t &global_features) const;
  Matrix posterior_variances_negative_multinomial(const features_t &global_features) const;

  inline Float &phi(size_t g, size_t t) { return features.matrix(g, t); };
  inline Float phi(size_t g, size_t t) const { return features.matrix(g, t); };

  inline Float &theta(size_t s, size_t t) { return weights.matrix(s, t); };
  inline Float theta(size_t s, size_t t) const { return weights.matrix(s, t); };

  /** sample count decomposition */
  void sample_contributions(const Matrix &var_phi);
  /** sub-routine for count decomposition sampling */
  void sample_contributions_sub(const Matrix &var_phi, size_t g, size_t s,
                                RNG &rng, IMatrix &contrib_gene_type,
                                IMatrix &contrib_spot_type);

  /** sample spot scaling factors */
  void sample_spot(const Matrix &var_phi);

  Vector marginalize_genes(const Matrix &var_phi) const;
  Vector marginalize_spots() const;

  // computes a matrix M(g,t)
  // with M(g,t) = var_phi(g,t) sum_s theta(s,t) sigma(s)
  Matrix explained_gene_type(const Matrix &var_phi) const;
  // computes a matrix M(s,t)
  // with M(s,t) = theta(s,t) sigma(s) sum_g phi(g,t) var_phi(g,t)
  Matrix expected_spot_type(const Matrix &global_phi) const;

  std::vector<std::vector<size_t>> active_factors(const Matrix &global_phi,
                                                  double threshold = 1.0) const;
};

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
      features(G, T, parameters),
      weights(S, T, parameters),
      lambda_gene_spot(G, S, arma::fill::zeros),
      spot(S, arma::fill::ones) {
  LOG(verbose) << "G = " << G << " S = " << S << " T = " << T;
/* TODO consider to reactivate
if (false) {
  // initialize:
  //  * contributions_gene_type
  //  * contributions_spot_type
  //  * lambda_gene_spot
  LOG(debug) << "Initializing contributions.";
  sample_contributions(c.counts);
}
*/

// initialize contributions_spot
//  TODO use initializer list together with a sums and a colSums function
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t s = 0; s < S; ++s)
    for (size_t g = 0; g < G; ++g)
      contributions_spot(s) += data.counts(g, s);

// initialize contributions_gene
//  TODO use initializer list together with a rowSums function
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t g = 0; g < G; ++g)
    for (size_t s = 0; s < S; ++s)
      contributions_gene(g) += data.counts(g, s);

  // initialize spot scaling factors
  {
    LOG(debug) << "Initializing spot scaling.";
    Float z = 0;
    for (size_t s = 0; s < S; ++s)
      z += spot(s) = contributions_spot(s);
    z /= S;
    for (size_t s = 0; s < S; ++s)
      spot(s) /= z;
  }
}

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
void Experiment<feat_kind, mix_kind>::store(
    const std::string &prefix, const features_t &global_features) const {
  std::vector<std::string> factor_names;
  for (size_t t = 1; t <= T; ++t)
    factor_names.push_back("Factor " + std::to_string(t));
  auto &gene_names = data.row_names;
  auto &spot_names = data.col_names;
  features.store(prefix, gene_names, factor_names);
  weights.store(prefix, spot_names, factor_names);
  write_vector(spot, prefix + "spot-scaling.txt", spot_names);
  write_matrix(expected_spot_type(global_features.matrix), prefix + "weighted-mix.txt", spot_names, factor_names);
  Matrix profiles = features.matrix % explained_gene_type(global_features.matrix);
  write_matrix(profiles, prefix + "weighted-features.txt", gene_names, factor_names);
  if (parameters.store_lambda)
    write_matrix(lambda_gene_spot, prefix + "lambda_gene_spot.txt", gene_names, spot_names);
  write_matrix(contributions_gene_type, prefix + "contributions_gene_type.txt", gene_names, factor_names);
  write_matrix(contributions_spot_type, prefix + "contributions_spot_type.txt", spot_names, factor_names);
  write_vector(contributions_gene, prefix + "contributions_gene.txt", gene_names);
  write_vector(contributions_spot, prefix + "contributions_spot.txt", spot_names);
  if (false) {
    write_matrix(posterior_expectations_poisson(), prefix + "counts_expected_poisson.txt", gene_names, spot_names);
    write_matrix(posterior_expectations_negative_multinomial(global_features), prefix + "counts_expected.txt", gene_names, spot_names);
    write_matrix(posterior_variances_negative_multinomial(global_features), prefix + "counts_variance.txt", gene_names, spot_names);
  }
}

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
void Experiment<feat_kind, mix_kind>::gibbs_sample(const Matrix &global_phi,
                                                   Target which) {
  // TODO reactivate
  if (false)
    if (flagged(which & Target::contributions))
      sample_contributions(global_phi);

  if (flagged(which & (Target::theta_r | Target::theta_p)))
    weights.prior.sample(features.matrix % global_phi, contributions_spot_type,
                         spot);

  if (flagged(which & Target::theta))
    weights.sample(*this, global_phi);

  if (parameters.sample_local_phi_priors)
    if (flagged(which & (Target::phi_r | Target::phi_p)))
      // TODO FIXME make this work!
      features.prior.sample(*this, global_phi);

  if (flagged(which & Target::phi))
    features.sample(*this, global_phi);

  if (flagged(which & Target::spot))
    sample_spot(global_phi);
}

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
double Experiment<feat_kind, mix_kind>::log_likelihood() const {
  double l_features = features.log_likelihood(contributions_gene_type);
  double l_mix = weights.log_likelihood(contributions_spot_type);

  double l = l_features + l_mix;

  for (size_t s = 0; s < S; ++s)
    // NOTE: log_gamma takes a shape and scale parameter
    l += log_gamma(spot(s), parameters.hyperparameters.spot_a,
                   1.0 / parameters.hyperparameters.spot_b);

  double poisson_logl = log_likelihood_poisson_counts();
  l += poisson_logl;

  return l;
}

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
double Experiment<feat_kind, mix_kind>::log_likelihood_poisson_counts() const {
  double l = 0;
#pragma omp parallel for reduction(+ : l) if (DO_PARALLEL)
  for (size_t g = 0; g < G; ++g)
    for (size_t s = 0; s < S; ++s) {
      double rate = lambda_gene_spot(g, s) * spot(s);
      auto cur = log_poisson(data.counts(g, s), rate);
      if (std::isinf(cur) or std::isnan(cur))
        LOG(warning) << "ll poisson(g=" << g << ",s=" << s << ") = " << cur
                     << " counts = " << data.counts(g, s)
                     << " lambda = " << lambda_gene_spot(g, s)
                     << " rate = " << rate;
      l += cur;
    }
  return l;
}

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
Matrix Experiment<feat_kind, mix_kind>::posterior_expectations_poisson() const {
  Matrix m(G, S);
  for (size_t g = 0; g < G; ++g)
    for (size_t s = 0; s < S; ++s)
      m(g, s) = lambda_gene_spot(g, s) * spot(s);
  return m;
}

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
Matrix
Experiment<feat_kind, mix_kind>::posterior_expectations_negative_multinomial(const features_t &global_features)
    const {
  Matrix m(G, S, arma::fill::zeros);
#pragma omp parallel for
  for (size_t g = 0; g < G; ++g)
    for (size_t s = 0; s < S; ++s)
      for (size_t t = 0; t < T; ++t) {
        m(g, s) += global_features.prior.r(g, t) / global_features.prior.p(g, t)
                   * phi(g, t) * theta(s, t) * spot(s);
      }
  return m;
}

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
Matrix Experiment<feat_kind,
                  mix_kind>::posterior_variances_negative_multinomial(const features_t &global_features) const {
  Matrix m(G, S, arma::fill::zeros);
#pragma omp parallel for
  for (size_t g = 0; g < G; ++g)
    for (size_t s = 0; s < S; ++s)
      for (size_t t = 0; t < T; ++t) {
        double x = phi(g, t) * theta(s, t) * spot(s);
        m(g, s) += x * (x + global_features.prior.p(g, t))
                   * global_features.prior.r(g, t)
                   / global_features.prior.p(g, t)
                   / global_features.prior.p(g, t);
      }
  return m;
}

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
/** sample count decomposition */
void Experiment<feat_kind, mix_kind>::sample_contributions(
    const Matrix &global_phi) {
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
        sample_contributions_sub(global_phi, g, s, EntropySource::rngs[thread_num],
                                 contrib_gene_type, contrib_spot_type);
#pragma omp critical
    {
      contributions_gene_type += contrib_gene_type;
      contributions_spot_type += contrib_spot_type;
    }
  }
}

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
void Experiment<feat_kind, mix_kind>::sample_contributions_sub(
    const Matrix &global_phi, size_t g, size_t s, RNG &rng,
    IMatrix &contrib_gene_type, IMatrix &contrib_spot_type) {
  std::vector<double> rel_rate(T);
  double z = 0;
  // NOTE: in principle, lambda(g,s,t) is proportional to spot(s).
  // However, this term would cancel. Thus, we don't respect it here.
  for (size_t t = 0; t < T; ++t)
    z += rel_rate[t] = phi(g, t) * global_phi(g, t) * theta(s, t);
  for (size_t t = 0; t < T; ++t)
    rel_rate[t] /= z;
  lambda_gene_spot(g, s) = z;
  if (data.counts(g, s) > 0) {
    auto v = sample_multinomial<Int>(data.counts(g, s), begin(rel_rate),
                                     end(rel_rate), rng);
    for (size_t t = 0; t < T; ++t) {
      contrib_gene_type(g, t) += v[t];
      contrib_spot_type(s, t) += v[t];
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

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
Vector Experiment<feat_kind, mix_kind>::marginalize_genes(
    const Matrix &global_phi) const {
  Vector intensities(T, arma::fill::zeros);
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t t = 0; t < T; ++t)
    for (size_t g = 0; g < G; ++g)
      intensities(t) += phi(g, t) * global_phi(g, t);
  return intensities;
};

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
Vector Experiment<feat_kind, mix_kind>::marginalize_spots() const {
  Vector intensities(T, arma::fill::zeros);
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t t = 0; t < T; ++t)
    for (size_t s = 0; s < S; ++s)
      intensities[t] += theta(s, t) * spot[s];
  return intensities;
}

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
Matrix Experiment<feat_kind, mix_kind>::explained_gene_type(
    const Matrix &global_phi) const {
  Vector theta_t = marginalize_spots();
  Matrix explained(G, T, arma::fill::zeros);
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t g = 0; g < G; ++g)
    for (size_t t = 0; t < T; ++t)
      explained(g, t) = global_phi(g, t) * theta_t(t);
  return explained;
};

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
Matrix Experiment<feat_kind, mix_kind>::expected_spot_type(
    const Matrix &global_phi) const {
  Matrix m = weights.matrix;
  for (size_t t = 0; t < T; ++t) {
    Float x = 0;
    for (size_t g = 0; g < G; ++g)
      x += phi(g, t) * global_phi(g, t);
    for (size_t s = 0; s < S; ++s)
      m(s, t) *= x * spot(s);
  }
  return m;
}

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
std::vector<std::vector<size_t>>
Experiment<feat_kind, mix_kind>::active_factors(const Matrix &global_phi,
                                                double threshold) const {
  auto w = expected_spot_type(global_phi);
  std::vector<std::vector<size_t>> vs;
  for (size_t s = 0; s < S; ++s) {
    std::vector<size_t> v;
    for (size_t t = 0; t < T; ++t)
      if (w(s, t) > threshold)
        v.push_back(t);
    vs.push_back(v);
  }
  return vs;
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
    */
  }

  return os;
}

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
Experiment<feat_kind, mix_kind> operator*(
    const Experiment<feat_kind, mix_kind> &a,
    const Experiment<feat_kind, mix_kind> &b) {
  Experiment<feat_kind, mix_kind> experiment = a;

  experiment.contributions_gene_type %= b.contributions_gene_type;
  experiment.contributions_spot_type %= b.contributions_spot_type;
  experiment.contributions_gene %= b.contributions_gene;
  experiment.contributions_spot %= b.contributions_spot;

  experiment.spot %= b.spot;

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

  experiment.spot += b.spot;

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

  experiment.spot -= b.spot;

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

  experiment.spot *= x;

  experiment.features.matrix *= x;
  experiment.weights.matrix *= x;

  return experiment;
}

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
Experiment<feat_kind, mix_kind> operator/(
    const Experiment<feat_kind, mix_kind> &a, double x) {
  Experiment<feat_kind, mix_kind> experiment = a;

  experiment.contributions_gene_type /= x; // TODO note that this is inaccurate due to integer division
  experiment.contributions_spot_type /= x; // TODO note that this is inaccurate due to integer division
  experiment.contributions_gene /= x; // TODO note that this is inaccurate due to integer division
  experiment.contributions_spot /= x; // TODO note that this is inaccurate due to integer division

  experiment.spot /= x;

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

  experiment.spot -= x;

  experiment.features.matrix -= x;
  experiment.weights.matrix -= x;

  return experiment;
}
}

#endif
