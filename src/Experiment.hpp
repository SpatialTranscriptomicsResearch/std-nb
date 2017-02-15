#ifndef EXPERIMENT_HPP
#define EXPERIMENT_HPP

#include <random>
#include "ModelType.hpp"
#include "Paths.hpp"
#include "compression.hpp"
#include "counts.hpp"
#include "entropy.hpp"
#include "metropolis_hastings.hpp"
#include "odds.hpp"
#include "parameters.hpp"
#include "stats.hpp"
#include "target.hpp"
#include "timer.hpp"
#include "verbosity.hpp"

namespace PoissonFactorization {

template <typename Type>
struct Experiment {
  using features_t = typename Type::features_t;
  using weights_t = typename Type::weights_t;

  /** number of genes */
  size_t G;
  /** number of samples */
  size_t S;
  /** number of factors */
  size_t T;

  Counts data;
  Matrix coords;

  Parameters parameters;

  /** hidden contributions to the count data due to the different factors */
  Matrix contributions_gene_type, contributions_spot_type;
  Vector contributions_gene, contributions_spot;

  /** factor loading matrix */
  features_t features;
  features_t baseline_feature;

  /** factor score matrix */
  weights_t weights;
  Matrix field;

  /** Normalizing factor to translate Poisson rates \lambda_{xgst} to relative
   * frequencies \lambda_{gst} / z_{gs} for the multionomial distribution */
  Matrix lambda_gene_spot;

  /** spot scaling vector */
  Vector spot;

  Experiment(const Counts &counts, size_t T,
             const Parameters &parameters);

  void store(const std::string &prefix, const features_t &global_features,
             const std::vector<size_t> &order) const;
  void restore(const std::string &prefix);
  void perform_pairwise_dge(const std::string &prefix,
                            const features_t &global_features) const;
  void perform_local_dge(const std::string &prefix,
                         const features_t &global_features) const;

  void gibbs_sample(const Matrix &global_phi);

  double log_likelihood() const;
  double log_likelihood_poisson_counts() const;

  Matrix posterior_expectations_poisson() const;
  Matrix posterior_expectations_negative_multinomial(
      const features_t &global_features) const;

  inline Float &phi(size_t g, size_t t) { return features.matrix(g, t); };
  inline Float phi(size_t g, size_t t) const { return features.matrix(g, t); };
  inline Float &baseline_phi(size_t g) {
    return baseline_feature.matrix(g, 0);
  };
  inline Float baseline_phi(size_t g) const {
    return baseline_feature.matrix(g, 0);
  };

  inline Float &theta(size_t s, size_t t) { return weights.matrix(s, t); };
  inline Float theta(size_t s, size_t t) const { return weights.matrix(s, t); };

  /** sample count decomposition */
  void sample_contributions(const Matrix &global_phi);
  /** sub-routine for count decomposition sampling */
  double sample_contributions_sub(const Matrix &global_phi, size_t g, size_t s,
                                  RNG &rng, Matrix &contrib_gene_type,
                                  Matrix &contrib_spot_type, bool dropout_gene,
                                  bool dropout_spot) const;

  /** sample spot scaling factors */
  void sample_spot(const Matrix &global_phi);

  /** sample baseline feature */
  void sample_baseline(const Matrix &global_phi);

  Vector marginalize_genes(const Matrix &global_phi) const;
  Vector marginalize_spots() const;

  // computes a matrix M(g,t)
  // with M(g,t) = baseline_phi(g) global_phi(g,t) sum_s theta(s,t) sigma(s)
  Matrix explained_gene_type(const Matrix &global_phi) const;
  // computes a matrix M(g,t)
  // with M(g,t) = baseline_phi(g) global_phi(g,t) phi(g,t) sum_s theta(s,t) sigma(s)
  Matrix expected_gene_type(const Matrix &global_phi) const;
  // computes a matrix M(s,t)
  // with M(s,t) = sigma(s) sum_g baseline_phi(g) phi(g,t) global_phi(g,t)
  Matrix explained_spot_type(const Matrix &global_phi) const;
  // computes a matrix M(s,t)
  // with M(s,t) = theta(s,t) sigma(s) sum_g baseline_phi(g) phi(g,t) global_phi(g,t)
  Matrix expected_spot_type(const Matrix &global_phi) const;
  // computes a vector V(g)
  // with V(g) = sum_t phi(g,t) global_phi(g,t) sum_s theta(s,t) sigma(s)
  Vector explained_gene(const Matrix &global_phi) const;

  std::vector<std::vector<size_t>> active_factors(const Matrix &global_phi,
                                                  double threshold = 1.0) const;

  Matrix pairwise_dge(const features_t &global_features) const;
  Vector pairwise_dge_sub(const features_t &global_features, size_t t1,
                          size_t t2) const;
  Float pairwise_dge_sub(const features_t &global_features, size_t t1,
                         size_t t2, size_t g, Float theta = 100,
                         Float p = 0.5) const;

  template <typename Fnc>
  Matrix local_dge(Fnc fnc, const features_t &global_features) const;
  template <typename Fnc>
  Float local_dge_sub(Fnc fnc, const features_t &global_features, size_t g,
                      size_t t, Float theta, Float p = 0.5) const;
};

#include "ExperimentDGE.hpp"

template <typename Type>
Experiment<Type>::Experiment(const Counts &data_, size_t T_,
                             const Parameters &parameters_)
    : G(data_.counts.n_rows),
      S(data_.counts.n_cols),
      T(T_),
      data(data_),
      coords(data.parse_coords()),
      parameters(parameters_),
      contributions_gene_type(G, T, arma::fill::zeros),
      contributions_spot_type(S, T, arma::fill::zeros),
      contributions_gene(rowSums<Vector>(data.counts)),
      contributions_spot(colSums<Vector>(data.counts)),
      features(G, T, parameters),
      baseline_feature(G, 1, parameters),
      weights(S, T, parameters),
      field(Matrix(S, T, arma::fill::ones)),
      lambda_gene_spot(G, S, arma::fill::zeros),
      spot(S, arma::fill::ones) {
  LOG(debug) << "Experiment G = " << G << " S = " << S << " T = " << T;
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
  LOG(debug) << "Coords: " << coords;

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

  if (not parameters.targeted(Target::phi_local))
    features.matrix.ones();

  if (not parameters.targeted(Target::phi_prior_local))
    features.prior.set_unit();

  if (not parameters.targeted(Target::baseline)) {
    baseline_feature.matrix.ones();
    baseline_feature.prior.set_unit();
  }

  if (not parameters.targeted(Target::theta))
    weights.matrix.ones();

  if (not parameters.targeted(Target::spot))
    spot.ones();
}

template <typename Type>
void Experiment<Type>::store(const std::string &prefix,
                             const features_t &global_features,
                             const std::vector<size_t> &order) const {
  auto factor_names = form_factor_names(T);
  auto &gene_names = data.row_names;
  auto &spot_names = data.col_names;

  std::string suffix = "";
  std::string extension
      = boost::filesystem::path(data.path).extension().c_str();
  if (extension == ".gz" or extension == ".bz2")
    suffix = extension;
  boost::filesystem::create_symlink(
      data.path, prefix + "counts" + FILENAME_ENDING + suffix);

#pragma omp parallel sections if (DO_PARALLEL)
  {
#pragma omp section
    features.store(prefix, gene_names, factor_names, order);
#pragma omp section
    baseline_feature.store(prefix + "baseline", gene_names, {1, "Baseline"},
                           {});
#pragma omp section
    weights.store(prefix, spot_names, factor_names, order);
#pragma omp section
    write_vector(spot, prefix + "spot-scaling" + FILENAME_ENDING,
                 parameters.compression_mode, spot_names);
#pragma omp section
    write_matrix(field, prefix + "raw-field" + FILENAME_ENDING,
                 parameters.compression_mode, spot_names, factor_names, order);
#pragma omp section
    write_matrix(expected_spot_type(global_features.matrix),
                 prefix + "expected-mix" + FILENAME_ENDING,
                 parameters.compression_mode, spot_names, factor_names, order);
#pragma omp section
    write_matrix(expected_gene_type(global_features.matrix),
                 prefix + "expected-features" + FILENAME_ENDING,
                 parameters.compression_mode, gene_names, factor_names, order);
#pragma omp section
    {
      auto phi_marginal = marginalize_genes(global_features.matrix);
      auto f = field;
      f.each_row() %= phi_marginal.t();
      f.each_col() %= spot;
      write_matrix(f, prefix + "expected-field" + FILENAME_ENDING,
                   parameters.compression_mode, spot_names, factor_names,
                   order);
    }
#pragma omp section
    if (parameters.store_lambda)
      write_matrix(lambda_gene_spot,
                   prefix + "lambda_gene_spot" + FILENAME_ENDING,
                   parameters.compression_mode, gene_names, spot_names);
#pragma omp section
    write_matrix(contributions_gene_type,
                 prefix + "contributions_gene_type" + FILENAME_ENDING,
                 parameters.compression_mode, gene_names, factor_names, order);
#pragma omp section
    write_matrix(contributions_spot_type,
                 prefix + "contributions_spot_type" + FILENAME_ENDING,
                 parameters.compression_mode, spot_names, factor_names, order);
#pragma omp section
    write_vector(contributions_gene,
                 prefix + "contributions_gene" + FILENAME_ENDING,
                 parameters.compression_mode, gene_names);
#pragma omp section
    write_vector(contributions_spot,
                 prefix + "contributions_spot" + FILENAME_ENDING,
                 parameters.compression_mode, spot_names);
  }
  if (false) {
    write_matrix(posterior_expectations_poisson(),
                 prefix + "counts_expected_poisson" + FILENAME_ENDING,
                 parameters.compression_mode, gene_names, spot_names);
    write_matrix(posterior_expectations_negative_multinomial(global_features),
                 prefix + "counts_expected" + FILENAME_ENDING,
                 parameters.compression_mode, gene_names, spot_names);
  }
}

template <typename Type>
void Experiment<Type>::restore(const std::string &prefix) {
  contributions_gene_type = parse_file<Matrix>(prefix + "contributions_gene_type" + FILENAME_ENDING, read_matrix, "\t");
  contributions_spot_type = parse_file<Matrix>(prefix + "contributions_spot_type" + FILENAME_ENDING, read_matrix, "\t");
  contributions_gene = parse_file<Vector>(prefix + "contributions_gene" + FILENAME_ENDING, read_vector<Vector>, "\t");
  contributions_spot = parse_file<Vector>(prefix + "contributions_spot" + FILENAME_ENDING, read_vector<Vector>, "\t");

  features.restore(prefix);
  baseline_feature.restore(prefix + "baseline");
  weights.restore(prefix);

  field = parse_file<Matrix>(prefix + "raw-field" + FILENAME_ENDING, read_matrix, "\t");
  spot = parse_file<Vector>(prefix + "spot-scaling" + FILENAME_ENDING, read_vector<Vector>, "\t");
}

template <typename Type>
void Experiment<Type>::perform_pairwise_dge(const std::string &prefix,
                             const features_t &global_features) const {
  auto &gene_names = data.row_names;
  auto x = pairwise_dge(global_features);
  std::vector<std::string> factor_pair_names;
  for (size_t t1 = 0; t1 < T; ++t1)
    for (size_t t2 = t1 + 1; t2 < T; ++t2)
      factor_pair_names.push_back("Factor" + std::to_string(t1 + 1)
          + "-Factor" + std::to_string(t2 + 1));
  write_matrix(x, prefix + "pairwise_differential_gene_expression" + FILENAME_ENDING,
      parameters.compression_mode, gene_names, factor_pair_names);
}

template <typename Type>
void Experiment<Type>::perform_local_dge(const std::string &prefix,
                             const features_t &global_features) const {
  if (std::is_same<typename Type::features_t::prior_type,
                   PRIOR::PHI::Dirichlet>::value)
    return;

  auto &gene_names = data.row_names;
  auto factor_names = form_factor_names(T);
  write_matrix(
      local_dge([](Float baseline __attribute__((unused)), Float local __attribute__((unused))) { return 1; }, global_features),
      prefix + "differential_gene_expression_baseline_and_local" + FILENAME_ENDING,
      parameters.compression_mode, gene_names, factor_names);

  write_matrix(
      local_dge([](Float baseline __attribute__((unused)), Float local) { return local; }, global_features),
      prefix + "differential_gene_expression_baseline" + FILENAME_ENDING,
      parameters.compression_mode, gene_names, factor_names);

  write_matrix(
      local_dge([](Float baseline, Float local __attribute__((unused))) { return baseline; }, global_features),
      prefix + "differential_gene_expression_local" + FILENAME_ENDING,
      parameters.compression_mode, gene_names, factor_names);
}

template <typename Type>
void Experiment<Type>::gibbs_sample(const Matrix &global_phi) {
  // TODO reactivate
  if (false)
    if (parameters.targeted(Target::contributions))
      sample_contributions(global_phi);

  if (parameters.targeted(Target::theta))
    weights.sample_field(*this, field, global_phi);

  // TODO add CLI switch
  auto order = random_order(5);
  for (auto &o : order)
    switch (o) {
      case 0:
        // TODO add baseline prior
        if (parameters.targeted(Target::baseline))
          sample_baseline(global_phi);
        break;

      case 1:
        if (parameters.targeted(Target::theta_prior)
            and parameters.theta_local_priors) {
          throw("Not implemented");
          // TODO re-implement
          /*
          Matrix feature_matrix = features.matrix % global_phi;
          // feature_matrix.each_col() *= baseline_feature.matrix.col(0);
          for (size_t g = 0; g < G; ++g)
            for (size_t t = 0; t < T; ++t)
              feature_matrix(g, t) *= baseline_phi(g);
          weights.prior.sample(feature_matrix, contributions_spot_type, spot);
          */
        }
        break;

      case 2:
        if (parameters.targeted(Target::phi_prior_local))
          // TODO FIXME make this work!
          features.prior.sample(*this, global_phi);
        break;

      case 3:
        if (parameters.targeted(Target::phi_local))
          features.sample(*this, global_phi);
        break;

      case 4:
        if (parameters.targeted(Target::spot))
          sample_spot(global_phi);
        break;

      default:
        break;
    }
}

template <typename Type>
double Experiment<Type>::log_likelihood() const {
  double l_features = features.log_likelihood();
  double l_baseline_feature = baseline_feature.log_likelihood();
  double l_mix = weights.log_likelihood();

  double l_spot = 0;
  for (size_t s = 0; s < S; ++s)
    // NOTE: log_gamma takes a shape and scale parameter
    l_spot += log_gamma(spot(s), parameters.hyperparameters.spot_a,
                        1.0 / parameters.hyperparameters.spot_b);

  double l_poisson = log_likelihood_poisson_counts();

  double l = l_features + l_baseline_feature + l_mix + l_spot + l_poisson;

  LOG(verbose) << "Local feature log likelihood: " << l_features;
  LOG(verbose) << "Local baseline feature log likelihood: " << l_baseline_feature;
  LOG(verbose) << "Factor activity log likelihood: " << l_mix;
  LOG(verbose) << "Spot scaling log likelihood: " << l_spot;
  LOG(verbose) << "Counts log likelihood: " << l_poisson;
  LOG(verbose) << "Experiment log likelihood: " << l;

  return l;
}

template <typename Type>
double Experiment<Type>::log_likelihood_poisson_counts() const {
  double l = 0;
#pragma omp parallel for reduction(+ : l) if (DO_PARALLEL)
  for (size_t g = 0; g < G; ++g)
    for (size_t s = 0; s < S; ++s) {
      double rate = lambda_gene_spot(g, s) * spot(s) * baseline_phi(g);
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

template <typename Type>
Matrix Experiment<Type>::posterior_expectations_poisson() const {
  Matrix m(G, S);
  for (size_t g = 0; g < G; ++g)
    for (size_t s = 0; s < S; ++s)
      m(g, s) = lambda_gene_spot(g, s) * spot(s) * baseline_phi(g);
  return m;
}

template <typename Type>
Matrix Experiment<Type>::posterior_expectations_negative_multinomial(
    const features_t &global_features) const {
  Matrix prior_ratio = global_features.prior.ratio();
  Matrix m(G, S, arma::fill::zeros);
#pragma omp parallel for
  for (size_t g = 0; g < G; ++g)
    for (size_t s = 0; s < S; ++s)
      for (size_t t = 0; t < T; ++t) {
        m(g, s) += prior_ratio(g, t) * phi(g, t) * theta(s, t) * spot(s);
      }
  return m;
}

template <typename Type>
/** sample count decomposition */
void Experiment<Type>::sample_contributions(const Matrix &global_phi) {
  LOG(verbose) << "Sampling contributions";
  contributions_gene_type.fill(0);
  contributions_spot_type.fill(0);

  std::vector<bool> dropout_gene(G, false);
  size_t dropped_genes = 0;
  if (parameters.dropout_gene > 0.0)
    for (size_t g = 0; g < G; ++g)
      if (RandomDistribution::Uniform(EntropySource::rng) < parameters.dropout_gene)
        dropped_genes += dropout_gene[g] = true;

  std::vector<bool> dropout_spot(S, false);
  size_t dropped_spots = 0;
  if (parameters.dropout_spot > 0.0)
    for (size_t s = 0; s < S; ++s)
      if (RandomDistribution::Uniform(EntropySource::rng) < parameters.dropout_spot)
        dropped_spots += dropout_spot[s] = true;

  if (parameters.dropout_gene > 0.0)
    LOG(verbose) << "Gene dropout rate = " << parameters.dropout_gene * 100
                 << "\% Dropping " << dropped_genes << " genes";
  if (parameters.dropout_spot > 0.0)
    LOG(verbose) << "Spot dropout rate = " << parameters.dropout_spot * 100
                 << "\% Dropping " << dropped_spots << " spots";

#pragma omp parallel if (DO_PARALLEL)
  {
    Matrix contrib_gene_type(G, T, arma::fill::zeros);
    Matrix contrib_spot_type(S, T, arma::fill::zeros);
    const size_t thread_num = omp_get_thread_num();
#pragma omp for
    for (size_t g = 0; g < G; ++g)
      for (size_t s = 0; s < S; ++s)
        lambda_gene_spot(g, s) = sample_contributions_sub(
            global_phi, g, s, EntropySource::rngs[thread_num],
            contrib_gene_type, contrib_spot_type, dropout_gene[g], dropout_spot[s]);
#pragma omp critical
    {
      contributions_gene_type += contrib_gene_type;
      contributions_spot_type += contrib_spot_type;
    }
  }

  parameters.dropout_gene *= parameters.dropout_anneal;
  parameters.dropout_spot *= parameters.dropout_anneal;
}

template <typename Type>
double Experiment<Type>::sample_contributions_sub(const Matrix &global_phi,
                                                  size_t g, size_t s, RNG &rng,
                                                  Matrix &contrib_gene_type,
                                                  Matrix &contrib_spot_type,
                                                  bool dropout_gene,
                                                  bool dropout_spot) const {
  // TODO remove conditionals; perhaps templatize?
  std::vector<double> rel_rate(T);
  double z = 0;
  // NOTE: in principle, lambda(g,s,t) is proportional to the baseline feature
  // and the spot scaling. However, these terms would cancel. Thus, we don't
  // respect them here.
  if (dropout_spot)
    for (size_t t = 0; t < T; ++t)
      z += rel_rate[t] = phi(g, t) * global_phi(g, t);
  else if (dropout_gene)
    for (size_t t = 0; t < T; ++t)
      z += rel_rate[t] = theta(s, t);
  else
    for (size_t t = 0; t < T; ++t)
      z += rel_rate[t] = phi(g, t) * global_phi(g, t) * theta(s, t);
  if (data.counts(g, s) > 0) {
    for (size_t t = 0; t < T; ++t)
      rel_rate[t] /= z;
    if (parameters.expected_contributions) {
      for (size_t t = 0; t < T; ++t) {
        double expected = data.counts(g, s) * rel_rate[t];
        contrib_gene_type(g, t) += expected;
        contrib_spot_type(s, t) += expected;
      }
    } else {
      auto v = sample_multinomial<Int>(data.counts(g, s), begin(rel_rate),
                                       end(rel_rate), rng);
      for (size_t t = 0; t < T; ++t) {
        contrib_gene_type(g, t) += v[t];
        contrib_spot_type(s, t) += v[t];
      }
    }
  }
  return z;
}

/** sample spot scaling factors */
template <typename Type>
void Experiment<Type>::sample_spot(const Matrix &global_phi) {
  LOG(verbose) << "Sampling spot scaling factors";
  auto phi_marginal = marginalize_genes(global_phi);
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
}

/** sample baseline feature */
template <typename Type>
void Experiment<Type>::sample_baseline(const Matrix &global_phi) {
  LOG(verbose) << "Sampling baseline feature from Gamma distribution";

  // TODO add CLI switch
  const double prior1 = parameters.hyperparameters.baseline1;
  const double prior2 = parameters.hyperparameters.baseline2;
  Vector observed = prior1 + contributions_gene;
  Vector explained = prior2 + explained_gene(global_phi);

  Partial::perform_sampling(observed, explained, baseline_feature.matrix,
                            parameters.over_relax);
}

template <typename Type>
Vector Experiment<Type>::marginalize_genes(const Matrix &global_phi) const {
  Vector intensities(T, arma::fill::zeros);
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t t = 0; t < T; ++t)
    for (size_t g = 0; g < G; ++g)
      intensities(t) += baseline_phi(g) * phi(g, t) * global_phi(g, t);
  return intensities;
};

template <typename Type>
Vector Experiment<Type>::marginalize_spots() const {
  Vector intensities(T, arma::fill::zeros);
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t t = 0; t < T; ++t)
    for (size_t s = 0; s < S; ++s)
      intensities[t] += theta(s, t) * spot[s];
  return intensities;
}

template <typename Type>
Matrix Experiment<Type>::explained_gene_type(const Matrix &global_phi) const {
  Vector theta_t = marginalize_spots();
  Matrix explained(G, T, arma::fill::zeros);
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t g = 0; g < G; ++g)
    for (size_t t = 0; t < T; ++t)
      explained(g, t) = baseline_phi(g) * global_phi(g, t) * theta_t(t);
  return explained;
}

template <typename Type>
Matrix Experiment<Type>::expected_gene_type(const Matrix &global_phi) const {
  return features.matrix % explained_gene_type(global_phi);
}

template <typename Type>
Vector Experiment<Type>::explained_gene(const Matrix &global_phi) const {
  Vector theta_t = marginalize_spots();
  Vector explained(G, arma::fill::zeros);
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t g = 0; g < G; ++g)
    for (size_t t = 0; t < T; ++t)
      explained(g) += phi(g, t) * global_phi(g, t) * theta_t(t);
  return explained;
};

template <typename Type>
Matrix Experiment<Type>::explained_spot_type(const Matrix &global_phi) const {
  Matrix m = Matrix(S, T, arma::fill::ones);
  for (size_t t = 0; t < T; ++t) {
    Float x = 0;
    for (size_t g = 0; g < G; ++g)
      x += baseline_phi(g) * phi(g, t) * global_phi(g, t);
    for (size_t s = 0; s < S; ++s)
      m(s, t) *= x * spot(s);
  }
  return m;
}

template <typename Type>
Matrix Experiment<Type>::expected_spot_type(const Matrix &global_phi) const {
  return weights.matrix % explained_spot_type(global_phi);
}

template <typename Type>
std::vector<std::vector<size_t>> Experiment<Type>::active_factors(
    const Matrix &global_phi, double threshold) const {
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

template <typename Type>
std::ostream &operator<<(std::ostream &os, const Experiment<Type> &experiment) {
  os << "Experiment "
     << "G = " << experiment.G << " "
     << "S = " << experiment.S << " "
     << "T = " << experiment.T << std::endl;

  if (verbosity >= Verbosity::debug) {
    print_matrix_head(os, experiment.baseline_feature.matrix, "Baseline Φ");
    print_matrix_head(os, experiment.features.matrix, "Φ");
    print_matrix_head(os, experiment.weights.matrix, "Θ");
    /* TODO reactivate
    os << experiment.baseline_feature.prior;
    os << experiment.features.prior;
    os << experiment.weights.prior;

    print_vector_head(os, experiment.spot, "Spot scaling factors");
    */
  }

  return os;
}

template <typename Type>
Experiment<Type> operator*(const Experiment<Type> &a,
                           const Experiment<Type> &b) {
  Experiment<Type> experiment = a;

  experiment.contributions_gene_type %= b.contributions_gene_type;
  experiment.contributions_spot_type %= b.contributions_spot_type;
  experiment.contributions_gene %= b.contributions_gene;
  experiment.contributions_spot %= b.contributions_spot;

  experiment.spot %= b.spot;

  experiment.features.matrix %= b.features.matrix;
  experiment.baseline_feature.matrix %= b.baseline_feature.matrix;
  experiment.weights.matrix %= b.weights.matrix;

  return experiment;
}

template <typename Type>
Experiment<Type> operator+(const Experiment<Type> &a,
                           const Experiment<Type> &b) {
  Experiment<Type> experiment = a;

  experiment.contributions_gene_type += b.contributions_gene_type;
  experiment.contributions_spot_type += b.contributions_spot_type;
  experiment.contributions_gene += b.contributions_gene;
  experiment.contributions_spot += b.contributions_spot;

  experiment.spot += b.spot;

  experiment.features.matrix += b.features.matrix;
  experiment.baseline_feature.matrix += b.baseline_feature.matrix;
  experiment.weights.matrix += b.weights.matrix;

  return experiment;
}

template <typename Type>
Experiment<Type> operator-(const Experiment<Type> &a,
                           const Experiment<Type> &b) {
  Experiment<Type> experiment = a;

  experiment.contributions_gene_type -= b.contributions_gene_type;
  experiment.contributions_spot_type -= b.contributions_spot_type;
  experiment.contributions_gene -= b.contributions_gene;
  experiment.contributions_spot -= b.contributions_spot;

  experiment.spot -= b.spot;

  experiment.features.matrix -= b.features.matrix;
  experiment.baseline_feature.matrix -= b.baseline_feature.matrix;
  experiment.weights.matrix -= b.weights.matrix;

  return experiment;
}

template <typename Type>
Experiment<Type> operator*(const Experiment<Type> &a, double x) {
  Experiment<Type> experiment = a;

  experiment.contributions_gene_type *= x;
  experiment.contributions_spot_type *= x;
  experiment.contributions_gene *= x;
  experiment.contributions_spot *= x;

  experiment.spot *= x;

  experiment.features.matrix *= x;
  experiment.baseline_feature.matrix *= x;
  experiment.weights.matrix *= x;

  return experiment;
}

template <typename Type>
Experiment<Type> operator/(const Experiment<Type> &a, double x) {
  Experiment<Type> experiment = a;

  experiment.contributions_gene_type /= x;
  experiment.contributions_spot_type /= x;
  experiment.contributions_gene /= x;
  experiment.contributions_spot /= x;

  experiment.spot /= x;

  experiment.features.matrix /= x;
  experiment.baseline_feature.matrix /= x;
  experiment.weights.matrix /= x;

  return experiment;
}

template <typename Type>
Experiment<Type> operator-(const Experiment<Type> &a, double x) {
  Experiment<Type> experiment = a;

  experiment.contributions_gene_type -= x;
  experiment.contributions_spot_type -= x;
  experiment.contributions_gene -= x;
  experiment.contributions_spot -= x;

  experiment.spot -= x;

  experiment.features.matrix -= x;
  experiment.baseline_feature.matrix -= x;
  experiment.weights.matrix -= x;

  return experiment;
}
}

#endif
