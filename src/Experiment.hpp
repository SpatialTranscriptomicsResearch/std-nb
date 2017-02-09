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

  Matrix prev_grad_theta;
  Vector prev_grad_spot;
  IMatrix prev_sign_theta;
  IVector prev_sign_spot;

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

  void enforce_positive_parameters();

  void store(const std::string &prefix, const features_t &global_features,
             const std::vector<size_t> &order) const;
  void restore(const std::string &prefix);
  void perform_pairwise_dge(const std::string &prefix,
                            const features_t &global_features) const;
  void perform_local_dge(const std::string &prefix,
                         const features_t &global_features) const;

  void gibbs_sample(const features_t &global_features);

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
  void sample_contributions(const features_t &global_features,
                            Matrix &gradient_r, Matrix &gradient_p,
                            Matrix &curv_r, Matrix &curv_p, Matrix &curv_rp);
  /** sub-routine for count decomposition sampling */
  double sample_contributions_sub(const features_t &global_features, size_t g,
                                  size_t s, RNG &rng, Matrix &contrib_gene_type,
                                  Matrix &contrib_spot_type, Matrix &log_ratios,
                                  Matrix &theta_explained_spot_type_,
                                  Float &sigma_explained_spot_,
                                  bool dropout_gene, bool dropout_spot) const;

  /** sample spot scaling factors */
  void sample_spot(const features_t &global_features);

  /** sample baseline feature */
  void sample_baseline(const features_t &global_features);

  Vector marginalize_genes(const features_t &global_features) const;
  Vector marginalize_spots() const;

  // computes a matrix M(g,t)
  // with M(g,t) = baseline_phi(g) global_phi(g,t) sum_s theta(s,t) sigma(s)
  Matrix explained_gene_type(const features_t &global_features) const;
  // computes a matrix M(g,t)
  // with M(g,t) = baseline_phi(g) global_phi(g,t) phi(g,t) sum_s theta(s,t) sigma(s)
  Matrix expected_gene_type(const features_t &global_features) const;
  // computes a matrix M(s,t)
  // with M(s,t) = sigma(s) sum_g baseline_phi(g) phi(g,t) global_phi(g,t)
  Matrix explained_spot_type(const features_t &global_features) const;
  // computes a matrix M(s,t)
  // with M(s,t) = theta(s,t) sigma(s) sum_g baseline_phi(g) phi(g,t) global_phi(g,t)
  Matrix expected_spot_type(const features_t &global_features) const;
  // computes a vector V(g)
  // with V(g) = sum_t phi(g,t) global_phi(g,t) sum_s theta(s,t) sigma(s)
  Vector explained_gene(const features_t &global_features) const;

  std::vector<std::vector<size_t>> active_factors(
      const features_t &global_features, double threshold = 1.0) const;

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
      prev_grad_theta(S, T),
      prev_grad_spot(S),
      prev_sign_theta(S, T, arma::fill::zeros),
      prev_sign_spot(S, arma::fill::zeros),
      features(G, T, parameters),
      baseline_feature(G, 1, parameters),
      weights(S, T, parameters),
      field(Matrix(S, T, arma::fill::ones)),
      lambda_gene_spot(G, S, arma::fill::zeros),
      spot(S, arma::fill::ones) {
  LOG(debug) << "Experiment G = " << G << " S = " << S << " T = " << T;
  prev_grad_theta.fill(parameters.sgd_step_size);
  prev_grad_spot.fill(parameters.sgd_step_size);
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

  // if (not parameters.targeted(Target::phi_local))
    features.matrix.ones();

  // if (not parameters.targeted(Target::phi_prior_local))
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
    write_matrix(expected_spot_type(global_features),
                 prefix + "expected-mix" + FILENAME_ENDING,
                 parameters.compression_mode, spot_names, factor_names, order);
#pragma omp section
    write_matrix(expected_gene_type(global_features),
                 prefix + "expected-features" + FILENAME_ENDING,
                 parameters.compression_mode, gene_names, factor_names, order);
#pragma omp section
    {
      auto phi_marginal = marginalize_genes(global_features);
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
void Experiment<Type>::gibbs_sample(const features_t &global_features) {
  // TODO reactivate
  // if (false)
  //   if (parameters.targeted(Target::contributions))
  //     sample_contributions(global_features, );

  if (parameters.targeted(Target::theta))
    weights.sample_field(*this, field, global_features);

  // TODO add CLI switch
  auto order = random_order(5);
  for (auto &o : order)
    switch (o) {
      case 0:
        // TODO add baseline prior
        if (parameters.targeted(Target::baseline))
          sample_baseline(global_features);
        break;

      case 1:
        if (parameters.targeted(Target::theta_prior)
            and parameters.theta_local_priors) {
          throw("Not implemented");
          // TODO re-implement
          /*
          Matrix feature_matrix = features.matrix % global_features.matrix;
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
          features.prior.sample(*this, global_features);
        break;

      case 3:
        if (parameters.targeted(Target::phi_local))
          features.sample(*this, global_features);
        break;

      case 4:
        if (parameters.targeted(Target::spot))
          sample_spot(global_features);
        break;

      default:
        break;
    }
  LOG(info) << "column sums of theta: " << colSums<Vector>(weights.matrix).t();
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

template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

template <typename T>
void newton_raphson(const T &grad_r, const T &grad_p, const T &curv_r,
                    const T &curv_p, const T &curv_rp, T &r, T &p, const T&cnts) {
  min_max("r", r);
  min_max("p", p);
  r = log(r);
  p = log(p);
  min_max("log r", r);
  min_max("log p", p);
  min_max("grad r", grad_r);
  min_max("grad p", grad_p);
  min_max("curv r", curv_r);
  min_max("curv p", curv_p);
  min_max("curv rp", curv_rp);
  // const double alpha = 1.00;
  const double alpha = 0.5;
  const size_t N = grad_r.size();

  size_t neg_det = 0;
  size_t zero_det = 0;
// #pragma omp parallel if(DO_PARALLEL)
  {
    // Hessian matrix
    Matrix H(2, 2);

    Vector v(2);
    Vector g(2);
    Vector w(2);
// #pragma omp for
    for (size_t n = 0; n < N; ++n) {
      // set up Hessian matrix
      H(0, 0) = curv_r[n];
      H(0, 1) = curv_rp[n];
      H(1, 0) = curv_rp[n];
      H(1, 1) = curv_p[n];

      // H(0, 0) = 0;
      // H(0, 1) = 0;
      // H(1, 0) = 0;
      // H(1, 1) = 1;

      // check determinant of Hessian to ensure H is negative definite
      // if it is, then the Newton direction will be an ascent direction.
      double det = arma::det(H);

      LOG(verbose) << "det(H)= " << det
                     << " n=" << n << " k=" << cnts[n]
                     << " r=" << r[n] << " p=" << p[n]
                     << " grad_r=" << grad_r[n]
                     << " grad_p=" << grad_p[n]
                     << " H=" << H;

      if (det == 0) {
        LOG(warning) << "Warning: determinant of Hessian is zero: " << det
                     << " n=" << n << " r=" << r[n] << " p=" << p[n]
                     << " grad_r=" << grad_r[n]
                     << " grad_p=" << grad_p[n]
                     << " H=" << H;
        zero_det++;
      } else {
      // else {
        if (det > 0 and H(0,0) < 0 and H(1,1) < 0) {
          LOG(debug) << "OK: positive determinant of Hessian: " << det
            << " and negative curvatures! n=" << n << " r=" << r[n] << " p=" << p[n]
            << " H=" << H;
        } else {
          neg_det++;
          LOG(warning) << "Something may be not right!";
          if (H(0, 0) >= 0 or H(1, 1) >= 0) {
            LOG(fatal) << "Non-negative curvatures! n=" << n << " r=" << r[n]
                       << " p=" << p[n] << " H=" << H;
            // assert(false);
          }
          if(det < 0)
            LOG(fatal) << "Negative determinant of Hessian! n=" << n << " r=" << r[n]
                       << " p=" << p[n] << " H=" << H;
        }
        if (H(0, 0) >= 0)
          LOG(warning) << "Warning: non-negative curvature for r!";
        if (H(1, 1) >= 0)
          LOG(warning) << "Warning: non-negative curvature for p!";

        if (H(0, 0) < 0 and H(1, 1) < 0) {
          v(0) = r[n];
          v(1) = p[n];

          g(0) = grad_r[n];
          g(1) = grad_p[n];

          // H(0,0) -= 1e-3;
          // H(1,1) -= 1e-3;

          auto H_inv = arma::inv(H);
          // compute next location
          Vector x = -alpha * H_inv * g;
          // x(0) = 0;
          // x(1) = 0;

          if (true) {
            LOG(info) << "H = " << std::endl << H;
            LOG(info) << "H_inv = " << std::endl << H_inv;
            LOG(info) << "g = " << std::endl << g;
            LOG(info) << "x = " << std::endl << x;
          }

          /*
          Vector y = exp(x);
          w = v % y;
          */
          w = v + x;

          if (true) {
            // LOG(info) << "y = " << y;
            LOG(info) << "v = " << std::endl << v;
            LOG(info) << "w = " << std::endl << w;
            LOG(info) << "exp v = " << std::endl << exp(v);
            LOG(info) << "exp w = " << std::endl << exp(w);
          }

          // update
          r[n] = w(0);
          p[n] = w(1);

          if (std::isnan(w(0)) or std::isnan(w(1))) { // or w(0) < 1e-300 or w(1) < 1e-300) {
            LOG(fatal) << "w=" << w;
            LOG(fatal) << "nan!";
            assert(false);
          }
          if (H(0, 0) >= 0 or H(1, 1) >= 0)
            assert(false);
        }
      }
    }
  }
  LOG(info) << neg_det << " / " << N << " = " << 100.0 * neg_det / N << "\% determinants were negative";
  LOG(info) << zero_det << " / " << N << " = " << 100.0 * zero_det / N << "\% determinants were zero";
  min_max("log r", r);
  min_max("log p", p);
  min_max("grad r", grad_r);
  min_max("grad p", grad_p);
  min_max("curv r", curv_r);
  min_max("curv p", curv_p);
  min_max("curv rp", curv_rp);

  r = exp(r);
  p = exp(p);

  min_max("r", r);
  min_max("p", p);
}

template <typename T, typename U>
void rprop_update(const T &grad, U &prev_sgn, T &rate, T &data) {
  // const double bump_up = 1.2;
  // const double bump_down = 0.5;
  const double eta_plus = 1.1;
  const double eta_minus = 1.0 - 1.0 / 3;

  const double max_change = log(10);
  // const double min_change = 1 / max_change;
  const double min_change = 0;

  auto grad_iter = grad.begin();
  auto data_iter = data.begin();
  auto sgn_iter = prev_sgn.begin();
  size_t caseP = 0, case0 = 0, caseN = 0;
  for (auto &r : rate) {
    auto sgn_grad = sgn(*grad_iter);
    switch (sgn_grad * *sgn_iter) {
      case 1:
        r = std::min(max_change, r * eta_plus);
        caseP++;
      case 0:
        *data_iter *= exp(sgn_grad * r);
        *sgn_iter = sgn_grad;
        case0++;
        break;
      case -1:
        r = std::max(min_change, r * eta_minus);
        *sgn_iter = 0;
        caseN++;
        break;
    }
    grad_iter++;
    data_iter++;
    sgn_iter++;
  }
  case0 -= caseP;
  LOG(info) << "+1/0/-1 " << caseP << "/" << case0 << "/" << caseN;
}

template <typename Type>
/** sample count decomposition */
void Experiment<Type>::sample_contributions(const features_t &global_features,
                                            Matrix &g_r, Matrix &g_p,
                                            Matrix &curv_r, Matrix &curv_p,
                                            Matrix &curv_rp) {
  LOG(verbose) << "Sampling contributions";
  Matrix grad_theta(S, T, arma::fill::zeros);
  Vector grad_spot(S, arma::fill::zeros);

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

  // reset contributions for those genes that are not dropped
  for (size_t g = 0; g < G; ++g)
    if (not dropout_gene[g])
      for (size_t t = 0; t < T; ++t)
        contributions_gene_type(g, t) = 0;

  // reset contributions for those spots that are not dropped
  for (size_t s = 0; s < S; ++s)
    if (not dropout_spot[s])
      for (size_t t = 0; t < T; ++t)
        contributions_spot_type(s, t) = 0;

  if (parameters.dropout_gene > 0.0)
    LOG(verbose) << "Gene dropout rate = " << parameters.dropout_gene * 100
                 << "\% Dropping " << dropped_genes << " genes";
  if (parameters.dropout_spot > 0.0)
    LOG(verbose) << "Spot dropout rate = " << parameters.dropout_spot * 100
                 << "\% Dropping " << dropped_spots << " spots";

  size_t total_accepted = 0;
  size_t total_performed = 0;

#pragma omp parallel if (DO_PARALLEL)
  {
    Matrix contrib_gene_type(G, T, arma::fill::zeros);
    Matrix contrib_spot_type(S, T, arma::fill::zeros);

    Matrix g_theta(S, T, arma::fill::zeros);
    Vector g_spot(S, arma::fill::zeros);
    size_t accepted = 0;
    size_t performed = 0;

    auto rng = EntropySource::rngs[omp_get_thread_num()];
#pragma omp for
    for (size_t g = 0; g < G; ++g) {
      for (size_t s = 0; s < S; ++s)
        // if(RandomDistribution::Uniform(rng) < parameters.sgd_inclusion_prob)
        if (not dropout_spot[s]) {
          std::vector<size_t> cnts(T, 0);
          if (data.counts(g, s) > 0) {
            auto log_posterior_difference = [&](const std::vector<size_t> &v,
                                                size_t i, size_t j, size_t n) {
              double l = 0;

              const double r_i = global_features.prior.r(g, i);
              const double no_i = global_features.prior.p(g, i);
              const double prod_i = r_i * theta(s, i) * spot(s);

              const double r_j = global_features.prior.r(g, j);
              const double no_j = global_features.prior.p(g, j);
              const double prod_j = r_j * theta(s, j) * spot(s);

              // TODO handle infinities
              /*
              if(prod + v[t] == 0)
                return -std::numeric_limits<double>::infinity();
               */

              // subtract current score contributions
              l -= lgamma(prod_i + v[i]) - lgamma(v[i] + 1)
                   - v[i] * log(1 + no_i);
              l -= lgamma(prod_j + v[j]) - lgamma(v[j] + 1)
                   - v[j] * log(1 + no_j);

              // add proposed score contributions
              l += lgamma(prod_i + v[i] - n) - lgamma(v[i] - n + 1)
                   - (v[i] - n) * log(1 + no_i);
              l += lgamma(prod_j + v[j] + n) - lgamma(v[j] + n + 1)
                   - (v[j] + n) * log(1 + no_j);

              return l;
            };

            // TODO use full-conditional expected counts instead of one sample

            // sample x_gst for all t
            std::vector<double> mean_prob(T, 0);
            double z = 0;
            for (size_t t = 0; t < T; ++t)
              z += mean_prob[t] = global_features.prior.r(g, t)
                                  / global_features.prior.p(g, t) * theta(s, t);
            for (size_t t = 0; t < T; ++t)
              mean_prob[t] /= z;
            cnts = sample_multinomial<size_t>(
                data.counts(g, s), begin(mean_prob), end(mean_prob), rng);

            if (T > 1) {
              // perform several Metropolis-Hastings steps
              const size_t initial = 100;
              int n_iter = initial;
              while (n_iter--) {
                // modify
                size_t i = std::uniform_int_distribution<size_t>(0, T - 1)(rng);
                while (cnts[i] == 0)
                  i = std::uniform_int_distribution<size_t>(0, T - 1)(rng);
                size_t j = std::uniform_int_distribution<size_t>(0, T - 1)(rng);
                while (i == j)
                  j = std::uniform_int_distribution<size_t>(0, T - 1)(rng);
                size_t n
                    = std::uniform_int_distribution<size_t>(1, cnts[i])(rng);

                // calculate score difference
                double l = log_posterior_difference(cnts, i, j, n);
                if (l > 0 or (std::isfinite(l)
                              and log(RandomDistribution::Uniform(rng))
                                          * parameters.temperature
                                      <= l)) {
                  // accept the candidate
                  cnts[i] -= n;
                  cnts[j] += n;
                  accepted++;
                }
                performed++;
              }
            }

            for (size_t t = 0; t < T; ++t) {
              contrib_gene_type(g, t) += cnts[t];
              contrib_spot_type(s, t) += cnts[t];
            }
          }

          // calculate gradients
          for (size_t t = 0; t < T; ++t) {
            const double r = global_features.prior.r(g, t);
            const double negodds = global_features.prior.p(g, t);
            const double p = neg_odds_to_prob(negodds);
            const auto prod = r * spot(s) * theta(s, t);
            if (prod < 1e-300)
              LOG(info) << g << " / " << s << " / " << t << " prod=" << prod
                        << " cnts=" << cnts[t]
                        << " cnts+prod=" << prod + cnts[t] << " r=" << r
                        << " sigma=" << spot(s) << " theta=" << theta(s, t)
                        << " no=" << negodds;  // << " p=" << p;
            const double digamma_diff
                // = (cnts[t] == 0)
                = (cnts[t] + prod == prod)
                      ? 0
                      // : (prod == 0 ? -std::numeric_limits<double>::infinity()
                                   : digamma(prod + cnts[t]) - digamma(prod); //);
            const double trigamma_diff
                // = (cnts[t] == 0)
                = (cnts[t] + prod == prod)
                      ? 0
                      // : (prod == 0 ? -std::numeric_limits<double>::infinity() // TODO check this value
                                   : trigamma(prod + cnts[t]) - trigamma(prod); //);

            // q = 1 - p
            const double log_q = log(negodds) - log(negodds + 1);
            const double grad = prod * (digamma_diff + log_q);
            g_r(g, t) += grad;
            g_p(g, t) += (prod - cnts[t] * negodds) / (negodds + 1);
            g_theta(s, t) += grad;
            g_spot(s) += grad;
            // g_p(g, t) += prod - cnts[t] * negodds;  // * (negodds^2 + negodds)^-1

            curv_r(g, t)
                += prod * (log_q + prod * trigamma_diff + digamma_diff);
            curv_p(g, t) += -(prod + cnts[t]) * negodds
                            / (negodds * negodds + 2 * negodds + 1);
            curv_rp(g, t) += prod / (negodds + 1);

            if (not(std::isfinite(curv_r(g, t)) and std::isfinite(curv_p(g, t))
                    and std::isfinite(curv_rp(g, t)))) {
              LOG(fatal) << "Error: found infinite curvature!";
              LOG(fatal) << "g = " << g;
              LOG(fatal) << "t = " << t;
              LOG(fatal) << "r = " << r;
              LOG(fatal) << "negodds = " << negodds;
              LOG(fatal) << "p = " << p;
              LOG(fatal) << "curv_r = " << curv_r(g, t);
              LOG(fatal) << "curv_p = " << curv_p(g, t);
              LOG(fatal) << "curv_rp = " << curv_rp(g, t);
              LOG(fatal) << "grad = " << grad;
              LOG(fatal) << "digamma_diff = " << digamma_diff;
              LOG(fatal) << "trigamma_diff = " << trigamma_diff;
              assert(false);
            }
          }
        }
    }
#pragma omp critical
    {
      contributions_gene_type += contrib_gene_type;
      contributions_spot_type += contrib_spot_type;
      grad_theta += g_theta;
      grad_spot += g_spot;
      total_accepted += accepted;
      total_performed += performed;
    }
  }

  if (total_performed > 0)
    LOG(info) << "accepted=" << 100.0 * total_accepted / total_performed << "%";

  if (true) {
    if (true)
      for (size_t s = 0; s < S; ++s)
        grad_spot(s) += parameters.hyperparameters.spot_a - 1
                        - parameters.hyperparameters.spot_b * spot(s);
    /*
      for (size_t s = 0; s < S; ++s)
        curv_spot(s) += - parameters.hyperparameters.spot_b * spot(s);
    */

    if (true)
      for (size_t s = 0; s < S; ++s)
        for (size_t t = 0; t < T; ++t)
          grad_theta(s, t)
              += weights.prior.r(t) - 1 - weights.prior.p(t) * theta(s, t);
    /*
      for (size_t s = 0; s < S; ++s)
        for (size_t t = 0; t < T; ++t)
        curv_theta(s, t) += - weights.prior.p(t) * theta(s, t);
    */
  }

  parameters.dropout_gene *= parameters.dropout_anneal;
  parameters.dropout_spot *= parameters.dropout_anneal;


  rprop_update(grad_spot, prev_sign_spot, prev_grad_spot, spot);
  rprop_update(grad_theta, prev_sign_theta, prev_grad_theta, weights.matrix);
  min_max("rates spot", prev_grad_spot);
  min_max("rates theta", prev_grad_theta);

  // spot.fill(1.0);
  // weights.matrix.fill(1.0);

  enforce_positive_parameters();

  min_max("theta", weights.matrix);
  min_max("spot", spot);
  min_max("r(theta)", weights.prior.r);
  min_max("p(theta)", weights.prior.p);
}

template <typename Type>
void Experiment<Type>::enforce_positive_parameters() {
  features.enforce_positive_parameters();
  baseline_feature.enforce_positive_parameters();
  weights.enforce_positive_parameters();
  for (size_t g = 0; g < G; ++g) {
    for (size_t t = 0; t < T; ++t) {
      phi(g, t) = std::max<double>(phi(g, t),
                                   std::numeric_limits<double>::denorm_min());
      features.prior.r(g, t) = std::max<double>(
          features.prior.r(g, t), std::numeric_limits<double>::denorm_min());
      features.prior.p(g, t) = std::max<double>(
          features.prior.p(g, t), std::numeric_limits<double>::denorm_min());
    }
  }

}

template <typename Type>
double Experiment<Type>::sample_contributions_sub(
    const features_t &global_features, size_t g, size_t s, RNG &rng,
    Matrix &contrib_gene_type, Matrix &contrib_spot_type, Matrix &log_ratios,
    Matrix &theta_explained_spot_type_, Float &sigma_explained_spot_,
    bool dropout_gene, bool dropout_spot) const {

  /*
  auto expected_observations = [&](size_t t) {
    return global_features.prior.r(g, t) / global_features.prior.p(g, t)
           * baseline_feature.matrix(g) * phi(g, t) * theta(s, t) * spot(s);
  };

  auto phi0 = [&](size_t t) {
    // NOTE: std::gamma_distribution takes a shape and scale parameter
    return std::gamma_distribution<double>(
        global_features.prior.r(g, t) + expected_observations(t),
        1 / (global_features.prior.p(g, t)
             + baseline_feature.matrix(g) * phi(g, t) * theta(s, t) * spot(s)))(
        rng);
        // NOTE: the expectation of this is: r / p
        // NOTE: the variance of this is:
        // r / p / p / (1 + baseline_feature.matrix(g) * phi(g, t)
        //                  * theta(s, t) * spot(s)
        //                  / global_features.prior.p(g, t))
  };

  auto phi_prior = [&](size_t t) {
    // NOTE: std::gamma_distribution takes a shape and scale parameter
    return std::gamma_distribution<double>(
        global_features.prior.r(g, t), 1 / global_features.prior.p(g, t))(rng);
  };
  */

  // NOTE: in principle, lambda(g,s,t) is proportional to the baseline feature
  // and the spot scaling. However, these terms would cancel. Thus, we don't
  // respect them here.

  Vector phi_(T);
  for (size_t t = 0; t < T; ++t)
    // phi_[t] = phi(g, t) * phi_prior(t);
    // phi_[t] = phi(g, t) * global_features.prior.r(g, t)
    //           / global_features.prior.p(g, t);
    phi_[t] = phi(g, t) * global_features.matrix(g, t);

  // TODO remove conditionals; perhaps templatize?
  std::vector<double> rel_rate(T);
  double z = 0;
  if (true) {
    if (dropout_spot)
      for (size_t t = 0; t < T; ++t)
        z += rel_rate[t] = phi_(t);
    else if (dropout_gene)
      for (size_t t = 0; t < T; ++t)
        z += rel_rate[t] = theta(s, t);
    else
      for (size_t t = 0; t < T; ++t)
        z += rel_rate[t] = phi_(t) * theta(s, t);
  } else
    for (size_t t = 0; t < T; ++t)
      z += rel_rate[t] = std::gamma_distribution<double>(
          global_features.prior.r(g, t),
          1 / (global_features.prior.p(g, t) + theta(s, t) * spot(s)))(rng);
  for (size_t t = 0; t < T; ++t)
    rel_rate[t] /= z;

  // if (data.counts(g, s) > 0) {
  if (true) {
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

        const Float x_gst = v[t];
        const Float r_gt = global_features.prior.r(g, t);
        const Float p_gt = global_features.prior.p(g, t);
        const Float r_gt_alt = global_features.alt_prior.r(g, t);
        const Float p_gt_alt = global_features.alt_prior.p(g, t);
        const Float other = baseline_phi(g)
                            * features.matrix(g, t) * theta(s, t) * spot[s];

        log_ratios(g, t) +=
          lgamma(r_gt + x_gst) - (r_gt + x_gst) * log(p_gt + other);
        log_ratios(g, t) -=
          lgamma(r_gt_alt + x_gst) - (r_gt_alt + x_gst) * log(p_gt_alt + other);

        if (true) {
          // NOTE: std::gamma_distribution takes a shape and scale parameter
          const double phi_gst
              = baseline_phi(g) * features.matrix(g, t)
                * std::gamma_distribution<Float>(
                      r_gt + x_gst,
                      1.0 / (p_gt
                             + baseline_phi(g) * features.matrix(g, t)
                                   * theta(s, t)
                                   * spot(s)))(EntropySource::rng);
          theta_explained_spot_type_(s, t) += phi_gst * spot(s);
          sigma_explained_spot_ += phi_gst * theta(s, t);
        } else {
          theta_explained_spot_type_(s, t) += phi_[t] * spot(s);
          sigma_explained_spot_ += phi_[t] * theta(s, t);
        }
      }
    }
  }
  return z;
}

/** sample spot scaling factors */
template <typename Type>
void Experiment<Type>::sample_spot(const features_t &global_features) {
  LOG(verbose) << "Sampling spot scaling factors";
  auto phi_marginal = marginalize_genes(global_features);
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

  if (false)
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t s = 0; s < S; ++s)
    // NOTE: std::gamma_distribution takes a shape and scale parameter
    spot[s] = std::gamma_distribution<Float>(
        parameters.hyperparameters.spot_a + contributions_spot(s),
        1.0 / (parameters.hyperparameters.spot_b + 0))( // TODO hack explained_spot(s)))(
        EntropySource::rng);
}

/** sample baseline feature */
template <typename Type>
void Experiment<Type>::sample_baseline(const features_t &global_features) {
  LOG(verbose) << "Sampling baseline feature from Gamma distribution";

  // TODO add CLI switch
  const double prior1 = parameters.hyperparameters.baseline1;
  const double prior2 = parameters.hyperparameters.baseline2;
  Vector observed = prior1 + contributions_gene;
  Vector explained = prior2 + explained_gene(global_features);

  Partial::perform_sampling(observed, explained, baseline_feature.matrix,
                            parameters.over_relax);
}

template <typename Type>
Vector Experiment<Type>::marginalize_genes(
    const features_t &global_features) const {
  Vector intensities(T, arma::fill::zeros);
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t t = 0; t < T; ++t) {
    double intensity = 0;
    for (size_t g = 0; g < G; ++g)
      intensity += baseline_phi(g) * phi(g, t) * global_features.prior.r(g, t)
                   / global_features.prior.p(g, t);
    intensities[t] = intensity;
  }
  return intensities;
};

template <typename Type>
Vector Experiment<Type>::marginalize_spots() const {
  Vector intensities(T, arma::fill::zeros);
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t t = 0; t < T; ++t) {
    double intensity = 0;
    for (size_t s = 0; s < S; ++s)
      intensity += theta(s, t) * spot[s];
    intensities[t] = intensity;
  }
  return intensities;
}

template <typename Type>
Matrix Experiment<Type>::explained_gene_type(
    const features_t &global_features) const {
  Vector theta_t = marginalize_spots();
  Matrix explained(G, T, arma::fill::zeros);
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t g = 0; g < G; ++g)
    for (size_t t = 0; t < T; ++t)
      explained(g, t) = baseline_phi(g) * global_features.prior.r(g, t)
                        / global_features.prior.p(g, t) * theta_t(t);
  return explained;
}

template <typename Type>
Matrix Experiment<Type>::expected_gene_type(
    const features_t &global_features) const {
  return features.matrix % explained_gene_type(global_features);
}

template <typename Type>
Vector Experiment<Type>::explained_gene(
    const features_t &global_features) const {
  Vector theta_t = marginalize_spots();
  Vector explained(G, arma::fill::zeros);
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t g = 0; g < G; ++g)
    for (size_t t = 0; t < T; ++t)
      explained(g) += phi(g, t) * global_features.prior.r(g, t)
                      / global_features.prior.p(g, t) * theta_t(t);
  return explained;
};

template <typename Type>
Matrix Experiment<Type>::explained_spot_type(
    const features_t &global_features) const {
  Matrix m = Matrix(S, T, arma::fill::ones);
  for (size_t t = 0; t < T; ++t) {
    Float x = 0;
    for (size_t g = 0; g < G; ++g)
      x += baseline_phi(g) * phi(g, t) * global_features.prior.r(g, t)
           / global_features.prior.p(g, t);
    for (size_t s = 0; s < S; ++s)
      m(s, t) *= x * spot(s);
  }
  return m;
}

template <typename Type>
Matrix Experiment<Type>::expected_spot_type(
    const features_t &global_features) const {
  return weights.matrix % explained_spot_type(global_features);
}

template <typename Type>
std::vector<std::vector<size_t>> Experiment<Type>::active_factors(
    const features_t &global_features, double threshold) const {
  auto w = expected_spot_type(global_features);
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
