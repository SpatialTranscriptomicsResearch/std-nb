#include "priors.hpp"
#include "compression.hpp"
#include "io.hpp"
#include "log.hpp"
#include "metropolis_hastings.hpp"
#include "odds.hpp"
#include "parallel.hpp"
#include "pdist.hpp"
#include "sampling.hpp"

using namespace std;

namespace PoissonFactorization {
namespace PRIOR {
namespace PHI {

Gamma::Gamma(size_t dim1_, size_t dim2_, const Parameters &params)
    : dim1(dim1_),
      dim2(dim2_),
      r(dim1, dim2),
      p(dim1, dim2),
      parameters(params) {
  initialize_r();
  initialize_p();
}

Gamma::Gamma(const Gamma &other)
    : dim1(other.dim1),
      dim2(other.dim2),
      r(other.r),
      p(other.p),
      parameters(other.parameters) {}

void Gamma::set_unit(double x) {
  r.fill(x);
  p.fill(x);
}

Matrix Gamma::ratio() const { return r / p; }

void Gamma::initialize_r() {
  // initialize r_phi
  LOG(debug) << "Initializing R of Φ.";
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t g = 0; g < dim1; ++g) {
    const size_t thread_num = omp_get_thread_num();
    for (size_t t = 0; t < dim2; ++t)
      // NOTE: std::gamma_distribution takes a shape and scale parameter
      r(g, t) = std::gamma_distribution<Float>(
          parameters.hyperparameters.phi_r_1,
          1 / parameters.hyperparameters.phi_r_2)(
          EntropySource::rngs[thread_num]);
  }
}
void Gamma::initialize_p() {
  // initialize p_phi
  LOG(debug) << "Initializing P of Φ.";
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t g = 0; g < dim1; ++g) {
    const size_t thread_num = omp_get_thread_num();
    for (size_t t = 0; t < dim2; ++t)
      p(g, t) = prob_to_neg_odds(sample_beta<Float>(
          parameters.hyperparameters.phi_p_1,
          parameters.hyperparameters.phi_p_2, EntropySource::rngs[thread_num]));
  }
}

void Gamma::store(const std::string &prefix,
                  const std::vector<std::string> &gene_names,
                  const std::vector<std::string> &factor_names,
                  const std::vector<size_t> &order) const {
  write_matrix(r, prefix + "_prior-r" + FILENAME_ENDING,
               parameters.compression_mode, gene_names, factor_names, order);
  write_matrix(p, prefix + "_prior-p" + FILENAME_ENDING,
               parameters.compression_mode, gene_names, factor_names, order);
}

void Gamma::restore(const std::string &prefix) {
  r = parse_file<Matrix>(prefix + "_prior-r" + FILENAME_ENDING, read_matrix, "\t");
  p = parse_file<Matrix>(prefix + "_prior-p" + FILENAME_ENDING, read_matrix, "\t");
}

void Gamma::enforce_positive_parameters() {
  for (size_t i = 0; i < dim1; ++i)
    for (size_t j = 0; j < dim2; ++j) {
      r(i, j) = std::max<double>(r(i, j),
                                 std::numeric_limits<double>::denorm_min());
      p(i, j) = std::max<double>(p(i, j),
                                 std::numeric_limits<double>::denorm_min());
    }
}

Dirichlet::Dirichlet(size_t dim1_, size_t dim2_, const Parameters &parameters)
    : dim1(dim1_),
      dim2(dim2_),
      alpha_prior(parameters.hyperparameters.feature_alpha),
      alpha(dim1, dim2) {
  alpha.fill(alpha_prior);
}

Dirichlet::Dirichlet(const Dirichlet &other)
    : dim1(other.dim1),
      dim2(other.dim2),
      alpha_prior(other.alpha_prior),
      alpha(other.alpha) {}

double Dirichlet::r(size_t a, size_t b) const { return 1; }

double Dirichlet::p(size_t a, size_t b) const { return 1; }

void Dirichlet::set_unit(double x) {}

Matrix Dirichlet::ratio() const { return Matrix(dim1, dim2, arma::fill::ones); }

void Dirichlet::store(const std::string &prefix __attribute__((unused)),
                      const std::vector<std::string> &gene_names
                      __attribute__((unused)),
                      const std::vector<std::string> &factor_names
                      __attribute__((unused)),
                      const std::vector<size_t> &order
                      __attribute__((unused))) const {}

void Dirichlet::restore(const std::string &prefix __attribute__((unused))) {}

ostream &operator<<(ostream &os, const Gamma &x) {
  print_matrix_head(os, x.r, "R of Φ");
  print_matrix_head(os, x.p, "P of Φ");
  return os;
}

ostream &operator<<(ostream &os, const Dirichlet &x __attribute__((unused))) {
  // do nothing, as Dirichlet class does not have random variables
  return os;
}
}

namespace THETA {

template <typename V>
double compute_conditional(const pair<Float, Float> &x, const V &observed,
                           const V &explained,
                           const Hyperparameters &hyperparameters) {
  const size_t S = observed.size();
  const Float r = x.first;
  const Float p = x.second;
  double l
      = log_beta_neg_odds(p, hyperparameters.theta_p_1,
                          hyperparameters.theta_p_2)
        // NOTE: gamma_distribution takes a shape and scale parameter
        + log_gamma(r, hyperparameters.theta_r_1, 1 / hyperparameters.theta_r_2)
        + S * (r * log(p) - lgamma(r));
  for (size_t s = 0; s < S; ++s)
    // The next line is part of the negative binomial distribution.
    // The other factors aren't needed as they don't depend on either of
    // r[t] and p[t], and thus would cancel when computing the score
    // ratio.
    l += lgamma(r + observed[s]) - (r + observed[s]) * log(p + explained[s]);
  return l;
}

template <typename V>
double compute_conditional_gamma(const pair<Float, Float> &x, const V &theta,
                                 const Hyperparameters &hyperparameters) {
  const size_t S = theta.size();
  const Float r = x.first;
  const Float p = x.second;
  double l
      = log_beta_neg_odds(p, hyperparameters.theta_p_1,
                          hyperparameters.theta_p_2)
        // NOTE: gamma_distribution takes a shape and scale parameter
        + log_gamma(r, hyperparameters.theta_r_1, 1 / hyperparameters.theta_r_2);
  for (size_t s = 0; s < S; ++s)
    // NOTE: gamma_distribution takes a shape and scale parameter
    l += log_gamma(theta(s), r, 1 / p);
  return l;
}

Gamma::Gamma(size_t dim1_, size_t dim2_, const Parameters &params)
    : dim1(dim1_), dim2(dim2_), r(dim2), p(dim2), parameters(params) {
  initialize_r();
  initialize_p();
}

Gamma::Gamma(const Gamma &other)
    : dim1(other.dim1),
      dim2(other.dim2),
      r(other.r),
      p(other.p),
      parameters(other.parameters) {}

void Gamma::initialize_r() {
  // initialize r_theta
  LOG(debug) << "Initializing R of Θ.";
  if (parameters.targeted(Target::theta_prior))
    for (size_t t = 0; t < dim2; ++t)
      // NOTE: std::gamma_distribution takes a shape and scale parameter
      r[t] = std::gamma_distribution<Float>(
          parameters.hyperparameters.theta_r_1,
          1 / parameters.hyperparameters.theta_r_2)(EntropySource::rng);
  else
    r.ones();
}

void Gamma::initialize_p() {
  // initialize p_theta
  LOG(debug) << "Initializing P of Θ.";
  // TODO make this CLI-switchable
  if (false and parameters.targeted(Target::theta_prior))
    for (size_t t = 0; t < dim2; ++t)
      p[t] = prob_to_neg_odds(
          sample_beta<Float>(parameters.hyperparameters.theta_p_1,
                             parameters.hyperparameters.theta_p_2));
  else
    p.ones();
}

void Gamma::sample(const Matrix &observed, const Matrix &explained __attribute__((unused))) {
  LOG(verbose) << "Sampling P and R of Θ";
  MetropolisHastings mh(parameters.temperature);
#pragma omp parallel if (DO_PARALLEL)
  {
    const size_t thread_num = omp_get_thread_num();
#pragma omp for
    for (size_t t = 0; t < observed.n_cols; ++t) {
      auto res = mh.sample(std::pair<Float, Float>(r[t], p[t]),
                           parameters.n_iter, EntropySource::rngs[thread_num],
                           gen_log_normal_pair<Float>,
                           compute_conditional_gamma<Vector>,
                           observed.col(t), parameters.hyperparameters);
      r[t] = res.first;
      p[t] = res.second;
    }
  }
}

void Gamma::store(const std::string &prefix,
                  const std::vector<std::string> &spot_names
                  __attribute__((unused)),
                  const std::vector<std::string> &factor_names,
                  const std::vector<size_t> &order) const {
  Vector r_ = r;
  Vector p_ = p;
  if (not order.empty()) {
    for (size_t i = 0; i < dim2; ++i)
      r_[i] = r[order[i]];
    for (size_t i = 0; i < dim2; ++i)
      p_[i] = p[order[i]];
  }
  write_vector(r_, prefix + "_prior-r" + FILENAME_ENDING,
               parameters.compression_mode, factor_names);
  write_vector(p_, prefix + "_prior-p" + FILENAME_ENDING,
               parameters.compression_mode, factor_names);
}

void Gamma::restore(const std::string &prefix) {
  r = parse_file<Vector>(prefix + "_prior-r" + FILENAME_ENDING,
                         read_vector<Vector>, "\t");
  p = parse_file<Vector>(prefix + "_prior-p" + FILENAME_ENDING,
                         read_vector<Vector>, "\t");
}

void Gamma::enforce_positive_parameters() {
  for (size_t i = 0; i < dim2; ++i) {
    r(i) = std::max<double>(r(i), std::numeric_limits<double>::denorm_min());
    p(i) = std::max<double>(p(i), std::numeric_limits<double>::denorm_min());
  }
}

Dirichlet::Dirichlet(size_t dim1_, size_t dim2_, const Parameters &parameters)
    : dim1(dim1_),
      dim2(dim2_),
      alpha_prior(parameters.hyperparameters.mix_alpha),
      alpha(dim1, alpha_prior) {}

Dirichlet::Dirichlet(const Dirichlet &other)
    : dim1(other.dim1),
      dim2(other.dim2),
      alpha_prior(other.alpha_prior),
      alpha(other.alpha) {}

void Dirichlet::sample(const Matrix &observed __attribute__((unused)),
                       const Matrix &explained __attribute__((unused))) const {}

void Dirichlet::store(const std::string &prefix __attribute__((unused)),
                      const std::vector<std::string> &spot_names
                      __attribute__((unused)),
                      const std::vector<std::string> &factor_names
                      __attribute__((unused)),
                      const std::vector<size_t> &order
                      __attribute__((unused))) const {}

void Dirichlet::restore(const std::string &prefix __attribute__((unused))) {}

ostream &operator<<(ostream &os, const Gamma &x) {
  print_vector_head(os, x.r, "R of Θ");
  print_vector_head(os, x.p, "P of Θ");
  return os;
}

ostream &operator<<(ostream &os, const Dirichlet &x __attribute__((unused))) {
  // do nothing, as Dirichlet class does not have random variables
  return os;
}
}
}
}
