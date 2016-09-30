#include "io.hpp"
#include "odds.hpp"
#include "parallel.hpp"
#include "pdist.hpp"
#include "priors.hpp"
#include "metropolis_hastings.hpp"
#include "sampling.hpp"
#include "log.hpp"

using namespace std;

namespace PoissonFactorization {
namespace PRIOR {
namespace PHI {

double compute_conditional(const pair<Float, Float> &x, Int observed,
                           Float expected,
                           const Hyperparameters &hyperparameters) {
  const Float r = x.first;
  const Float p = x.second;
  return log_beta_neg_odds(p, hyperparameters.phi_p_1,
                           hyperparameters.phi_p_2)
         // NOTE: gamma_distribution takes a shape and scale parameter
         + log_gamma(r, hyperparameters.phi_r_1,
                     1 / hyperparameters.phi_r_2)
         // The next lines are part of the negative binomial distribution.
         // The other factors aren't needed as they don't depend on either of
         // r[g][t] and p[g][t], and thus would cancel when computing the score
         // ratio.
         + r * log(p)
         - (r + observed) * log(p + expected)
         + lgamma(r + observed) - lgamma(r);
}

Gamma::Gamma(size_t dim1_, size_t dim2_, const Parameters &params)
    : dim1(dim1_), dim2(dim2_), r(dim1, dim2), p(dim1, dim2), parameters(params) {
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

double fnc2(double r, double x, double gamma, double theta) {
  return digamma(r+x) - digamma(r) + log(gamma) - log(theta+gamma);
}

double dfnc2(double r, double x, double gamma, double theta) {
  return trigamma(r+x) - trigamma(r);
}

double fnc(double r, double x) {
  return digamma(r+x) - digamma(r) + log(r) - log(r+x);
}

double dfnc(double r, double x) {
  return trigamma(r+x) - trigamma(r) + 1/r - 1/(r+x);
}

void Gamma::sample_mh(const Matrix &theta, const IMatrix &contributions_gene_type,
                   const Vector &spot_scaling,
                   Float experiment_scaling) {
  LOG(info) << "Sampling P and R of Φ";

  auto gen = [&](const std::pair<Float, Float> &x, std::mt19937 &rng) {
    std::normal_distribution<double> rnorm;
    const double f1 = exp(rnorm(rng));
    const double f2 = exp(rnorm(rng));
    return std::pair<Float, Float>(f1 * x.first, f2 * x.second);
  };

  for (size_t t = 0; t < theta.n_cols; ++t) {
    Float weight_sum = 0;
    for (size_t s = 0; s < theta.n_rows; ++s)
      weight_sum += theta(s, t) * spot_scaling[s];

    MetropolisHastings mh(parameters.temperature);

#pragma omp parallel for if (DO_PARALLEL)
    for (size_t g = 0; g < dim1; ++g) {
      const Int count_sum = contributions_gene_type(g, t);
      const size_t thread_num = omp_get_thread_num();
      auto res = mh.sample(std::pair<Float, Float>(r(g, t), p(g, t)),
                           parameters.n_iter, EntropySource::rngs[thread_num],
                           gen, compute_conditional, count_sum, weight_sum,
                           parameters.hyperparameters);
      r(g, t) = res.first;
      p(g, t) = res.second;
    }
  }
}

void Gamma::store(const std::string &prefix,
                  const std::vector<std::string> &gene_names,
                  const std::vector<std::string> &factor_names) const {
  write_matrix(r, prefix + "_prior-r.txt", gene_names, factor_names);
  write_matrix(p, prefix + "_prior-p.txt", gene_names, factor_names);
}

void Gamma::lift_sub_model(const Gamma &sub_model, size_t t1, size_t t2) {
  for (size_t g = 0; g < dim1; ++g) {
    r(g, t1) = sub_model.r(g, t2);
    p(g, t1) = sub_model.p(g, t2);
  }
}

Dirichlet::Dirichlet(size_t dim1_, size_t dim2_, const Parameters &parameters)
    : dim1(dim1_),
      dim2(dim2_),
      alpha_prior(parameters.hyperparameters.alpha),
      alpha(dim1, dim2) {
  alpha.fill(alpha_prior);
}

Dirichlet::Dirichlet(const Dirichlet &other)
    : dim1(other.dim1),
      dim2(other.dim2),
      alpha_prior(other.alpha_prior),
      alpha(other.alpha) {}

void Dirichlet::sample(const Matrix &theta,
                       const IMatrix &contributions_gene_type,
                       const Vector &spot_scaling,
                       Float experiment_scaling) const {}

void Dirichlet::store(const std::string &prefix,
                      const std::vector<std::string> &gene_names,
                      const std::vector<std::string> &factor_names) const {}

void Dirichlet::lift_sub_model(const Dirichlet &sub_model, size_t t1,
                               size_t t2) const {}

ostream &operator<<(ostream &os, const Gamma &x) {
  print_matrix_head(os, x.r, "R of Φ");
  print_matrix_head(os, x.p, "P of Φ");
  return os;
}

ostream &operator<<(ostream &os, const Dirichlet &x) {
  // do nothing, as Dirichlet class does not have random variables
  return os;
}
}

namespace THETA {

double compute_conditional(const pair<Float, Float> &x,
                           const vector<Int> &count_sums,
                           const vector<Float> &weight_sums,
                           const Hyperparameters &hyperparameters) {
  const size_t S = count_sums.size();
  const Float current_r = x.first;
  const Float current_p = x.second;
  double r = log_beta_neg_odds(current_p, hyperparameters.theta_p_1,
                               hyperparameters.theta_p_2)
             // NOTE: gamma_distribution takes a shape and scale parameter
             + log_gamma(current_r, hyperparameters.theta_r_1,
                         1 / hyperparameters.theta_r_2)
             + S * (current_r * log(current_p) - lgamma(current_r));
#pragma omp parallel for reduction(+ : r) if (DO_PARALLEL)
  for (size_t s = 0; s < S; ++s)
    // The next line is part of the negative binomial distribution.
    // The other factors aren't needed as they don't depend on either of
    // r[t] and p[t], and thus would cancel when computing the score
    // ratio.
    r += lgamma(current_r + count_sums[s])
         - (current_r + count_sums[s]) * log(current_p + weight_sums[s]);
  return r;
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
  for (size_t t = 0; t < dim2; ++t)
    // NOTE: std::gamma_distribution takes a shape and scale parameter
    r[t] = std::gamma_distribution<Float>(
        parameters.hyperparameters.theta_r_1,
        1 / parameters.hyperparameters.theta_r_2)(EntropySource::rng);
}

void Gamma::initialize_p() {
  // initialize p_theta
  LOG(debug) << "Initializing P of Θ.";
  for (size_t t = 0; t < dim2; ++t)
    if (false)  // TODO make this CLI-switchable
      p[t] = prob_to_neg_odds(
          sample_beta<Float>(parameters.hyperparameters.theta_p_1,
                             parameters.hyperparameters.theta_p_2));
    else
      p[t] = 1;
}

void Gamma::sample(const Matrix &phi, const IMatrix &contributions_spot_type,
                   const Vector &spot_scaling,
                   Float experiment_scaling) {
  LOG(info) << "Sampling P and R of Θ";

  auto gen = [&](const std::pair<Float, Float> &x, std::mt19937 &rng) {
    std::normal_distribution<double> rnorm;
    const double f1 = exp(rnorm(rng));
    const double f2 = exp(rnorm(rng));
    return std::pair<Float, Float>(f1 * x.first, f2 * x.second);
  };

  for (size_t t = 0; t < phi.n_cols; ++t) {
    Float weight_sum = 0;
#pragma omp parallel for reduction(+ : weight_sum) if (DO_PARALLEL)
    for (size_t g = 0; g < phi.n_rows; ++g)
      weight_sum += phi(g, t);
    MetropolisHastings mh(parameters.temperature);

    std::vector<Int> count_sums(dim1, 0);
    std::vector<Float> weight_sums(dim1, 0);
#pragma omp parallel for if (DO_PARALLEL)
    for (size_t s = 0; s < dim1; ++s) {
      count_sums[s] = contributions_spot_type(s, t);
      weight_sums[s] = weight_sum * spot_scaling[s];
    }
    auto res = mh.sample(std::pair<Float, Float>(r[t], p[t]), parameters.n_iter,
                         EntropySource::rng, gen, compute_conditional,
                         count_sums, weight_sums, parameters.hyperparameters);
    r[t] = res.first;
    p[t] = res.second;
  }
}

void Gamma::store(const std::string &prefix,
                  const std::vector<std::string> &spot_names,
                  const std::vector<std::string> &factor_names) const {
  write_vector(r, prefix + "_prior-r.txt", factor_names);
  write_vector(p, prefix + "_prior-p.txt", factor_names);
}

void Gamma::lift_sub_model(const Gamma &sub_model, size_t t1, size_t t2) {
  r(t1) = sub_model.r(t2);
  p(t1) = sub_model.p(t2);
}

Dirichlet::Dirichlet(size_t dim1_, size_t dim2_,
                     const Parameters &parameters)
    : dim1(dim1_),
      dim2(dim2_),
      alpha_prior(parameters.hyperparameters.alpha),
      alpha(dim1, alpha_prior) {}

Dirichlet::Dirichlet(const Dirichlet &other)
    : dim1(other.dim1),
      dim2(other.dim2),
      alpha_prior(other.alpha_prior),
      alpha(other.alpha) {}

void Dirichlet::sample(const Matrix &theta,
                       const IMatrix &contributions_gene_type,
                       const Vector &spot_scaling,
                       Float experiment_scaling) const {}

void Dirichlet::store(const std::string &prefix,
                      const std::vector<std::string> &spot_names,
                      const std::vector<std::string> &factor_names) const {}

void Dirichlet::lift_sub_model(const Dirichlet &sub_model, size_t t1,
                               size_t t2) const {}

ostream &operator<<(ostream &os, const Gamma &x) {
  print_vector_head(os, x.r, "R of Θ");
  print_vector_head(os, x.p, "P of Θ");
  return os;
}

ostream &operator<<(ostream &os, const Dirichlet &x) {
  // do nothing, as Dirichlet class does not have random variables
  return os;
}
}
}
}
