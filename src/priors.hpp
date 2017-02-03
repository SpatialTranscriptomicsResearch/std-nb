#ifndef PRIORS_HPP
#define PRIORS_HPP

#include <cstddef>
#include "entropy.hpp"
#include "log.hpp"
#include "metropolis_hastings.hpp"
#include "odds.hpp"
#include "parallel.hpp"
#include "pdist.hpp"
#include "parameters.hpp"
#include "sampling.hpp"
#include "types.hpp"

namespace PoissonFactorization {

const std::string FILENAME_ENDING = ".tsv";

namespace PRIOR {

template <typename T>
std::pair<T, T> gen_log_normal_pair(const std::pair<T, T> &x,
                                    std::mt19937 &rng) {
  std::normal_distribution<double> rnorm;
  const double f1 = exp(rnorm(rng));
  const double f2 = exp(rnorm(rng));
  return {f1 * x.first, f2 * x.second};
};

namespace PHI {

struct Gamma {
  size_t dim1, dim2;
  /** shape parameter for the prior of the loading matrix */
  Matrix r;
  /** scale parameter for the prior of the loading matrix */
  /* Stored as negative-odds */
  Matrix p;
  Parameters parameters;

  Gamma(size_t dim1_, size_t dim2_, const Parameters &params);
  Gamma(const Gamma &other);

  /* This is a simple Metropolis-Hastings sampling scheme */
  template <typename Type, typename... Args>
  void sample(const Type &experiment, const Args &... args);

  void store(const std::string &prefix,
             const std::vector<std::string> &gene_names,
             const std::vector<std::string> &factor_names,
             const std::vector<size_t> &order) const;
  void restore(const std::string &prefix);
  void set_unit(double x = 1.0);
  Matrix ratio() const;

  void enforce_positive_parameters();

private:
  void initialize_r();
  void initialize_p();
};

template <typename F, typename... Args>
inline double log_normal_generator(double x, std::mt19937 &rng) {
  return x * exp(std::normal_distribution<Float>(0, 1)(rng));
}

template <typename T>
double compute_conditional(const std::pair<T, T> &x, Float observed,
                           Float explained,
                           const Hyperparameters &hyperparameters) {
  const T r = x.first;
  const T p = x.second;
  return log_beta_neg_odds(p, hyperparameters.phi_p_1, hyperparameters.phi_p_2)
         // NOTE: gamma_distribution takes a shape and scale parameter
         + log_gamma(r, hyperparameters.phi_r_1, 1 / hyperparameters.phi_r_2)
         // The next lines are part of the negative binomial distribution.
         // Other factors aren't needed as they don't depend on either of
         // r and p, and thus would cancel when computing the score ratio.
         + r * log(p) - (r + observed) * log(p + explained)
         + lgamma(r + observed) - lgamma(r);
}

template <typename Type, typename... Args>
void Gamma::sample(const Type &experiment, const Args &... args) {
  LOG(verbose) << "Sampling P and R of Φ";

  auto explained_gene_type = experiment.explained_gene_type(args...);
  MetropolisHastings mh(parameters.temperature);

  for (size_t t = 0; t < explained_gene_type.n_cols; ++t)
#pragma omp parallel if (DO_PARALLEL)
  {
    const size_t thread_num = omp_get_thread_num();
#pragma omp for
    for (size_t g = 0; g < experiment.G; ++g) {
      const Float observed = experiment.contributions_gene_type(g, t);
      const Float explained = explained_gene_type(g, t);

      auto res
          = mh.sample(std::pair<Float, Float>(r(g, t), p(g, t)),
                      parameters.n_iter, EntropySource::rngs[thread_num],
                      gen_log_normal_pair<Float>, compute_conditional<Float>,
                      observed, explained, parameters.hyperparameters);
      r(g, t) = res.first;
      p(g, t) = res.second;
    }
  }
}

struct Dirichlet {
  size_t dim1, dim2;
  Float alpha_prior;
  Matrix alpha;

  Dirichlet(size_t dim1_, size_t dim2_, const Parameters &parameters);
  Dirichlet(const Dirichlet &other);
  /** This routine does nothing, as this sub-model doesn't have random variables
   * but only hyper-parameters */
  template <typename Type, typename... Args>
  void sample(const Type &experiment, const Args &... args);
  void store(const std::string &prefix,
             const std::vector<std::string> &gene_names,
             const std::vector<std::string> &factor_names,
             const std::vector<size_t> &order) const;
  void restore(const std::string &prefix);
  void set_unit(double x = 1.0);
  Matrix ratio() const;
  double r(size_t a, size_t b) const;
  double p(size_t a, size_t b) const;
};

template <typename Type, typename... Args>
void Dirichlet::sample(const Type &experiment, const Args &... args) {}

/** This routine doesn't print, for the same reason as sample() does nothing */
std::ostream &operator<<(std::ostream &os, const Gamma &x);
std::ostream &operator<<(std::ostream &os, const Dirichlet &x);
}

namespace THETA {

struct Gamma {
  size_t dim1, dim2;
  /** shape parameter for the prior of the mixing matrix */
  Vector r;
  /** scale parameter for the prior of the mixing matrix */
  /* Stored as negative-odds */
  Vector p;
  Parameters parameters;

  Gamma(size_t dim1_, size_t dim2_, const Parameters &params);
  Gamma(const Gamma &other);
  /** sample p_phi and r_phi */
  /* This is a simple Metropolis-Hastings sampling scheme */
  void sample(const Matrix &observed);

  void store(const std::string &prefix,
             const std::vector<std::string> &spot_names,
             const std::vector<std::string> &factor_names,
             const std::vector<size_t> &order) const;
  void restore(const std::string &prefix);

  void enforce_positive_parameters();

private:
  void initialize_r();
  void initialize_p();
};

struct Dirichlet {
  size_t dim1, dim2;
  Float alpha_prior;
  std::vector<Float> alpha;

  Dirichlet(size_t G_, size_t dim2_, const Parameters &parameters);
  Dirichlet(const Dirichlet &other);
  /** This routine does nothing, as this sub-model doesn't have random variables
   * but only hyper-parameters */
  void sample(const Matrix &observed) const;
  void store(const std::string &prefix,
             const std::vector<std::string> &spot_names,
             const std::vector<std::string> &factor_names,
             const std::vector<size_t> &order) const;
  void restore(const std::string &prefix);
};

/** This routine doesn't print, for the same reason as sample() does nothing */
std::ostream &operator<<(std::ostream &os, const Gamma &x);
std::ostream &operator<<(std::ostream &os, const Dirichlet &x);
}
}
}

#endif
