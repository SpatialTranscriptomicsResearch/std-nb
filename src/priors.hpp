#ifndef PRIORS_HPP
#define PRIORS_HPP

#include <cstddef>
#include "entropy.hpp"
#include "log.hpp"
#include "metropolis_hastings.hpp"
#include "odds.hpp"
#include "parallel.hpp"
#include "parameters.hpp"
#include "sampling.hpp"
#include "types.hpp"

namespace PoissonFactorization {

const std::string FILENAME_ENDING = ".tsv";

namespace PRIOR {
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
  /** sample p_phi and r_phi */
  /* This chooses first r with Metropolis-Hastings then p from the posterior */
  template <typename Type, typename... Args>
  void sample(const Type &experiment, const Args &... args);

  /* This is a simple Metropolis-Hastings sampling scheme */
  template <typename Type, typename... Args>
  void sample_mh(const Type &experiment, const Args &... args);

  void store(const std::string &prefix,
             const std::vector<std::string> &gene_names,
             const std::vector<std::string> &factor_names,
             const std::vector<size_t> &order) const;
  void restore(const std::string &prefix, const std::string &suffix);

private:
  void initialize_r();
  void initialize_p();
};

template <typename F, typename... Args>
size_t solve_newton(double eps, F fnc, F dfnc, double &x, Args... args) {
  size_t n = 0;
  double f = fnc(x, args...);
  while (fabs(f = fnc(x, args...)) > eps) {
    double df = dfnc(x, args...);
    LOG(debug) << "x = " << x << " f = " << f << " df = " << df;
    double ratio = f / df;
    if (ratio > x)
      x /= 2;
    else
      x -= f / df;
    n++;
  }
  return n;
}

double fnc(double r, double x);
double dfnc(double r, double x);
double fnc2(double r, double x, double gamma, double theta);
double dfnc2(double r, double x, double gamma, double theta);

inline double log_normal_generator(double x, std::mt19937 &rng) {
  return x * exp(std::normal_distribution<Float>(0, 1)(rng));
}

inline double score(double r, double p, double observed, double explained,
                    double h1, double h2) {
  double nb_term
      = lgamma(r + observed) - lgamma(r) + r * (log(p) - log(p + explained));
  // double nb_term = lgamma(r + observed) - lgamma(r) + r * log(p)
  //                  - (r + observed) * log(p + explained);
  // NOTE: log_gamma takes a shape and scale parameter
  double prior_term = log_gamma(r, h1, 1 / h2);
  return nb_term + prior_term;
}

template <typename Type, typename... Args>
void Gamma::sample(const Type &experiment, const Args &... args) {
  LOG(verbose)
      << "Sampling R and P of Φ using Metropolis-Hastings and from the "
         "posterior, respectively.";

  auto expected_gene_type = experiment.expected_gene_type(args...);
  MetropolisHastings mh(parameters.temperature);
  for (size_t t = 0; t < experiment.T; ++t) {
#pragma omp parallel if (DO_PARALLEL)
    {
      const size_t thread_num = omp_get_thread_num();
#pragma omp for
      for (size_t g = 0; g < experiment.G; ++g) {
        const Float observed = experiment.contributions_gene_type(g, t);
        const Float explained = expected_gene_type(g, t);
        LOG(debug) << "observed = " << observed;
        LOG(debug) << "explained = " << explained;
        LOG(debug) << "r(" << g << ", " << t << ") = " << r(g, t);
        LOG(debug) << "p(" << g << ", " << t << ") = " << p(g, t);
        if (parameters.phi_prior_maximum_likelihood) {
          if (observed == 0) {
            r(g, t) = std::gamma_distribution<Float>(
                parameters.hyperparameters.phi_r_1,
                1 / parameters.hyperparameters.phi_r_2)(
                EntropySource::rngs[thread_num]);
          } else {
            // TODO should this be deactivated?
            // // set to arithmetic mean of current value and 1
            r(g, t) = (1 + r(g, t)) / 2;
            auto num_steps = solve_newton(1e-6, fnc2, dfnc2, r(g, t), observed,
                                          p(g, t), explained);
            LOG(debug) << "r'(" << g << ", " << t << ") = " << r(g, t);
            LOG(debug) << "number of steps = " << num_steps;
          }
        } else {
          r(g, t) = mh.sample(r(g, t), parameters.n_iter,
                              EntropySource::rngs[thread_num],
                              log_normal_generator, score, p(g, t), observed,
                              explained, parameters.hyperparameters.phi_r_1,
                              parameters.hyperparameters.phi_r_2);
        }

        p(g, t) = sample_compound_gamma(
            parameters.hyperparameters.phi_p_1 + r(g, t),
            parameters.hyperparameters.phi_p_2 + observed, explained,
            EntropySource::rngs[thread_num]);

        assert(r(g, t) >= 0);
        assert(p(g, t) >= 0);

        LOG(debug) << "p'(" << g << ", " << t << ") = " << p(g, t);

        if (false)
          if (observed > 0) {
            const double pseudo_cnt = 1e-6;
            auto p_ml = r(g, t) / observed * explained;
            auto p_ml_ps
                = r(g, t) / (observed + pseudo_cnt) * (explained + pseudo_cnt);

            LOG(debug) << "p*(" << g << ", " << t << ") = " << p_ml;
            LOG(debug) << "pML " << r(g, t) << " " << p(g, t) << " " << p_ml
                       << " " << p_ml_ps;
          }
        LOG(debug) << std::endl;
      }
    }
  }
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
         // The other factors aren't needed as they don't depend on either of
         // r(g,t) and p(g,t), and thus would cancel when computing the score
         // ratio.
         + r * log(p) - (r + observed) * log(p + explained)
         + lgamma(r + observed) - lgamma(r);
}

template <typename T>
std::pair<T, T> gen_log_normal_pair(const std::pair<T, T> &x,
                                    std::mt19937 &rng) {
  std::normal_distribution<double> rnorm;
  const double f1 = exp(rnorm(rng));
  const double f2 = exp(rnorm(rng));
  return {f1 * x.first, f2 * x.second};
  // return std::pair<Float, Float>(f1 * x.first, f2 * x.second);
};

template <typename Type, typename... Args>
void Gamma::sample_mh(const Type &experiment, const Args &... args) {
  LOG(verbose) << "Sampling P and R of Φ";

  auto expected_gene_type = experiment.expected_gene_type(args...);
  MetropolisHastings mh(parameters.temperature);

  for (size_t t = 0; t < expected_gene_type.n_cols; ++t)
#pragma omp parallel if (DO_PARALLEL)
  {
    const size_t thread_num = omp_get_thread_num();
#pragma omp for
    for (size_t g = 0; g < experiment.G; ++g) {
      const Float observed = experiment.contributions_gene_type(g, t);
      const Float explained = expected_gene_type(g, t);

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
  void sample(const Matrix &theta, const Matrix &contributions_gene_type,
              const Vector &spot_scaling) const;
  void store(const std::string &prefix,
             const std::vector<std::string> &gene_names,
             const std::vector<std::string> &factor_names,
             const std::vector<size_t> &order) const;
  void restore(const std::string &prefix, const std::string &suffix);
};

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
  void sample(const Matrix &observed, const Matrix &explained);

  void store(const std::string &prefix,
             const std::vector<std::string> &spot_names,
             const std::vector<std::string> &factor_names,
             const std::vector<size_t> &order) const;
  void restore(const std::string &prefix, const std::string &suffix);

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
  void sample(const Matrix &observed, const Matrix &explained) const;
  void store(const std::string &prefix,
             const std::vector<std::string> &spot_names,
             const std::vector<std::string> &factor_names,
             const std::vector<size_t> &order) const;
  void restore(const std::string &prefix, const std::string &suffix);
};

/** This routine doesn't print, for the same reason as sample() does nothing */
std::ostream &operator<<(std::ostream &os, const Gamma &x);
std::ostream &operator<<(std::ostream &os, const Dirichlet &x);
}
}
}

#endif
