#ifndef PDIST_HPP
#define PDIST_HPP

#include <unistd.h>
#include <cassert>
#include <cmath>
#include <vector>

// Discrete probability distributions

/** Poisson probability mass function for k given a rate parameter lambda */
double log_poisson(size_t k, double lambda);
/** Negative binomial probability mass function for x given r failures and
 * success probability p */
double log_negative_binomial(size_t x, double r, double p);
double log_negative_binomial(size_t x, double r, double p1, double p2);

/** Negative multinomial probability mass function for x given r failures and
 * vector of success probabilities p */
template <typename I, typename F>
double log_negative_multinomial(const std::vector<I> &x, F r,
                                const std::vector<F> &p) {
  size_t S = p.size();
  double q = 0;
  double logp = 0;
  double log_fac_sum = 0;
  double sum = 0;
  for (size_t s = 0; s < S; ++s) {
    assert(p[s] >= 0);
    assert(p[s] <= 1);
    q += p[s];
    sum += x[s];
    log_fac_sum += lgamma(x[s] + 1);
    logp += x[s] * log(p[s]);
  }
  assert(q <= 1);
  q = 1 - q;
  logp += lgamma(r + sum);
  logp -= log_fac_sum;
  logp -= lgamma(r);
  logp += r * log(q);

  return logp;
};

// Continuous probability distributions

/** Gamma probability density function for x given a shape and a scale parameter
 */
double log_gamma(double x, double shape, double scale);
/** Beta probability density function for probability p given shape parameters
 * alpha and beta */
double log_beta(double p, double alpha, double beta);
/** Beta probability density function for odds x=p/(1-p) given shape parameters
 * alpha and beta */
double log_beta_odds(double x, double alpha, double beta);
/** Beta probability density function for odds x=(1-p)/p given shape parameters
 * alpha and beta */
double log_beta_neg_odds(double x, double alpha, double beta);
/** Generalized beta prime probability density function for parameters alpha,
 * beta, p, and q */
double log_generalized_beta_prime(double x, double alpha, double beta, double p,
                                  double q);
/** Generalized beta prime probability density function for parameters alpha,
 * beta, and q, where p=1 */
double log_generalized_beta_prime(double x, double alpha, double beta,
                                  double q);
/** Dirichlet probability density function for a probability distribution p
 * given a vector of concentration aparameters alpha */
double log_dirichlet(const std::vector<double> &p,
                     const std::vector<double> &alpha);

double gamma_cdf(double x, double shape, double scale);
double inverse_gamma_cdf(double p, double shape, double scale);

#endif
