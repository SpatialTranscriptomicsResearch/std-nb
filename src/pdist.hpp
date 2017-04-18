#ifndef PDIST_HPP
#define PDIST_HPP

#include "log.hpp"
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

/* Based on the following publication:
 * The distribution of the sum of independent gamma random variables
 * P. G. Moschopoulos
 * Ann. Inst. Statist. Math.
 * 37 (1985), Part A, 541-544
 */
template <typename V>
double convolved_gamma(double x, size_t K, const V &shapes, const V &scales) {
  const size_t N = shapes.size();
  assert(scales.size() == N);

  if (N == 1)
    return exp(log_gamma(x, shapes[0], scales[0]));

  if(x == 0)
    return 0;
  /*
  if(x == 0) {
    assert(false);
    // TODO this case needs special treatment
    double p = 0;
    for (size_t n = 0; n < N; ++n)
      p += log_gamma(0, shapes[n], scales[n]);
    return exp(p);
  }
  */

  const double min_scale = *std::min_element(begin(scales), end(scales));
  double C = 1;
  for (size_t n = 0; n < N; ++n)
    C *= std::pow(min_scale / scales[n], shapes[n]);

  LOG(verbose) << "C=" << C;

  V gamma(K);
  for (size_t k = 0; k < K; ++k) {
    gamma[k] = 0;
    for (size_t n = 0; n < N; ++n) {
      gamma[k] += shapes[n] * std::pow(1 - min_scale / scales[n], k + 1);
      gamma[k] /= (k + 1);
      LOG(verbose) << "k=" << k << " n=" << n << " gamma[k]=" << gamma[k];
    }
  }

  double rho = 0;
  for (size_t n = 0; n < N; ++n)
    rho += shapes[n];
  LOG(verbose) << "rho=" << rho;

  V delta(K);
  delta[0] = 1;
  for (size_t k = 1; k < K; ++k) {
    delta[k] = 0;
    for (size_t i = 1; i <= k; ++i)
      delta[k] += i * gamma[i-1] * delta[k - i];
    delta[k] /= k;
    LOG(verbose) << "k=" << k << " delta[k]=" << delta[k];
  }

  LOG(verbose) << "C=" << C << " rho=" << rho;

  double q = 0;
  for (size_t k = 0; k < K; ++k) {
    double p = exp(log(delta[k]) + (rho + k - 1) * log(x) - x / min_scale
                    - lgamma(rho + k) - (rho + k) * log(min_scale));
    q += p;
    LOG(verbose) << "k=" << k << " p=" << p << " q=" << q;
  }

  return C * q;
}

/** Compute the logarithms of the sum of two logarithmic values */
inline double exp_add(double x, double y) {
  if (std::isinf(x) and x < 0 and std::isinf(y) and y < 0)
    return -std::numeric_limits<double>::infinity();
  double m = std::max(x, y);
  return log(exp(x - m) + exp(y - m)) + m;
  // TODO make more efficient
  // if(x > y)
  // return log(1 + exp(y - x)) + x;
  // else
  // return log(exp(x-y) + 1) + y
}

/* Based on the following publication:
 * On the convolution of the negative binomial random variables
 * Edward Furman
 * Statistics & probability Letters
 * 77 (2007), 169-172
 *
 * returns the logarithm of the probability of observing X=x counts where X is
 * distributed as the convolution of negative binomially distributed variables
 *
 * rs: vector of r parameters for the negative binomial distributions
 * ps: vector of p parameters for the negative binomial distributions
 */
template <typename V>
double convolved_negative_binomial(double x, size_t K, const V &rs,
                                   const V &ps) {
  const size_t N = rs.size();
  assert(static_cast<size_t>(ps.size()) == N);

  K = std::min<size_t>(K, N);

  V neg_odds(N);
  for (size_t n = 0; n < N; ++n)
    neg_odds[n] = (1 - ps[n]) / ps[n];

  const double max_p = *std::max_element(begin(ps), end(ps));
  const double max_neg_odds = (1 - max_p) / max_p;
  double R = 1;
  for (size_t n = 0; n < N; ++n)
    R *= std::pow(neg_odds[n] / max_neg_odds, -rs[n]);

  LOG(debug) << "R=" << R;

  V xi(K);
  for (size_t k = 0; k < K; ++k) {
    xi[k] = 0;
    for (size_t n = 0; n < N; ++n) {
      xi[k] += rs[n] * std::pow(1 - max_neg_odds / neg_odds[n], k + 1);
      xi[k] /= (k + 1);
      LOG(debug) << "k=" << k << " n=" << n << " xi[k]=" << xi[k];
    }
  }

  double alpha = 0;
  for (size_t n = 0; n < N; ++n)
    alpha += rs[n];
  LOG(debug) << "alpha=" << alpha;

  V delta(K);
  delta[0] = 1;
  for (size_t k = 1; k < K; ++k) {
    delta[k] = 0;
    for (size_t i = 1; i <= k; ++i)
      delta[k] += i * xi[i - 1] * delta[k - i];
    delta[k] /= k;
    LOG(debug) << "k=" << k << " delta[k]=" << delta[k];
  }

  double q = -std::numeric_limits<double>::infinity();
  for (size_t k = 0; k < K; ++k) {
    // double p = exp(log(delta[k]) + (rho + k - 1) * log(x) - x / min_scale
    //                 - lgamma(rho + k) - (rho + k) * log(min_scale));
    if (delta[k] > 0) {
      double log_p = log(delta[k]) + lgamma(alpha + x + k) - lgamma(alpha + k)
                     - lgamma(x + 1) + (alpha + k) * log(max_p)
                     + x * log(1 - max_p);
      q = exp_add(q, log_p);
      LOG(debug) << "k=" << k << " p=" << log_p << " q=" << q;
    } else {
      LOG(debug) << "k=" << k
                 << " p=" << -std::numeric_limits<double>::infinity()
                 << " q=" << q;
    }
  }

  return R * q;
};

#endif
