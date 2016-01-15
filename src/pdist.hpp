#ifndef PDIST_HPP
#define PDIST_HPP

#include <unistd.h>
#include <vector>

// Discrete probability distributions

/** Poisson probability mass function for k given a rate parameter lambda */
double log_poisson(size_t k, double lambda);
/** Negative binomial probability mass function for x given a number of failures r and a success probability p */
double log_negative_binomial(size_t x, double r, double p);
double log_negative_binomial(size_t x, double r, double p1, double p2);

/** Negative multinomial probability mass function for x given a number of * failures r and a vector of success probabilities p */
template <typename I, typename F>
double log_negative_multinomial(const std::vector<I> &x, F r,
                                const std::vector<F> &p) {
  size_t S = p.size();
  double q = 0;
  double logp = 0;
  double log_fac_sum = 0;
  double sum = 0;
  for (size_t s = 0; s < S; ++s) {
    q += p[s];
    sum += x[s];
    log_fac_sum += lgamma(x[s] + 1);
    logp += x[s] * log(p[s]);
  }
  q = 1 - q;
  logp += lgamma(r + sum);
  logp -= log_fac_sum;
  logp -= lgamma(r);
  logp += r * log(q);

  return logp;
};

// Continuous probability distributions

/** Gamma probability density function for x given a shape parameter k and a scale parameter theta */
double log_gamma(double x, double k, double theta);
/** Beta probability density function for x given shape parameters alpha and beta */
double log_beta(double x, double alpha, double beta);
/** Dirichlet probability density function for a probability distribution p given a vector of concentration aparameters alpha */
double log_dirichlet(const std::vector<double> &p,
                     const std::vector<double> &alpha);

#endif
