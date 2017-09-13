#include "pdist.hpp"
#include <boost/math/special_functions/gamma.hpp>
#include <cmath>
#include <iostream>

using namespace std;

double log_poisson(size_t k, double lambda) {
  if (lambda == 0)
    return (k == 0 ? 0 : -numeric_limits<double>::infinity());
  return k * log(lambda) - lambda - lgamma(k + 1);
}

double log_dirichlet(const vector<double> &p, const vector<double> &alpha) {
  double l = 0;
  double sum = 0;
  for (auto &a : alpha) {
    l -= lgamma(a);
    sum += a;
  }
  l += lgamma(sum);

  const size_t N = p.size();
  for (size_t n = 0; n < N; ++n)
    l += (alpha[n] - 1) * log(p[n]);

  return l;
}

double log_normal(double x, double mu, double sigma) {
  double std_diff = (x - mu) / sigma;
  return -0.5 * log(2 * M_PI) - log(sigma) - 0.5 * std_diff * std_diff;
}

double log_gamma_rate(double x, double shape, double rate) {
  if (x == 0) {
    if (shape == 1)
      return log(rate);
    else if (shape > 1) {
      LOG(warning) << "Warning: log probability of Gamma distribution wanted "
                      "for x=0 and shape > 1!";
      LOG(warning) << "We could return negative infinity. Aborting.";
      assert(false);
      return -std::numeric_limits<double>::infinity();
    } else {
      LOG(warning) << "Warning: log probability of Gamma distribution wanted "
                      "for x=0 and shape < 1!";
      LOG(warning) << "Returning infinity and continuing.";
      return +std::numeric_limits<double>::infinity();
    }
  } else
    return (shape - 1) * log(x) - x * rate - lgamma(shape) + shape * log(rate);
}

double log_gamma_scale(double x, double shape, double scale) {
  if (x == 0) {
    if (shape == 1)
      return -log(scale);
    else if (shape > 1) {
      LOG(warning) << "Warning: log probability of Gamma distribution wanted "
                      "for x=0 and shape > 1!";
      LOG(warning) << "We could return negative infinity. Aborting.";
      assert(false);
      return -std::numeric_limits<double>::infinity();
    } else {
      LOG(warning) << "Warning: log probability of Gamma distribution wanted "
                      "for x=0 and shape < 1!";
      LOG(warning) << "Returning infinity and continuing.";
      return +std::numeric_limits<double>::infinity();
    }
  } else
    return (shape - 1) * log(x) - x / scale - lgamma(shape)
           - shape * log(scale);
}

double log_beta(double p, double a, double b) {
  double x = (a - 1) * log(p) + (b - 1) * log(1 - p);
  double y = lgamma(a + b) - lgamma(a) - lgamma(b);
  return x + y;
}

double log_beta_odds(double x, double a, double b) {
  return lgamma(a + b) - lgamma(a) - lgamma(b) + (a - 1) * log(x)
         - (a + b - 2) * log(1 + x);
}

double log_beta_neg_odds(double x, double a, double b) {
  return lgamma(a + b) - lgamma(a) - lgamma(b) + (b - 1) * log(x)
         - (a + b - 2) * log(1 + x);
}

double log_negative_binomial(size_t x, double r, double p) {
  return lgamma(x + r) - lgamma(x + 1) - lgamma(r) + x * log(p)
         + r * log(1 - p);
}

double log_negative_binomial(size_t x, double r, double p1, double p2) {
  double logp = log(p1 + p2);
  return lgamma(x + r) - lgamma(x + 1) - lgamma(r) + x * (log(p1) - logp)
         + r * (log(p2) - logp);
}

double log_generalized_beta_prime(double x, double alpha, double beta, double p,
                                  double q) {
  return log(p) + (alpha * p - 1) * log(x / q)
         - (alpha + beta) * log(1 + pow(x / q, p)) - log(q)
         + lgamma(alpha + beta) - lgamma(alpha) - lgamma(beta);
}

double log_generalized_beta_prime(double x, double alpha, double beta,
                                  double q) {
  return (alpha - 1) * log(x / q) - (alpha + beta) * log(1 + x / q) - log(q)
         + lgamma(alpha + beta) - lgamma(alpha) - lgamma(beta);
}

double gamma_cdf(double x, double shape, double scale) {
  return boost::math::gamma_p(shape, x / scale);
}

double inverse_gamma_cdf(double p, double shape, double scale) {
  return boost::math::gamma_p_inv(shape, p) * scale;
}
