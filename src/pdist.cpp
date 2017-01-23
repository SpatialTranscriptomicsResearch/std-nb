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

double log_gamma(double x, double shape, double scale) {
  if (x == 0) {
    if (shape == 1)
      return -log(scale);
    else if (shape > 1) {
      assert(false);
      return -std::numeric_limits<double>::infinity();
    } else {
      assert(false);
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

/* Based on the following publication:
 * On the convolution of the negative binomial random variables
 * Edward Furman
 * Statistics & probability Letters
 * 77 (2007), 169-172
 */
double convolved_negative_binomial(double x, size_t K,
                                   const std::vector<double> &rs,
                                   const std::vector<double> &ps) {
  using V = vector<double>;
  const size_t N = rs.size();
  assert(ps.size() == N);

  V neg_odds(N);
  for (size_t n = 0; n < N; ++n)
    neg_odds[n] = (1 - ps[n]) / ps[n];

  const double max_p = *std::max_element(begin(ps), end(ps));
  const double max_neg_odds = (1 - max_p) / max_p;
  double R = 1;
  for (size_t n = 0; n < N; ++n)
    R *= std::pow(neg_odds[n] / max_neg_odds, -rs[n]);

  LOG(verbose) << "R=" << R;

  V xi(K);
  for (size_t k = 0; k < K; ++k) {
    xi[k] = 0;
    for (size_t n = 0; n < N; ++n) {
      xi[k] += rs[n] * std::pow(1 - max_neg_odds / neg_odds[n], k + 1);
      xi[k] /= (k + 1);
      LOG(verbose) << "k=" << k << " n=" << n << " xi[k]=" << xi[k];
    }
  }

  double alpha = 0;
  for (size_t n = 0; n < N; ++n)
    alpha += rs[n];
  LOG(verbose) << "alpha=" << alpha;

  V delta(K);
  delta[0] = 1;
  for (size_t k = 1; k < K; ++k) {
    delta[k] = 0;
    for (size_t i = 1; i <= k; ++i)
      delta[k] += i * xi[i - 1] * delta[k - i];
    delta[k] /= k;
    LOG(verbose) << "k=" << k << " delta[k]=" << delta[k];
  }

  double q = 0;
  for (size_t k = 0; k < K; ++k) {
    // double p = exp(log(delta[k]) + (rho + k - 1) * log(x) - x / min_scale
    //                 - lgamma(rho + k) - (rho + k) * log(min_scale));
    double p
        = exp(log(delta[k]) + lgamma(alpha + x + k) - lgamma(alpha + k)
              - lgamma(x + 1) + (alpha + k) * log(max_p) + x * log(1 - max_p));
    q += p;
    LOG(verbose) << "k=" << k << " p=" << p << " q=" << q;
  }

  return R * q;
}
