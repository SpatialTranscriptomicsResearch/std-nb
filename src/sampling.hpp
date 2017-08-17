#ifndef SAMPLING_HPP
#define SAMPLING_HPP

#include <vector>
#include <random>
#include <unistd.h>
#include "entropy.hpp"
#include "log.hpp"
#include "types.hpp"

struct RandomDistribution {
  static std::uniform_real_distribution<double> Uniform;
  // static std::uniform_int_distribution<size_t> Binary;
  // static std::uniform_int_distribution<size_t> Nucleotide;
  // static std::uniform_real_distribution<double> Probability;
};

template <typename T>
T sample_compound_gamma(T a, T b, T c, std::mt19937 &rng) {
  // NOTE: gamma_distribution takes a shape and scale parameter
  return std::gamma_distribution<T>(
      a, 1/std::gamma_distribution<>(b, 1/c)(rng))(rng);
}

template <typename V, class Iter>
V sample_multinomial(size_t n, const Iter begin, const Iter end,
                                  std::mt19937 &rng = EntropySource::rng) {
  const size_t k = std::distance(begin, end);
  V x = V::Zero(k);
  if (n == 0)
    return x;
  Iter p = begin;
  double cum_prob = 0;
  for(size_t i = 0; i < k; ++i) {
    const double current_p = std::min(1.0, *p / (1 - cum_prob));
    n -= x[i] = std::binomial_distribution<size_t>(n, current_p)(rng);
    if (n == 0)
      break;
    cum_prob += *p;
    p++;
  }
  return x;
}

template <class T>
std::vector<T> sample_multinomial_slow(size_t n, const std::vector<double> &p,
                                       std::mt19937 &rng = EntropySource::rng) {
  // TODO could also use std::discrete_distribution
  // TODO re-order p in decreasing order
  const size_t k = p.size();
  std::vector<T> x(k, 0);
  for (size_t i = 0; i < n; ++i) {
    double u = RandomDistribution::Uniform(rng);
    double cumul = 0;
    size_t j = 0;
    auto iter = begin(p);
    while ((cumul += *iter++) < u)
      j++;
    x[j]++;
  }
  return x;
}

template <class T>
T sample_beta(double alpha, double beta,
              std::mt19937 &rng = EntropySource::rng) {
  T x = std::gamma_distribution<T>(alpha, 1)(rng);
  T y = std::gamma_distribution<T>(beta, 1)(rng);
  return x / (x + y);
}

template <class Y, class Iter>
std::vector<Y> sample_dirichlet(const Iter begin, const Iter end,
                                std::mt19937 &rng = EntropySource::rng) {
  const size_t K = std::distance(begin, end);
  std::vector<Y> p(K, 0);
  Iter iter = begin;
  Y sum = 0;
  for (auto &q : p) {
    LOG(debug) << "dirichlet sampling: " << *iter;
    sum += q = std::gamma_distribution<Y> (*iter, 1)(rng);
    iter++;
  }
  if (sum > 0)
    for (auto &q : p)
      q /= sum;
  else
    p[std::uniform_int_distribution<size_t>(0, K - 1)(rng)] = 1;
  return p;
};

size_t sample_poisson(double lambda, std::mt19937 &rng = EntropySource::rng);

size_t sample_negative_binomial(double r, double p,
                                std::mt19937 &rng = EntropySource::rng);

#endif
