#ifndef SAMPLING_HPP
#define SAMPLING_HPP

#include <vector>
#include <random>
#include <unistd.h>
#include "entropy.hpp"

struct RandomDistribution {
  static std::uniform_real_distribution<double> Uniform;
  // static std::uniform_int_distribution<size_t> Binary;
  // static std::uniform_int_distribution<size_t> Nucleotide;
  // static std::uniform_real_distribution<double> Probability;
};

template <class T>
std::vector<T> sample_multinomial(size_t n, const std::vector<double> &p, std::mt19937 &rng=EntropySource::rng) {
  const size_t K = p.size();
  std::vector<T> x(K, 0);
  double cum_prob = 0;
  for (size_t k = 0; k < K; ++k) {
    const double current_p = std::min(1.0, p[k] / (1 - cum_prob));
    n -= x[k] = std::binomial_distribution<T>(n, current_p)(rng);
    cum_prob += p[k];
    if(cum_prob >= 1)
      break;
  }
  return x;
}

template <class T>
std::vector<T> sample_multinomial_slow(size_t n, const std::vector<double> &p, std::mt19937 &rng=EntropySource::rng) {
  // TODO could also use std::discrete_distribution
  // TODO re-order p in decreasing order
  const size_t k = p.size();
  std::vector<T> x(k, 0);
  for (size_t i = 0; i < n; ++i) {
    double u = RandomDistribution::Uniform(rng);
    double cumul = 0;
    size_t j = 0;
    auto iter = begin(p);
    while ((cumul += *iter++) < u) j++;
    x[j]++;
  }
  return x;
}

template <class T>
T sample_beta(double alpha, double beta, std::mt19937 &rng=EntropySource::rng) {
  T x = std::gamma_distribution<T>(alpha, 1)(rng);
  T y = std::gamma_distribution<T>(beta, 1)(rng);
  return x / (x + y);
}

template <class Y, class X>
std::vector<Y> sample_dirichlet(const std::vector<X> &a, std::mt19937 &rng=EntropySource::rng) {
  std::vector<Y> p(a.size(), 0);
  auto iter = begin(a);
  Y sum = 0;
  for (auto &q : p)
    sum += q = std::gamma_distribution<Y>(*iter++, 1)(rng);
  for (auto &q : p) q /= sum;
  return p;
};

size_t sample_poisson(double lambda, std::mt19937 &rng=EntropySource::rng);

size_t sample_negative_binomial(double r, double p, std::mt19937 &rng=EntropySource::rng);

#endif
