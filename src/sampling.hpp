#ifndef SAMPLING_HPP
#define SAMPLING_HPP

#include <vector>
#include <random>
#include <unistd.h>

struct RandomDistribution {
  static std::uniform_real_distribution<double> Uniform;
  // static std::uniform_int_distribution<size_t> Binary;
  // static std::uniform_int_distribution<size_t> Nucleotide;
  // static std::uniform_real_distribution<double> Probability;
};

struct EntropySource {
  static void seed(size_t new_seed = std::random_device()()) {
    rng.seed(new_seed);
  }
  static std::mt19937 rng;
  static std::vector<std::mt19937> rngs;
};

template <class T>
std::vector<T> sample_multinomial(size_t n, const std::vector<double> &p, std::mt19937 &rng=EntropySource::rng) {
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

#endif
