#ifndef SAMPLING_HPP
#define SAMPLING_HPP

#include <vector>
#include <random>
#include <unistd.h>

std::vector<size_t> sample_multinomial(size_t n, const std::vector<double> &p);
double sample_beta(double alpha, double beta);

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
};

template <class X>
std::vector<double> sample_dirichlet(const std::vector<X> &a) {
  std::vector<double> p(a.size(), 0);
  auto iter = begin(a);
  double sum = 0;
  for (auto &q : p)
    sum += q = std::gamma_distribution<double>(*iter++, 1)(EntropySource::rng);
  for (auto &q : p) q /= sum;
  return p;
};

#endif
