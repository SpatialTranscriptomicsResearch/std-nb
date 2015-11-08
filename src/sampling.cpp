#include "sampling.hpp"
#include <random>

using namespace std;

uniform_real_distribution<double> RandomDistribution::Uniform(0, 1);
mt19937 EntropySource::rng;

// TODO could also use std::discrete_distribution
vector<size_t> sample_multinomial(size_t n, const vector<double> &p) {
  // TODO re-order p in decreasing order
  const size_t k = p.size();
  vector<size_t> x(k, 0);
  for (size_t i = 0; i < n; ++i) {
    double u = RandomDistribution::Uniform(EntropySource::rng);
    double cumul = 0;
    size_t j = 0;
    auto iter = begin(p);
    while ((cumul += *iter++) < u) j++;
    x[j]++;
  }
  return x;
}

double sample_beta(double alpha, double beta) {
  double x = std::gamma_distribution<double>(alpha, 1)(EntropySource::rng);
  double y = std::gamma_distribution<double>(beta, 1)(EntropySource::rng);
  return x / (x+y);
}
