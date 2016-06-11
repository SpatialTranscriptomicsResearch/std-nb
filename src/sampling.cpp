#include "sampling.hpp"
#include <random>
#include <iostream>

using namespace std;

uniform_real_distribution<double> RandomDistribution::Uniform(0, 1);

// TODO: this is very slow for large lambda
size_t sample_poisson(double lambda, std::mt19937 &rng) {
  size_t k = 0;
  double logp = 0;
  do {
    k++;
    logp += log(RandomDistribution::Uniform(rng));
  } while (logp > -lambda);
  return k - 1;
};

size_t sample_negative_binomial(double r, double p, std::mt19937 &rng) {
  // NOTE: gamma_distribution takes a shape and scale parameter
  return sample_poisson(std::gamma_distribution<double>(r, p / (1 - p))(rng),
                        rng);
}
