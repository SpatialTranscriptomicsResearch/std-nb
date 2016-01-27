#include "sampling.hpp"
#include <random>
#include <iostream>
#include <omp.h>

#define DO_PARALLEL 1
using namespace std;

mt19937 EntropySource::rng;

vector<mt19937> init_rngs() {
  vector<mt19937> rngs;
#pragma omp parallel if (DO_PARALLEL)
  {
#pragma omp single
    {
      for (int thread_num = 0; thread_num < omp_get_num_threads(); ++thread_num)
        rngs.push_back(mt19937());
      for (auto &rng : rngs) rng.seed(EntropySource::rng());
    }
  }
  return rngs;
}

uniform_real_distribution<double> RandomDistribution::Uniform(0, 1);
vector<mt19937> EntropySource::rngs = init_rngs();

size_t sample_poisson(double lambda, std::mt19937 &rng) {
  size_t k = 0;
  double logp = 0;
  do {
    k++;
    logp += log(RandomDistribution::Uniform(rng));
  } while (logp > -lambda);
  // cerr << "Poisson sample for lambda = " << lambda << " -> " << k-1 << endl;
  return k - 1;
};

size_t sample_negative_binomial(double r, double p, std::mt19937 &rng) {
  // NOTE: gamma_distribution takes a shape and scale parameter
  return sample_poisson(std::gamma_distribution<double>(r, p / (1 - p))(rng),
                        rng);
}
