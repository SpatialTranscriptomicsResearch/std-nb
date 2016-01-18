#include "sampling.hpp"
#include <random>
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
