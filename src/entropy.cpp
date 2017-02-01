#include "entropy.hpp"
#include "parallel.hpp"

using namespace std;

mt19937 EntropySource::rng;

vector<mt19937> init_rngs() {
  vector<mt19937> rngs;
#pragma omp parallel
  {
#pragma omp single
    {
      for (int thread_num = 0; thread_num < omp_get_num_threads(); ++thread_num) {
        // LOG(verbose) << "Allocating RNG for thread #" << thread_num;
        rngs.push_back(mt19937());
      }
      for (auto &rng : rngs) {
        // LOG(verbose) << "Initializing RNG for thread #" << thread_num;
        rng.seed(EntropySource::rng());
      }
    }
  }
  return rngs;
}

vector<mt19937> EntropySource::rngs = init_rngs();
