#ifndef ENTROPY_HPP
#define ENTROPY_HPP

#include <random>
#include <vector>
#include <unistd.h>

using RNG = std::mt19937;

struct EntropySource {
  static void seed(size_t new_seed = std::random_device()()) {
    rng.seed(new_seed);
  }
  static RNG rng;
  static std::vector<RNG> rngs;
};

#endif
