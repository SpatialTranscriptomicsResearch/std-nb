#ifndef ENTROPY_HPP
#define ENTROPY_HPP

#include <random>
#include <vector>
#include <unistd.h>

struct EntropySource {
  static void seed(size_t new_seed = std::random_device()()) {
    rng.seed(new_seed);
  }
  static std::mt19937 rng;
  static std::vector<std::mt19937> rngs;
};

#endif
