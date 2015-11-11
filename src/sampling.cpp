#include "sampling.hpp"
#include <random>

using namespace std;

uniform_real_distribution<double> RandomDistribution::Uniform(0, 1);
mt19937 EntropySource::rng;
