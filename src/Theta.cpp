#include "Theta.hpp"
#include "aux.hpp"
#include "odds.hpp"

using namespace std;

namespace STD {
Theta::Theta(size_t S_, size_t T_, const Parameters &params,
             const prior_type &prior)
    : S(S_), T(T_), matrix(S, T), parameters(params) {
  LOG(debug) << "Initializing Θ from Gamma distribution";
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t s = 0; s < S; ++s) {
    const size_t thread_num = omp_get_thread_num();
    for (size_t t = 0; t < T; ++t)
      // NOTE: std::gamma_distribution takes a shape and scale parameter
      matrix(s, t) = std::gamma_distribution<Float>(
          prior.r(t), 1 / prior.p(t))(EntropySource::rngs[thread_num]);
  }
}
}
