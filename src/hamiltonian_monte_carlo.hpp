#ifndef HAMILTONIAN_MONTE_CARLO_HPP
#define HAMILTONIAN_MONTE_CARLO_HPP

#include <iostream>
#include <random>
#include "log.hpp"
#include "sampling.hpp"

namespace HMC {
const bool hmc_noisy = false;
template <typename T, typename F, typename G, typename RNG, typename... Args>
T sample(T current_q, F func, G grad, size_t L, double epsilon, RNG &rng,
         Args &... args) {
  const size_t n = current_q.size();
  T q = current_q;
  T p(n);
  for (auto &x : p)
    x = std::normal_distribution<double>()(rng);
  T current_p = p;

  // make a half step for momentum at the beginning
  p = p - epsilon * grad(q, args...) / 2;

  // alternate full steps for position and momentum
  for (size_t l = 0; l < L; ++l) {
    // make a full step for the position
    q = q + epsilon * p;
    // make a full step for the momentum, except at end of trajectory
    if (l != L - 1)
      p = p - epsilon * grad(q, args...);
  }

  // make a half step for momentum at the end
  p = p - epsilon * grad(q, args...) / 2;

  // Negate momentum at end of trajectory to make the proposal symmetric
  p = -p;

  // Evaluate potential and kinetic energies at start and end of trajectory
  auto current_U = func(current_q, args...);
  auto current_K = sum(current_p % current_p) / 2;
  auto proposed_U = func(q, args...);
  auto proposed_K = sum(p % p) / 2;

  double score = current_U - proposed_U + current_K - proposed_K;

  // Accept or reject the state at end of trajectory, returning either
  // the position at the end of the trajectory or the initial position
  if (score > 0 or RandomDistribution::Uniform(rng) < exp(score)) {
    if (hmc_noisy)
      LOG(debug) << "HMC: " << score << " accepted";
    return q;  // accept
  } else {
    if (hmc_noisy)
      LOG(debug) << "HMC: " << score << " rejected";
    return current_q;  // reject
  }
};
};

#endif
