#ifndef METROPOLIS_HASTINGS_HPP
#define METROPOLIS_HASTINGS_HPP

#include <iostream>
#include <random>
#include "log.hpp"
#include "sampling.hpp"

struct MetropolisHastings {
  inline static double boltzdist(double dG, double T) { return exp(dG / T); };

  double temperature;

  MetropolisHastings(double temp);

  template <typename T, typename RNG, typename Gen, typename Score,
            typename... Args>
  T sample(T current, size_t n_iter_initial, RNG &rng, Gen generate, Score fnc,
           Args &... args) const {
    auto current_score = fnc(current, args...);
    size_t n_iter = n_iter_initial;
    while (n_iter--) {
      const T proposition = generate(current, rng);
      const auto proposition_score = fnc(proposition, args...);

      bool accept = false;
      if (proposition_score > current_score) {
        LOG(debug) << "Improved!";
        accept = true;
      } else {
        const auto dG = proposition_score - current_score;
        const double rnd = RandomDistribution::Uniform(rng);
        const double prob = boltzdist(dG, temperature);
        if (std::isnan(proposition_score) == 0 and rnd <= prob) {
          accept = true;
          LOG(debug) << "Accepted!";
        } else {
          LOG(debug) << "Rejected!";
        }
      }
      if (accept) {
        current = proposition;
        current_score = proposition_score;
      }
    }
    LOG(debug) << "Performed  " << (n_iter_initial - n_iter)
               << " MCMC sampling iterations.";
    return current;
  }
};

#endif
