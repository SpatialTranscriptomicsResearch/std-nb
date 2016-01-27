#ifndef METROPOLIS_HASTINGS_HPP
#define METROPOLIS_HASTINGS_HPP

#include <random>
#include <iostream>
#include "verbosity.hpp"
#include "sampling.hpp"

struct MetropolisHastings {
  inline double boltzdist(double dG, double T) const { return exp(-dG / T); };

  double temperature;
  double prop_sd;
  Verbosity verbosity;
  // std::normal_distribution<double> rnorm;

  MetropolisHastings(double temp, double prop_sd_, Verbosity verb);

  template <typename T, typename RNG, typename Gen, typename Score,
            typename... Args>
  T sample(T current, size_t n_iter_initial, RNG &rng, Gen generate,
           Score fnc, Args &... args) const {
    const auto current_score = fnc(current, args...);
    size_t n_iter = n_iter_initial;
    T accepted = current;
    bool accept = false;
    while (n_iter--) {
      // const double f = exp(rnorm(rng));
      // const T proposition = current * f;
      const T proposition = generate(current, rng);
      const auto propsition_score = fnc(proposition, args...);

      if (propsition_score > current_score) {
        if (verbosity >= Verbosity::Debug)
          std::cout << "Improved!" << std::endl;
        accept = true;
      } else {
        const auto dG = propsition_score - current_score;
        const double rnd = RandomDistribution::Uniform(rng);
        const double prob = std::min<double>(1.0, boltzdist(-dG, temperature));
        if (std::isnan(propsition_score) == 0 and (dG > 0 or rnd <= prob)) {
          accept = true;
          if (verbosity >= Verbosity::Debug)
            std::cout << "Accepted!" << std::endl;
        } else {
          if (verbosity >= Verbosity::Debug)
            std::cout << "Rejected!" << std::endl;
        }
      }
      if (accept) {
        accepted = proposition;
        break;
      }
    }
    if (verbosity >= Verbosity::Debug)
      std::cout << "Left MCMC " << (accept ? "" : "un") << "successfully after "
                << (n_iter_initial - n_iter) << " iterations." << std::endl;
    return accepted;
  }
};

#endif
