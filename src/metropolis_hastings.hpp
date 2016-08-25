#ifndef METROPOLIS_HASTINGS_HPP
#define METROPOLIS_HASTINGS_HPP

#include <iostream>
#include <random>
#include "log.hpp"
#include "sampling.hpp"

struct MetropolisHastings {
  inline static double boltzdist(double dG, double T) { return exp(-dG / T); };

  double temperature;
  double prop_sd;
  // std::normal_distribution<double> rnorm;

  MetropolisHastings(double temp, double prop_sd_);

  template <typename T, typename RNG, typename Gen, typename Score,
            typename... Args>
  T sample(T current, size_t n_iter_initial, RNG &rng, Gen generate, Score fnc,
           Args &... args) const {
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
        LOG(debug) << "Improved!";
        accept = true;
      } else {
        const auto dG = propsition_score - current_score;
        const double rnd = RandomDistribution::Uniform(rng);
        const double prob = std::min<double>(1.0, boltzdist(-dG, temperature));
        if (std::isnan(propsition_score) == 0 and (dG > 0 or rnd <= prob)) {
          accept = true;
          LOG(debug) << "Accepted!";
        } else {
          LOG(debug) << "Rejected!";
        }
      }
      if (accept) {
        accepted = proposition;
        break;
      }
    }
    LOG(debug) << "Performed  " << (n_iter_initial - n_iter)
               << " MCMC sampling iterations.";
    return accepted;
  }
};

#endif
