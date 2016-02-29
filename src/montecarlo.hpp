/* =====================================================================================
 * Copyright (c) 2011, Jonas Maaskola
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * =====================================================================================
 *
 *       Filename:  montecarlo.hpp
 *
 *    Description:  Code for MCMC sampling, including parallel tempering
 *
 *        Created:  Wed Aug 3 02:08:55 2011 +0200
 *
 *         Author:  Jonas Maaskola <jonas@maaskola.de>
 *
 * =====================================================================================
 */

#ifndef MONTECARLO_HPP
#define MONTECARLO_HPP

#include <cstdlib>
#include <iostream>
#include <list>
#include <vector>
#include <cmath>
#include "verbosity.hpp"
#include "entropy.hpp"
#include "sampling.hpp"
// #include "../random_distributions.hpp"

namespace MCMC {
inline double boltzdist(double dG, double T) { return exp(-dG / T); };

template <class T>
class Evaluator {
public:
  double evaluate(T &i) const;
};

template <class T>
class Generator {
public:
  T generate(const T &i) const;
};

template <class T>
class MonteCarlo {
  using E = std::pair<T, double>;
  Verbosity verbosity;

public:
  MonteCarlo(Verbosity ver)
      : verbosity(ver), generator(Generator<T>()), evaluator(Evaluator<T>()){};
  MonteCarlo(const Generator<T> &gen, const Evaluator<T> &eval, Verbosity ver)
      : verbosity(ver), generator(gen), evaluator(eval){};
  ~MonteCarlo(){};

  Generator<T> generator;
  Evaluator<T> evaluator;

  friend Generator<T>;
  friend Evaluator<T>;

private:
  bool GibbsStep(double temp, T &state, double &G) const {
    T nextstate = generator.generate(state);
    double nextG = evaluator.evaluate(nextstate);
    double dG = nextG - G;
    double r = RandomDistribution::Uniform(EntropySource::rng);
    double p = std::min<double>(1.0, boltzdist(-dG, temp));
    if (verbosity >= Verbosity::Verbose)
      std::cerr << "T = " << temp << " next = " << nextstate << std::endl
                << "nextG = " << nextG << " G = " << G << " dG = " << dG
                << " p = " << p << " r = " << r << std::endl;
    if (std::isnan(nextG) == 0 and (dG > 0 or r <= p)) {
      if (verbosity >= Verbosity::Verbose)
        std::cerr << "Accepted!" << std::endl;
      state = nextstate;
      G = nextG;
      return true;
    } else {
      if (verbosity >= Verbosity::Verbose)
        std::cerr << "Rejected!" << std::endl;
      return false;
    }
  }

  bool swap(double temp1, double temp2, T &state1, T &state2, double &G1,
            double &G2) const {
    double r = RandomDistribution::Uniform(EntropySource::rng);
    double p = std::min<double>(
        1.0, exp(-(G1 / temp1 + G2 / temp2 - G1 / temp2 - G2 / temp1)));
    if (verbosity >= Verbosity::Verbose)
      std::cerr << "T1 = " << temp1 << " T2 " << temp2 << " G1 = " << G1
                << " G2 = " << G2 << std::endl << "r = " << r << " p = " << p
                << std::endl;
    if (r <= p) {
      if (verbosity >= Verbosity::Verbose) std::cerr << "Swap!" << std::endl;
      std::swap<T>(state1, state2);
      std::swap<double>(G1, G2);
      return true;
    } else {
      if (verbosity >= Verbosity::Verbose)
        std::cerr << "Swap rejected!" << std::endl;
      return false;
    }
  }

public:
  std::list<E> run(double temp, double anneal, const T &init,
                   size_t steps) const {
    T state = T(init);
    double G = evaluator.evaluate(state);
    std::list<E> trajectory;
    // trajectory.push_back(E(state, G));
    for (size_t i = 0; i < steps; i++) {
      if (verbosity >= Verbosity::Info)
        std::cerr << std::endl << "Iteration " << i << " of " << steps
                  << std::endl;
      GibbsStep(temp, state, G);
      // if (GibbsStep(temp, state, G))
        // trajectory.push_back(E(state, G));
        // std::cout << "Not storing temp result!" << std::endl;
      temp *= anneal;
    }
    trajectory.push_back(E(state, G));
    return trajectory;
  };

  std::vector<std::list<E>> parallel_tempering(const std::vector<double> &temp,
                                               const std::vector<T> &init,
                                               size_t steps) const {
    size_t n = temp.size();
    std::uniform_int_distribution<size_t> r_unif(0, n - 2);
    std::vector<T> state = init;

    std::vector<double> G;
    for (auto s : state) G.push_back(evaluator.evaluate(s));

    std::vector<std::list<E>> trajectory(temp.size());
    for (size_t t = 0; t < temp.size(); t++)
      trajectory[t].push_back(E(state[t], G[t]));

    for (size_t i = 0; i < steps; i++) {
      if (verbosity >= Verbosity::Info)
        std::cerr << "Iteration " << i << " of " << steps << std::endl;
      for (size_t t = 0; t < n; t++)
        // TODO: if one wants to determine means one should respect the failed
        // changes, and input once more the original state to the trajectory.
        if (GibbsStep(temp[t], state[t], G[t]))
          trajectory[t].push_back(E(state[t], G[t]));

      if (verbosity >= Verbosity::Info) {
        std::cout << "Scores =";
        for (size_t t = 0; t < n; t++) std::cout << " " << G[t];
        std::cout << std::endl;
      }

      if (temp.size() > 1) {
        size_t r = r_unif(EntropySource::rng);
        if (verbosity >= Verbosity::Verbose)
          std::cerr << "Testing swap of " << r << " and " << r + 1 << std::endl;
        if (swap(temp[r], temp[r + 1], state[r], state[r + 1], G[r],
                 G[r + 1])) {
          trajectory[r].push_back(E(state[r], G[r]));
          trajectory[r + 1].push_back(E(state[r + 1], G[r + 1]));
          if (verbosity >= Verbosity::Info)
            std::cerr << "Swapping chains " << r << " and " << r + 1 << "."
                      << std::endl;
        }
      }
    }
    return trajectory;
  };
};
}

#endif
