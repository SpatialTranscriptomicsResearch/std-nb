#ifndef RANDOMVARIABLE_HPP
#define RANDOMVARIABLE_HPP

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../coefficient.hpp"

struct RandomVariable {
  // TODO: consider using separate Distribution type to increase modularity
  using Distribution = Coefficient::Distribution;

  static Distribution stodistr(const std::string& s)
  {
    static const std::unordered_map<std::string, Distribution> map{
      { "Beta'", Distribution::beta_prime },
      { "Betaprime", Distribution::beta_prime },
      { "Gamma", Distribution::gamma },
      { "Loggp", Distribution::log_gp },
      { "Lognormal", Distribution::log_normal },
      { "logN", Distribution::log_normal },
    };
    auto it = map.find(s);
    if (it == map.end()) {
      throw std::invalid_argument("Invalid distribution '" + s + "'.");
    }
    return it->second;
  }

  Distribution distribution;
  std::vector<std::string> arguments;
  std::unordered_set<std::string> covariates;

  RandomVariable(std::string distribution, std::unordered_set<std::string> covariates,
      std::vector<std::string> arguments)
      : distribution(stodistr(distribution))
      , arguments(arguments)
      , covariates(covariates)
  {
  }
};

#endif // RANDOMVARIABLE_HPP
