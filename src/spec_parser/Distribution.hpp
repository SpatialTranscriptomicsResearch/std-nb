#ifndef DISTRIBUTION_HPP
#define DISTRIBUTION_HPP

#include <unordered_map>

#include "../coefficient.hpp"

struct Distribution {
  // TODO: consider using separate Distribution type to increase modularity
  using Type = Coefficient::Distribution;

  static Type stodistr(const std::string& s)
  {
    static const std::unordered_map<std::string, Type> map{
      { "Beta'", Type::beta_prime }, { "Betaprime", Type::beta_prime },
      { "Gamma", Type::gamma }, { "Loggp", Type::log_gp },
      { "Lognormal", Type::log_normal }, { "logN", Type::log_normal },
    };
    auto it = map.find(s);
    if (it == map.end()) {
      throw std::invalid_argument("Invalid distribution '" + s + "'.");
    }
    return it->second;
  }
  static std::string distrtos(const Type& d)
  {
    switch (d) {
    case Type::beta_prime:
      return "Betaprime";
    case Type::fixed:
      return "Fixed";
    case Type::gamma:
      return "Gamma";
    case Type::log_gp:
      return "Loggp";
    case Type::log_normal:
      return "Lognormal";
    }
  }

  Type type;
  std::vector<std::string> arguments;

  // TODO: Specify default distribution somewhere
  Distribution()
      : type(Type::fixed)
      , arguments{ "0" }
  {
  }
  Distribution(std::string _type, const std::vector<std::string> _arguments)
      : type(stodistr(_type))
      , arguments(_arguments)
  {
  }
};

#endif // DISTRIBUTION_HPP
