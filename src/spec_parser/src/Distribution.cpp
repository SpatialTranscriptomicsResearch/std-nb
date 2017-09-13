#include "spec_parser/Distribution.hpp"

using namespace spec_parser;

const Distribution::Type default_dist = Distribution::Type::fixed;
const std::vector<std::string> default_args = { "0" };

Distribution::Type Distribution::stodistr(const std::string& s)
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
std::string Distribution::distrtos(const Type& d)
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
    default:
      throw std::logic_error("Not implemented.");
  }
}

Distribution::Distribution()
    : type(default_dist)
    , arguments(default_args)
{
}
Distribution::Distribution(
    const std::string& _type, const std::vector<std::string> _arguments)
    : type(stodistr(_type))
    , arguments(_arguments)
{
}
