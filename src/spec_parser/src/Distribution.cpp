#include "spec_parser/Distribution.hpp"

#include "aux.hpp"

using namespace spec_parser;

const Distribution::Type default_dist = Distribution::Type::fixed;
const std::vector<std::string> default_args = {"0"};

Distribution::Type Distribution::stodistr(const std::string& s) {
  static const std::unordered_map<std::string, Type> map{
      {"Beta", Type::beta},
      {"Beta'", Type::beta_prime},
      {"Betaprime", Type::beta_prime},
      {"Gamma", Type::gamma},
      {"GP", Type::gp_points},
      {"Normal", Type::normal}};
  auto it = map.find(s);
  if (it == map.end()) {
    throw std::invalid_argument("Invalid distribution '" + s + "'.");
  }
  return it->second;
}

std::string Distribution::distrtos(const Type& d) {
  switch (d) {
    case Type::beta:
      return "Beta";
    case Type::beta_prime:
      return "Beta'";
    case Type::fixed:
      return "Fixed";
    case Type::gamma:
      return "Gamma";
    case Type::gp_points:
      return "GP";
    case Type::normal:
      return "Normal";
    default:
      throw std::logic_error("Not implemented.");
  }
}

Distribution::Distribution() : type(default_dist), arguments(default_args) {}
Distribution::Distribution(Type _type,
                           const std::vector<std::string> _arguments)
    : type(_type), arguments(_arguments) {
  if (not(arguments.size() == desired_argument_number()))
    throw std::runtime_error("Error: wrong number of arguments for "
                             + to_string(*this));
}
Distribution::Distribution(const std::string& _type,
                           const std::vector<std::string> _arguments)
    : Distribution(stodistr(_type), _arguments) {}

size_t Distribution::desired_argument_number() {
  switch (type) {
    case Type::beta:
    case Type::beta_prime:
    case Type::gamma:
    case Type::normal:
      return 2;
    case Type::fixed:
      return 0;
    case Type::gp_points:
      return 4;
    default:
      throw std::logic_error("Not implemented.");
  }
}

std::string spec_parser::to_string(const Distribution& d) {
  auto distr_name = Distribution::distrtos(d.type);
  auto args
      = intercalate<std::vector<std::string>::const_iterator, std::string>(
          d.arguments.begin(), d.arguments.end(), ",");
  return distr_name + "(" + args + ")";
}
