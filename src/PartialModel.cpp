#include "PartialModel.hpp"
#include "aux.hpp"
#include "odds.hpp"

using namespace std;

namespace PoissonFactorization {
namespace Partial {

string to_string(Variable variable) {
  switch (variable) {
    case Variable::Feature:
      return "Feature";
      break;
    case Variable::Mix:
      return "Mix";
      break;
    case Variable::Spot:
      return "Spot";
      break;
    case Variable::Experiment:
      return "Experiment";
      break;
    default:
      throw std::logic_error("Implementation of to_string(Kind) incomplete!");
      break;
  }
}

string to_string(Kind kind) {
  switch (kind) {
    case Kind::Constant:
      return "Constant";
      break;
    case Kind::Dirichlet:
      return "Dirichlet";
      break;
    case Kind::Gamma:
      return "Gamma";
      break;
    case Kind::HierGamma:
      return "HierGamma";
      break;
    default:
      throw std::logic_error("Implementation of to_string(Kind) incomplete!");
      break;
  }
}

std::ostream &operator<<(std::ostream &os, Variable variable) {
  os << to_string(variable);
  return os;
}

std::ostream &operator<<(std::ostream &os, Kind kind) {
  os << to_string(kind);
  return os;
}

std::istream &operator>>(std::istream &is, Variable &variable) {
  string token;
  is >> token;
  token = to_lower(token);
  if (token == "feature")
    variable = Variable::Feature;
  else if (token == "mix")
    variable = Variable::Mix;
  else if (token == "spot")
    variable = Variable::Spot;
  else if (token == "experiment")
    variable = Variable::Experiment;
  else
    throw(std::runtime_error("Cannot parse variable type '" + token + "'."));
  return is;
}

std::istream &operator>>(std::istream &is, Kind &kind) {
  string token;
  is >> token;
  token = to_lower(token);
  if (token == "constant")
    kind = Kind::Constant;
  else if (token == "dirichlet")
    kind = Kind::Dirichlet;
  else if (token == "gamma")
    kind = Kind::Gamma;
  else if (token == "hiergamma")
    kind = Kind::HierGamma;
  else
    throw(
        std::runtime_error("Cannot parse distribution type '" + token + "'."));
  return is;
}
}
}
