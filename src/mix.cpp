#include "mix.hpp"

using namespace std;

namespace PoissonFactorization {
namespace Mix {

std::ostream &operator<<(std::ostream &os, Kind kind) {
  switch (kind) {
    case Kind::Dirichlet:
      os << "Dirichlet";
      break;
    case Kind::Gamma:
      os << "Gamma";
      break;
  }
  return os;
}

std::istream &operator>>(std::istream &is, Kind &kind) {
  string token;
  is >> token;
  if (token == "dirichlet")
    kind = Kind::Dirichlet;
  else if (token == "gamma")
    kind = Kind::Gamma;
  else
    throw(std::runtime_error("Cannot parse mixing distribution type '" + token
                             + "'."));
  return is;
}
}
}
