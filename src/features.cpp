#include "features.hpp"

using namespace std;

namespace PoissonFactorization {
namespace Feature {

ostream &operator<<(ostream &os, Kind kind) {
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

istream &operator>>(istream &is, Kind &kind) {
  string token;
  is >> token;
  if (token == "dirichlet")
    kind = Kind::Dirichlet;
  else if (token == "gamma")
    kind = Kind::Gamma;
  else
    throw(runtime_error("Cannot parse mixing distribution type '" + token
                             + "'."));
  return is;
}
}
}
