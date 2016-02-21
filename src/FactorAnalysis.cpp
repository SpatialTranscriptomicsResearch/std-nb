#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/trigamma.hpp>
#include "FactorAnalysis.hpp"
#include "aux.hpp"

using namespace std;

namespace FactorAnalysis {
Float digamma(Float x) { return boost::math::digamma(x); }

Float trigamma(Float x) { return boost::math::trigamma(x); }

istream &operator>>(istream &is, ForceMean &force) {
  // TODO accept comma separated token lists
  string token;
  is >> token;
  token = to_lower(token);
  if (token == "theta")
    force |= ForceMean::Theta;
  else if (token == "phi")
    force |= ForceMean::Phi;
  else if (token == "spot")
    force |= ForceMean::Spot;
  else if (token == "experiment")
    force |= ForceMean::Experiment;
  else if (token == "default")
    force |= ForceMean::Theta | ForceMean::Phi | ForceMean::Spot |
             ForceMean::Experiment;
  else
    throw runtime_error("Error: could not parse mean forcing options'" + token +
                        "'.");
  return is;
}

ostream &operator<<(ostream &os, const ForceMean &force) {
  if (force == ForceMean::None)
    os << "None";
  else {
    bool first = true;
    if ((force & ForceMean::Theta) != ForceMean::None) {
      os << (first ? "" : ",") << "Theta";
      first = false;
    }
    if ((force & ForceMean::Phi) != ForceMean::None) {
      os << (first ? "" : ",") << "Phi";
      first = false;
    }
    if ((force & ForceMean::Spot) != ForceMean::None) {
      os << (first ? "" : ",") << "Spot";
      first = false;
    }
    if ((force & ForceMean::Experiment) != ForceMean::None) {
      os << (first ? "" : ",") << "Experiment";
      first = false;
    }
  }
  return os;
}
}
