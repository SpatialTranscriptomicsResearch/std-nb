#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/trigamma.hpp>
#include "FactorAnalysis.hpp"
#include "aux.hpp"

using namespace std;

namespace FactorAnalysis {
Float digamma(Float x) { return boost::math::digamma(x); }

Float trigamma(Float x) { return boost::math::trigamma(x); }

istream &operator>>(istream &is, Parameters::ForceMean &force) {
  // TODO accept comma separated token lists
  string token;
  is >> token;
  token = to_lower(token);
  if (token == "theta")
    force |= Parameters::ForceMean::Theta;
  else if (token == "phi")
    force |= Parameters::ForceMean::Phi;
  else if (token == "spot")
    force |= Parameters::ForceMean::Spot;
  else if (token == "experiment")
    force |= Parameters::ForceMean::Experiment;
  else if (token == "default")
    force |= Parameters::ForceMean::Theta | Parameters::ForceMean::Phi |
             Parameters::ForceMean::Spot | Parameters::ForceMean::Experiment;
  else
    throw runtime_error("Error: could not parse mean forcing options'" +
                             token + "'.");
  return is;
}

ostream &operator<<(ostream &os, const Parameters::ForceMean &force) {
  if (force == Parameters::ForceMean::None)
    os << "None";
  else {
    bool first;
    if ((force & Parameters::ForceMean::Theta) != Parameters::ForceMean::None) {
      os << (first ? "" : "|") << "Theta";
      first = false;
    }
    if ((force & Parameters::ForceMean::Phi) != Parameters::ForceMean::None) {
      os << (first ? "" : "|") << "Phi";
      first = false;
    }
    if ((force & Parameters::ForceMean::Spot) != Parameters::ForceMean::None) {
      os << (first ? "" : "|") << "Spot";
      first = false;
    }
    if ((force & Parameters::ForceMean::Experiment) !=
        Parameters::ForceMean::None) {
      os << (first ? "" : "|") << "Experiment";
      first = false;
    }
  }
  return os;
}
}
