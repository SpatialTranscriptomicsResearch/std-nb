#include "covariate.hpp"
#include "aux.hpp"
#include "compression.hpp"
#include "design.hpp"
#include "io.hpp"

using namespace std;
using STD::Matrix;

string Covariate::to_string() const {
  string str = "Covariate: '" + label + "':";
  for (auto &value : values)
    str += " '" + value + "'";
  return str;
}

ostream &operator<<(ostream &os, const Covariate &covariate) {
  os << covariate.to_string();
  return os;
}

string CovariateInformation::to_string(const Covariates &covariates) const {
  string s;
  for (size_t i = 0; i < idxs.size(); ++i) {
    if (i > 0)
      s += ",";
    if (covariates[idxs[i]].label == DesignNS::unit_label)
      s += "intercept";
    else
      s += covariates[idxs[i]].label + "="
           + covariates[idxs[i]].values[vals[i]];
  }
  if (idxs.size() == 0)
    s = "global";
  return s;
}

// TODO: maybe better implementing a hash function and using unordered_map/set
// wherever needed. This feels a bit odd.
bool CovariateInformation::operator<(const CovariateInformation& other) const {
  if (idxs != other.idxs) {
    return idxs < other.idxs;
  }
  return vals < other.vals;
}
