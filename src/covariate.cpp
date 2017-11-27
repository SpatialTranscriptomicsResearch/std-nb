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
    if (covariates[idxs[i]].label == Design::unit_label)
      s += "intercept";
    else
      s += covariates[idxs[i]].label + "="
           + covariates[idxs[i]].values[vals[i]];
  }
  if (idxs.size() == 0)
    s = "global";
  return s;
}

bool CovariateInformation::operator==(const CovariateInformation &other) const {
  if (idxs.size() != other.idxs.size() or vals.size() != other.vals.size())
    return false;
  for (size_t i = 0; i < idxs.size(); ++i)
    if (idxs[i] != other.idxs[i])
      return false;
  for (size_t i = 0; i < vals.size(); ++i)
    if (vals[i] != other.vals[i])
      return false;
  return true;
}

bool CovariateInformation::operator<(const CovariateInformation &other) const {
  return std::lexicographical_compare(begin(idxs), end(idxs), begin(other.idxs),
                                      end(other.idxs))
         or std::lexicographical_compare(begin(vals), end(vals),
                                         begin(other.vals), end(other.vals));
}
