#include "covariate.hpp"
#include "aux.hpp"

using namespace std;

string Covariate::to_string() const {
  string str = "Covariate: '" + label + "':";
  for (auto &value : values)
    str += " '" + value + "'";
  return str;
}

std::ostream &operator<<(std::ostream &os, const Covariate &covariate) {
  os << covariate.to_string();
  return os;
}
