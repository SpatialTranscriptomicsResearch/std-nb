#include "target.hpp"
#include <boost/tokenizer.hpp>
#include "aux.hpp"

using namespace std;

namespace STD {

ostream &operator<<(ostream &os, const Target &which) {
  if (which == Target::empty) {
    os << "empty";
    return os;
  } else {
    bool first = true;
    if (flagged(which & Target::covariates)) {
      os << (first ? "" : ",") << "covariates";
      first = false;
    }
    if (flagged(which & Target::gamma_prior)) {
      os << (first ? "" : ",") << "gamma_prior";
      first = false;
    }
    if (flagged(which & Target::rho)) {
      os << (first ? "" : ",") << "rho";
      first = false;
    }
    if (flagged(which & Target::rho_prior)) {
      os << (first ? "" : ",") << "rho_prior";
      first = false;
    }
    if (flagged(which & Target::theta)) {
      os << (first ? "" : ",") << "theta";
      first = false;
    }
    if (flagged(which & Target::theta_prior)) {
      os << (first ? "" : ",") << "theta_prior";
      first = false;
    }
    if (flagged(which & Target::spot)) {
      os << (first ? "" : ",") << "spot";
      first = false;
    }
    if (flagged(which & Target::field)) {
      os << (first ? "" : ",") << "field";
      first = false;
    }
  }
  return os;
}

istream &operator>>(istream &is, Target &which) {
  which = Target::empty;
  using tokenizer = boost::tokenizer<boost::char_separator<char>>;
  boost::char_separator<char> sep(",");

  string line;
  getline(is, line);
  tokenizer tok(line, sep);
  for (auto token : tok) {
    token = to_lower(token);
    if (token == "covariates")
      which = which | Target::covariates;
    else if (token == "rho")
      which = which | Target::rho;
    else if (token == "rho_prior")
      which = which | Target::rho_prior;
    else if (token == "theta")
      which = which | Target::theta;
    else if (token == "theta_prior")
      which = which | Target::theta_prior;
    else if (token == "spot")
      which = which | Target::spot;
    else if (token == "field")
      which = which | Target::field;
    else if (token == "gamma_prior")
      which = which | Target::gamma_prior;
    else
      throw(runtime_error("Unknown sampling token: " + token));
  }
  return is;
}
}
