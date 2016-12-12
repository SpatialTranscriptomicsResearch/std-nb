#include "target.hpp"
#include <boost/tokenizer.hpp>

using namespace std;

namespace PoissonFactorization {

ostream &operator<<(ostream &os, const Target &which) {
  if (which == Target::empty) {
    os << "empty";
    return os;
  } else {
    bool first = true;
    if (flagged(which & Target::contributions)) {
      os << "contributions";
      first = false;
    }
    if (flagged(which & Target::phi)) {
      os << (first ? "" : ",") << "phi";
      first = false;
    }
    if (flagged(which & Target::phi_prior)) {
      os << (first ? "" : ",") << "phi_prior";
      first = false;
    }
    if (flagged(which & Target::phi_local)) {
      os << (first ? "" : ",") << "phi_local";
      first = false;
    }
    if (flagged(which & Target::phi_prior_local)) {
      os << (first ? "" : ",") << "phi_prior_local";
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
    if (flagged(which & Target::baseline)) {
      os << (first ? "" : ",") << "baseline";
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
    if (token == "contributions")
      which = which | Target::contributions;
    else if (token == "features")
      which = which | Target::phi | Target::phi_prior;
    else if (token == "mixing")
      which = which | Target::theta | Target::theta_prior;
    else if (token == "phi")
      which = which | Target::phi;
    else if (token == "phi_prior")
      which = which | Target::phi_prior;
    else if (token == "phi_local")
      which = which | Target::phi_local;
    else if (token == "phi_prior_local")
      which = which | Target::phi_prior_local;
    else if (token == "theta")
      which = which | Target::theta;
    else if (token == "theta_prior")
      which = which | Target::theta_prior;
    else if (token == "spot")
      which = which | Target::spot;
    else if (token == "baseline")
      which = which | Target::baseline;
    else if (token == "field")
      which = which | Target::field;
    else
      throw(runtime_error("Unknown sampling token: " + token));
  }
  return is;
}
}
