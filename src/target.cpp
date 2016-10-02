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
    if (flagged(which & Target::phi_r)) {
      os << (first ? "" : ",") << "phi_r";
      first = false;
    }
    if (flagged(which & Target::phi_p)) {
      os << (first ? "" : ",") << "phi_p";
      first = false;
    }
    if (flagged(which & Target::theta)) {
      os << (first ? "" : ",") << "theta";
      first = false;
    }
    if (flagged(which & Target::theta_p)) {
      os << (first ? "" : ",") << "theta_p";
      first = false;
    }
    if (flagged(which & Target::theta_r)) {
      os << (first ? "" : ",") << "theta_r";
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
      which = which | Target::phi | Target::phi_r | Target::phi_p;
    else if (token == "mixing")
      which = which | Target::theta | Target::theta_r | Target::theta_p;
    else if (token == "phi")
      which = which | Target::phi;
    else if (token == "phi_r")
      which = which | Target::phi_r;
    else if (token == "phi_p")
      which = which | Target::phi_p;
    else if (token == "theta")
      which = which | Target::theta;
    else if (token == "theta_r")
      which = which | Target::theta_r;
    else if (token == "theta_p")
      which = which | Target::theta_p;
    else if (token == "spot")
      which = which | Target::spot;
    else if (token == "baseline")
      which = which | Target::baseline;
    else
      throw(runtime_error("Unknown sampling token: " + token));
  }
  return is;
}
}
