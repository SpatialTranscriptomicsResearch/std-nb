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
    if (flagged(which & Target::global)) {
      os << (first ? "" : ",") << "global";
      first = false;
    }
    if (flagged(which & Target::local)) {
      os << (first ? "" : ",") << "local";
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
    else if (token == "global")
      which = which | Target::global;
    else if (token == "local")
      which = which | Target::local;
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
