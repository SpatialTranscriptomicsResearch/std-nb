#include "aux.hpp"
#include "optimization_method.hpp"

using namespace std;

namespace Optimize {

ostream &operator<<(ostream &os, const Method &method) {
  switch (method) {
    case Method::RPROP:
      os << "RPROP";
      break;
    case Method::Gradient:
      os << "Gradient";
      break;
    case Method::lBFGS:
      os << "lBFGS";
      break;
  }
  return os;
}

istream &operator>>(istream &is, Method &method) {
  string line;
  getline(is, line);
  line = to_lower(line);
  if (line == "gradient")
    method = Method::Gradient;
  else if (line == "rprop")
    method = Method::RPROP;
  else if (line == "lbfgs")
    method = Method::lBFGS;
  else
    throw(runtime_error("Unknown optimization method: " + line));
  return is;
}
}
