#include "aux.hpp"
#include "sampling_method.hpp"

using namespace std;

namespace Sampling {

ostream &operator<<(ostream &os, const Method &method) {
  switch (method) {
    case Method::Mean:
      os << "Mean";
      break;
    case Method::Multinomial:
      os << "Multinomial";
      break;
    case Method::MH:
      os << "MH";
      break;
    case Method::HMC:
      os << "HMC";
      break;
    case Method::RPROP:
      os << "RPROP";
      break;
  }
  return os;
}

istream &operator>>(istream &is, Method &method) {
  string line;
  getline(is, line);
  line = to_lower(line);
  if (line == "mean")
    method = Method::Mean;
  else if (line == "multinomial")
    method = Method::Multinomial;
  else if (line == "mh")
    method = Method::MH;
  else if (line == "hmc")
    method = Method::HMC;
  else if (line == "rprop")
    method = Method::RPROP;
  else
    throw(runtime_error("Unknown sampling method: " + line));
  return is;
}
}
