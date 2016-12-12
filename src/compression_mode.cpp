#include <stdexcept>
#include "compression_mode.hpp"

using namespace std;

string to_string(CompressionMode mode) {
  switch(mode) {
    case CompressionMode::gzip:
      return ".gz";
    case CompressionMode::bzip2:
      return ".bz2";
    default:
      return "";
  }
}

ostream &operator<<(ostream &os, CompressionMode mode) {
  os << to_string(mode);
  return os;
}
istream &operator>>(istream &is, CompressionMode &mode) {
  string token;
  is >> token;
  if(token == "gz" or token == ".gz")
    mode = CompressionMode::gzip;
  else if(token == "bz2" or token == ".bz2")
    mode = CompressionMode::bzip2;
  else if(token == "none")
    mode = CompressionMode::none;
  else
    throw runtime_error("Error: compression mode '" + token + "' not understood.");
  return is;
}
