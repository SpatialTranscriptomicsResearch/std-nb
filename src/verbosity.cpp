#include "verbosity.hpp"

using namespace std;

string to_string(Verbosity verb) {
  switch (verb) {
    case Verbosity::everything:
      return "everything";
    case Verbosity::trace:
      return "trace";
    case Verbosity::debug:
      return "debug";
    case Verbosity::verbose:
      return "verbose";
    case Verbosity::info:
      return "info";
    case Verbosity::warning:
      return "warning";
    case Verbosity::error:
      return "error";
    case Verbosity::fatal:
      return "fatal";
    default:
      throw logic_error("Implementation of to_string(Verbosity) incomplete!");
  }
}

Verbosity verbosity = Verbosity::info;

ostream &operator<<(ostream &os, Verbosity verb) {
  os << to_string(verb);
  return os;
}
istream &operator>>(istream &is, Verbosity &verb) {
  // TODO implement
  return is;
}
