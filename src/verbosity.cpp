#include "verbosity.hpp"
#include <stdexcept>

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
  string token;
  is >> token;
  if (token == "everything")
    verb = Verbosity::everything;
  else if (token == "trace")
    verb = Verbosity::trace;
  else if (token == "debug")
    verb = Verbosity::debug;
  else if (token == "verbose")
    verb = Verbosity::verbose;
  else if (token == "info")
    verb = Verbosity::info;
  else if (token == "warning")
    verb = Verbosity::warning;
  else if (token == "error")
    verb = Verbosity::error;
  else if (token == "fatal")
    verb = Verbosity::fatal;
  else
    throw runtime_error("Error: unkown verbosity level '" + token + "'.");
  return is;
}
