#ifndef DRIVER_HH
#define DRIVER_HH

#include <unordered_map>
#include <string>

#include "RandomVariable.hpp"
#include "RegressionEquation.hpp"
#include "spec_parser/parser.tab.hh"

// Tell Flex the lexer's prototype ...
#define YY_DECL yy::parser::symbol_type yylex(Driver& Driver)
// ... and declare it for the parser's sake.
YY_DECL;

// TODO: add namespace

class Driver {
  public:
  static void error(const yy::location& l, const std::string& m);
  static void error(const std::string& m);

  bool trace_scanning, trace_parsing;

  std::unordered_map<std::string, RegressionEquation> regression_equations;
  std::unordered_map<std::string, RandomVariable> random_variables;

  // The name of the file being parsed.
  // Used later to pass the file name to the location tracker.
  std::string file;

  Driver();
  virtual ~Driver();

  int parse(const std::string& f);

  private:
  // Handling the scanner.
  void scan_begin();
  void scan_end();
};

#endif // ! DRIVER_HH
