#ifndef DRIVER_HH
#define DRIVER_HH

#include <unordered_map>
#include <string>

#include "RandomVariable.hpp"
#include "RegressionEquation.hpp"
#include "parser.tab.hpp"

// Tell Flex the lexer's prototype ...
#define YY_DECL yy::parser::symbol_type yylex(Driver& Driver)
// ... and declare it for the parser's sake.
YY_DECL;

// TODO: add namespace

class Driver {
  public:
  void error(const yy::location& l, const std::string& m);
  void error(const std::string& m);

  bool trace_scanning, trace_parsing;

  std::unordered_map<std::string, RegressionEquation> regression_equations;
  std::unordered_map<std::string, RandomVariable> random_variables;

  Driver();
  virtual ~Driver();

  int parse(const std::string& s);

  private:
  std::string cur_line;
};

#endif // ! DRIVER_HH
