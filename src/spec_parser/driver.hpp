#ifndef DRIVER_HH
#define DRIVER_HH

#include <unordered_map>
#include <string>

#include "Formula.hpp"
#include "RandomVariable.hpp"
#include "RegressionEquation.hpp"
#include "parser.tab.hpp"

// Tell Flex the lexer's prototype ...
#define YY_DECL yy::parser::symbol_type yylex(spec_parser::Driver& Driver)
// ... and declare it for the parser's sake.
YY_DECL;

namespace spec_parser {

struct ParseError : public std::runtime_error {
  const std::string line;
  const yy::location where;
  ParseError() = delete;
  ParseError(const std::string& line, const yy::location& where,
      const std::string& what);
};

class Driver {
  public:
  void error(const yy::location& l, const std::string& m);
  void error(const std::string& m);

  bool trace_scanning, trace_parsing;

  std::unordered_map<std::string, RegressionEquation> regression_equations;
  std::unordered_map<std::string, RandomVariable> random_variables;

  Driver();
  virtual ~Driver();

  void add_formula(const std::string& id, const Formula& formula);

  int parse(const std::string& s);
  yy::location& location() const;

  private:
  std::string cur_line;
};

} // namespace spec_parser

#endif // ! DRIVER_HH
