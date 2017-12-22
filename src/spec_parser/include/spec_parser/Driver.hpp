#ifndef DRIVER_HH
#define DRIVER_HH

#include <memory>
#include <unordered_map>
#include <set>
#include <string>

#include "spec_parser/Expression.hpp"
#include "spec_parser/Formula.hpp"
#include "spec_parser/RandomVariable.hpp"
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
  using ExpType = ExpressionPtr<VariablePtr>;
  using VarType = VariablePtr;

  void error(const yy::location& l, const std::string& m);
  void error(const std::string& m);

  bool trace_scanning, trace_parsing;

  std::unordered_map<std::string, ExpType> regression_exprs;
  std::unordered_map<std::string, VarType> random_variables;

  Driver();
  virtual ~Driver();

  void add_formula(const std::string& id, const Formula& formula);

  ExpType& get_expr(const std::string& id);
  VarType& get_variable(const std::string& id, std::set<std::string> covariates);

  int parse(const std::string& s);
  yy::location& location() const;

  private:
  std::string cur_line;
};

} // namespace spec_parser

#endif // ! DRIVER_HH
