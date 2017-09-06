#include <exception>
#include <sstream>

#include "driver.hpp"
#include "parser.tab.hpp"

using namespace spec_parser;

ParseError::ParseError(const std::string& _line, const yy::location& _where,
    const std::string& what)
    : runtime_error(what)
    , line(_line)
    , where(_where)
{
}

Driver::~Driver() {}

Driver::Driver()
    : trace_scanning(false)
    , trace_parsing(false)
{
}

void Driver::error(const yy::location& l, const std::string& m)
{
  throw ParseError(cur_line, l, m);
}

void Driver::add_formula(const std::string& id, const Formula& formula)
{
  RegressionEquation req;
  for (auto& term : formula.terms) {
    RandomVariable rv(id, std::set<std::string>(term.cbegin(), term.cend()));
    req.variables.push_back(rv.full_id());
  }
  regression_equations[id] = req;
}
