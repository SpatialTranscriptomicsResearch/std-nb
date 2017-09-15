#include "spec_parser/Driver.hpp"

#include <exception>
#include <sstream>

#include "parser.tab.hpp"
#include "spec_parser/RandomVariable.hpp"

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
    auto variable
        = get_variable(id, std::set<std::string>(term.cbegin(), term.cend()));
    req.variables.push_back(variable->full_id());
  }
  regression_equations[id] = req;
}

RegressionEquation* Driver::get_equation(const std::string& id) {
  return &regression_equations[id];
}

RandomVariable* Driver::get_variable(
    const std::string& id, std::set<std::string> covariates)
{
  { // disregard unit covariate
    auto it = covariates.find(unit_covariate);
    if (it != covariates.end()) {
      covariates.erase(it);
    }
  }
  auto variable = RandomVariable(id, covariates);
  auto it = random_variables.find(variable.full_id());
  if (it != random_variables.end()) {
    return &(it->second);
  }
  random_variables[variable.full_id()] = variable;
  return &random_variables[variable.full_id()];
}
