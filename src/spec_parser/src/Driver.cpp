#include "spec_parser/Driver.hpp"

#include <exception>
#include <numeric>
#include <sstream>
#include <vector>

#include "aux.hpp"
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

void Driver::add_formula(const std::string& id, const Formula& formula) {
  auto n = formula.terms.size();
  assert(n > 0);
  std::vector<ExpType> exps(n);
  std::transform(
      begin(formula.terms), end(formula.terms), begin(exps),
      [this, &id](const auto& x) {
        return var(this->get_variable(id, std::set<std::string>(begin(x), end(x))));
      });
  regression_exprs[id] = accumulate1(begin(exps), end(exps));
}

Driver::ExpType& Driver::get_expr(const std::string& id) {
  return regression_exprs[id];
}

Driver::VarType& Driver::get_variable(
    const std::string& id, std::set<std::string> covariates)
{
  { // disregard unit covariate
    auto it = covariates.find(unit_covariate);
    if (it != covariates.end()) {
      covariates.erase(it);
    }
  }
  auto variable = std::make_shared<RandomVariable>(id, covariates);
  auto full_id = variable->full_id();
  { // check if variable already exists
    auto it = random_variables.find(full_id);
    if (it != random_variables.end()) {
      return it->second;
    }
  }
  auto res = random_variables.emplace(full_id, std::move(variable));
  return res.first->second;
}
