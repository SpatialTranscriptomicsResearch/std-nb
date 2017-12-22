#include "modelspec.hpp"

#include <map>

#include "log.hpp"

#define RATE_VARIABLE "rate"
#define ODDS_VARIABLE "odds"

static std::string loc_indicator(const yy::location& loc) {
  std::string ret;
  size_t i = 1;
  for (; i < loc.begin.column; ++i) {
    ret.push_back(' ');
  }
  for (; i < loc.end.column; ++i) {
    ret.push_back('^');
  }
  return ret;
}

void ModelSpec::from_string(const std::string& str)
{
  std::istringstream is(str);
  from_stream(is);
}

void ModelSpec::from_stream(std::istream& is)
{
  std::string line;
  for (size_t i = 1; std::getline(is, line); ++i) {
    try {
      LOG(debug) << "Parsing line " << i << ": " << line;
      parser.parse(line);
      parser.location().lines(1);
    } catch(const spec_parser::ParseError& e) {
      LOG(error) << "Failed to parse model specification (" << e.where << "):";
      LOG(error) << e.line;
      LOG(error) << loc_indicator(e.where);
      LOG(error) << "The error returned was: " << e.what();
      throw Exception::ModelSpec::UnrecoverableParseError();
    }
  }
  for (auto &x : parser.regression_exprs) {
    if (x.first != RATE_VARIABLE and x.first != ODDS_VARIABLE) {
      throw Exception::ModelSpec::InvalidModel(
          "declaration of '" + x.first
          + "' is meaningless; please check the model specification");
    }
  }
  rate_expr = parser.regression_exprs[RATE_VARIABLE];
  odds_expr = parser.regression_exprs[ODDS_VARIABLE];
  variables = parser.random_variables;
}

std::istream& operator>>(std::istream& is, ModelSpec& model_spec)
{
  model_spec.from_stream(is);
  return is;
}

void log(const std::function<void(const std::string& s)> log_func,
    const ModelSpec& model_spec)
{
  using namespace spec_parser;

  log_func(">>>");

  log_func("Rate expression");
  log_func("-----------------");
  log_func(show(model_spec.rate_expr));
  log_func("");

  log_func("Odds expression");
  log_func("-----------------");
  log_func(show(model_spec.odds_expr));
  log_func("");

  log_func("Coefficient distributions");
  log_func("-------------------------");
  std::map<std::string, ModelSpec::Variable> sorted_variables(
      model_spec.variables.begin(), model_spec.variables.end());
  for (auto &x : sorted_variables) {
    auto &v = x.second;
    if (v->distribution != nullptr) {
      log_func(to_string(*v) + "~" + to_string(*v->distribution));
    } else {
      log_func(to_string(*v) + "~undefined");
    }
  }

  log_func("<<<");
}
