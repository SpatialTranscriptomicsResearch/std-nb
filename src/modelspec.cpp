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
  rate_coefficients = parser.regression_equations[RATE_VARIABLE].variables;
  odds_coefficients = parser.regression_equations[ODDS_VARIABLE].variables;
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
  log_func(">>>");

  log_func("Rate coefficients");
  log_func("-----------------");
  for (auto &x : model_spec.rate_coefficients) {
    log_func(x);
  }
  log_func("");

  log_func("Odds coefficients");
  log_func("-----------------");
  for (auto &x : model_spec.odds_coefficients) {
    log_func(x);
  }
  log_func("");

  log_func("Coefficient distributions");
  log_func("-------------------------");
  std::map<std::string, spec_parser::RandomVariable> sorted_variables(
      model_spec.variables.begin(), model_spec.variables.end());
  for (auto &x : sorted_variables) {
    std::string distr_spec;
    if (x.second.distribution != nullptr) {
      auto distr_name = spec_parser::Distribution::distrtos(x.second.distribution->type);
      auto args = intercalate<std::vector<std::string>::const_iterator, std::string>(
          x.second.distribution->arguments.begin(),
          x.second.distribution->arguments.end(), ",");
      distr_spec = distr_name + "(" + args + ")";
    } else {
      distr_spec = "default";
    }
    log_func(x.first + "~" + distr_spec);
  }

  log_func("<<<");
}
