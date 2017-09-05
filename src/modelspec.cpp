#include "modelspec.hpp"

#include "log.hpp"
#include "spec_parser/driver.hpp"

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

  LOG(verbose) << "Model consists of " << variables.size() << " variables.";
  LOG(verbose) << rate_coefficients.size() << " variables are rate coefficients.";
  LOG(verbose) << odds_coefficients.size() << " variables are odds coefficients.";
}

std::istream& operator>>(std::istream& is, ModelSpec& model_spec)
{
  model_spec.from_stream(is);
  return is;
}
