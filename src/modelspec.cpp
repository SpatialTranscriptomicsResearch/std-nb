#include "modelspec.hpp"

#include "log.hpp"
#include "spec_parser/driver.hpp"

#define RATE_VARIABLE "rate"
#define ODDS_VARIABLE "odds"

void ModelSpec::from_string(const std::string& str)
{
  std::istringstream is(str);
  from_stream(is);
}

void ModelSpec::from_stream(std::istream& is)
{
  std::string line;
  while (std::getline(is, line)) {
    try {
      parser.parse(line);
    } catch(const std::runtime_error& e) {
      LOG(error) << "Unrecoverable parse error:";
      LOG(error) << e.what();
      exit(EXIT_FAILURE);
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
