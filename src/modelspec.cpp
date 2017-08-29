#include "modelspec.hpp"

#include "log.hpp"
#include "spec_parser/driver.hpp"

#define RATE_VARIABLE "rate"
#define ODDS_VARIABLE "odds"

static ModelSpec::RandomVariable interpret_variable(
    const std::unordered_map<std::string, RandomVariable>& random_variables,
    const std::string& identifier)
{
  LOG(debug) << "Treating variable '" << identifier << "'.";
  auto it = random_variables.find(identifier);
  if (it == random_variables.end()) {
    throw std::runtime_error("Variable '" + identifier + "' is not defined.");
  }
  ModelSpec::RandomVariable ret;
  ret.distribution = it->second.distribution;
  ret.covariates = it->second.covariates;
  if (ret.distribution == ModelSpec::RandomVariable::Distribution::fixed) {
    if (ret.arguments.size() != 1) {
      throw std::runtime_error(
          "Fixed variables must have exactly one argument (their value).");
    }
    ret.value = std::stod(it->second.arguments[0]);
  } else {
    for (auto& arg : it->second.arguments) {
      ret.arguments.push_back(interpret_variable(random_variables, arg));
    }
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
  while (std::getline(is, line)) {
    parser.parse(line);
  }

  LOG(debug) << "Interpreting rate variables.";
  for (auto& variable : parser.regression_equations[RATE_VARIABLE].variables) {
    rate_coefficients.push_back(
        interpret_variable(parser.random_variables, variable));
  }
  LOG(debug) << "Interpreting odds variables.";
  for (auto& variable : parser.regression_equations[ODDS_VARIABLE].variables) {
    odds_coefficients.push_back(
        interpret_variable(parser.random_variables, variable));
  }
}

std::string ModelSpec::to_string() const { return "NOT IMPLEMENTED"; }

std::istream& operator>>(std::istream& is, ModelSpec& model_spec)
{
  model_spec.from_stream(is);
  return is;
}

std::ostream& operator<<(std::ostream& os, const ModelSpec& model_spec)
{
  os << model_spec.to_string();
  return os;
}
