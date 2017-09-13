#include "RegressionEquation.hpp"

using namespace spec_parser;

RegressionEquation::RegressionEquation() = default;

RegressionEquation::RegressionEquation(const std::string& s)
    : variables({ s })
{
}

RegressionEquation RegressionEquation::operator*(
    const RegressionEquation& other) const
{
  RegressionEquation ret;
  ret.variables = variables;
  for (auto& var : other.variables) {
    ret.variables.push_back(var);
  }
  return ret;
}
