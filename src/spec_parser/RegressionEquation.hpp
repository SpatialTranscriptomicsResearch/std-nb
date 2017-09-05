#ifndef REGRESSIONEQUATION_HPP
#define REGRESSIONEQUATION_HPP

#include <vector>
#include <string>

namespace spec_parser {

struct RegressionEquation {
  std::vector<std::string> variables;

  RegressionEquation() = default;
  RegressionEquation(const std::string& s) : variables({s}) {}

  RegressionEquation operator*(const RegressionEquation& other) {
    RegressionEquation ret;
    ret.variables = variables;
    for (auto& var : other.variables) {
      ret.variables.push_back(var);
    }
    return ret;
  }
};

} // namespace spec_parser

#endif // REGRESSIONEQUATION_HPP
