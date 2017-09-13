#ifndef REGRESSIONEQUATION_HPP
#define REGRESSIONEQUATION_HPP

#include <vector>
#include <string>

namespace spec_parser {

struct RegressionEquation {
  std::vector<std::string> variables;

  RegressionEquation();
  RegressionEquation(const std::string& s);

  RegressionEquation operator*(const RegressionEquation& other) const;
};

} // namespace spec_parser

#endif // REGRESSIONEQUATION_HPP
