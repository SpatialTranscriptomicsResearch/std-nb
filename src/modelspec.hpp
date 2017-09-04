#ifndef MODELSPEC_HPP
#define MODELSPEC_HPP

#include <string>
#include <vector>

#include "coefficient.hpp"
#include "spec_parser/RandomVariable.hpp"
#include "spec_parser/driver.hpp"

namespace Exception {
namespace ModelSpec {
} // namespace Exception
} // namespace ModelSpec

struct ModelSpec {
  void from_string(const std::string& str);
  void from_stream(std::istream& is);

  // TODO: make into functions
  std::vector<std::string> rate_coefficients, odds_coefficients;
  std::unordered_map<std::string, RandomVariable> variables;

  private:
  Driver parser;
};

std::istream& operator>>(std::istream& is, ModelSpec& model_spec);

#endif // MODELSPEC_HPP
