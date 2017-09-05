#ifndef MODELSPEC_HPP
#define MODELSPEC_HPP

#include <string>
#include <vector>

#include "coefficient.hpp"
#include "spec_parser/RandomVariable.hpp"
#include "spec_parser/driver.hpp"

namespace Exception {
namespace ModelSpec {
struct UnrecoverableParseError : public std::runtime_error {
  UnrecoverableParseError() : runtime_error("Error: unrecoverable parse error.") {}
};
} // namespace Exception
} // namespace ModelSpec

struct ModelSpec {
  void from_string(const std::string& str);
  void from_stream(std::istream& is);

  // TODO: make into functions
  std::vector<std::string> rate_coefficients, odds_coefficients;
  std::unordered_map<std::string, spec_parser::RandomVariable> variables;

  private:
  spec_parser::Driver parser;
};

std::istream& operator>>(std::istream& is, ModelSpec& model_spec);

#endif // MODELSPEC_HPP
