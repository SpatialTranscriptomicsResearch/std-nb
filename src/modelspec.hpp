#ifndef MODELSPEC_HPP
#define MODELSPEC_HPP

#include <string>
#include <vector>

#include "coefficient.hpp"
#include "spec_parser/RandomVariable.hpp"
#include "spec_parser/Driver.hpp"

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

void log(const std::function<void(const std::string& s)> log_func,
    const ModelSpec& model_spec);

std::istream& operator>>(std::istream& is, ModelSpec& model_spec);

#endif // MODELSPEC_HPP
