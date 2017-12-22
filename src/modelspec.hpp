#ifndef MODELSPEC_HPP
#define MODELSPEC_HPP

#include <string>
#include <vector>

#include "coefficient.hpp"
#include "spec_parser/RandomVariable.hpp"
#include "spec_parser/Driver.hpp"

namespace Exception {
namespace ModelSpec {
struct InvalidModel : public std::runtime_error {
  InvalidModel(const std::string& why)
      : runtime_error("Error: invalid model (" + why + ").") {}
};
struct UnrecoverableParseError : public std::runtime_error {
  UnrecoverableParseError() : runtime_error("Error: unrecoverable parse error.") {}
};
} // namespace Exception
} // namespace ModelSpec

struct ModelSpec {
  using Variable = spec_parser::Driver::VarType;
  using Expression = spec_parser::Driver::ExpType;

  void from_string(const std::string& str);
  void from_stream(std::istream& is);

  Expression rate_expr, odds_expr;
  std::unordered_map<std::string, Variable> variables;

  private:
  spec_parser::Driver parser;
};

void log(const std::function<void(const std::string& s)> log_func,
    const ModelSpec& model_spec);

std::istream& operator>>(std::istream& is, ModelSpec& model_spec);

#endif // MODELSPEC_HPP
