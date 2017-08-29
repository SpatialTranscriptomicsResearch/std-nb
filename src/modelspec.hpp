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
  struct RandomVariable;

  void from_string(const std::string& str);
  void from_stream(std::istream& is);
  std::string to_string() const;

  std::vector<RandomVariable> rate_coefficients, odds_coefficients;

  private:
  Driver parser;
};

// TODO: would be nicer with a polymorphic, std::variant-like approach
struct ModelSpec::RandomVariable {
  using Distribution = Coefficient::Distribution;

  Distribution distribution;
  double value;  // value for fixed variables
  std::vector<RandomVariable> arguments;
  std::vector<std::string> covariates;
};

std::istream& operator>>(std::istream& is, ModelSpec& model_spec);
std::ostream& operator<<(std::ostream& os, const ModelSpec& model_spec);

#endif // MODELSPEC_HPP
