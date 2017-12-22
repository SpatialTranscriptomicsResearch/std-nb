#ifndef RANDOMVARIABLE_HPP
#define RANDOMVARIABLE_HPP

#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "spec_parser/Distribution.hpp"
#include "spec_parser/Expression.hpp"

namespace spec_parser {

struct RandomVariable;

using VariablePtr = std::shared_ptr<RandomVariable>;

struct RandomVariable {
  std::set<std::string> covariates;
  std::string id;

  std::shared_ptr<Distribution> distribution;

  RandomVariable();
  RandomVariable(const std::string& id, std::set<std::string> covariates);

  void set_distribution(const Distribution& distribution);

  std::string full_id() const;
};

std::string to_string(const RandomVariable& rv);

std::ostream& operator<<(std::ostream& os, const RandomVariable& var);
std::ostream& operator<<(std::ostream& os, const VariablePtr& var_ptr);

} // namespace spec_parser

#endif // RANDOMVARIABLE_HPP
