#ifndef RANDOMVARIABLE_HPP
#define RANDOMVARIABLE_HPP

#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "aux.hpp"
#include "spec_parser/Distribution.hpp"

namespace spec_parser {

struct RandomVariable {
  std::set<std::string> covariates;
  std::string id;

  Distribution distribution;

  RandomVariable();
  RandomVariable(std::string id, std::set<std::string> covariates);

  std::string full_id() const;
};

} // namespace spec_parser

#endif // RANDOMVARIABLE_HPP
