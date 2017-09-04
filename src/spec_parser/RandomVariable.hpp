#ifndef RANDOMVARIABLE_HPP
#define RANDOMVARIABLE_HPP

#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "../aux.hpp"
#include "Distribution.hpp"

struct RandomVariable {
  std::set<std::string> covariates;
  std::string id;

  Distribution distribution;

  RandomVariable() = default;
  RandomVariable(std::string _id, std::set<std::string> _covariates)
      : covariates(_covariates)
      , id(_id)
  {
  }

  std::string full_id() const
  {
    return id + "(" +
      intercalate<std::set<std::string>::iterator, std::string>(
          covariates.begin(), covariates.end(), ",")
      + ")";
  }
};

#endif // RANDOMVARIABLE_HPP
