#include "RandomVariable.hpp"

using namespace spec_parser;

RandomVariable::RandomVariable() = default;

RandomVariable::RandomVariable(
    std::string _id, std::set<std::string> _covariates)
    : covariates(_covariates)
    , id(_id)
{
}

std::string RandomVariable::full_id() const
{
  return id + "(" + intercalate<std::set<std::string>::iterator, std::string>(
                        covariates.begin(), covariates.end(), ",")
      + ")";
}
