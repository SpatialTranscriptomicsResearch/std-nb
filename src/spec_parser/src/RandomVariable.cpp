#include "spec_parser/RandomVariable.hpp"

using namespace spec_parser;

RandomVariable::RandomVariable() = default;

RandomVariable::RandomVariable(
    const std::string& _id, std::set<std::string> _covariates)
    : covariates(_covariates)
    , id(_id)
    , distribution(nullptr)
{
}

void RandomVariable::set_distribution(const Distribution& _distribution) {
  distribution = std::make_shared<Distribution>(_distribution);
}

std::string RandomVariable::full_id() const
{
  return id + "(" + intercalate<std::set<std::string>::iterator, std::string>(
                        covariates.begin(), covariates.end(), ",")
      + ")";
}
