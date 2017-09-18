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

std::string spec_parser::to_string(const RandomVariable& rv) {
  std::string distr_spec;
  if (rv.distribution != nullptr) {
    auto distr_name
        = Distribution::distrtos(rv.distribution->type);
    auto args
        = intercalate<std::vector<std::string>::const_iterator, std::string>(
            rv.distribution->arguments.begin(),
            rv.distribution->arguments.end(), ",");
    distr_spec = distr_name + "(" + args + ")";
  } else {
    distr_spec = "unspecified";
  }
  return rv.full_id() + "~" + distr_spec;
}
