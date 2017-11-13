#include "spec_parser/RandomVariable.hpp"

#include "aux.hpp"

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
  return rv.full_id();
}

std::ostream& spec_parser::operator<<(std::ostream& os,
                                      const RandomVariable& var) {
  os << to_string(var);
  return os;
}
std::ostream& spec_parser::operator<<(std::ostream& os,
                                      const VariablePtr& var_ptr) {
  os << *var_ptr;
  return os;
}
