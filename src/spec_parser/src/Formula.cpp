#include "spec_parser/Formula.hpp"

#include <exception>

using namespace spec_parser;

Formula::Formula(const std::string& covariate)
    : terms{ { covariate } }
{
}

Formula Formula::add(const Formula& other) const
{
  Formula ret = *this;
  ret.terms.insert(other.terms.begin(), other.terms.end());
  return ret;
}

Formula Formula::interact(const Formula& other) const
{
  Formula ret;
  for (auto& x : terms) {
    Term term = x;
    for (auto& y : other.terms) {
      term.insert(y.begin(), y.end());
    }
    ret.terms.insert(term);
  }
  return ret;
}

Formula Formula::multiply(const Formula& other) const
{
  return this->interact(other).add(*this).add(other);
}

Formula Formula::subtract(const Formula& other) const
{
  Formula ret = *this;
  for (auto& x : other.terms) {
    auto it = ret.terms.find(x);
    if (it != ret.terms.end()) {
      ret.terms.erase(it);
    }
  }
  return ret;
}

Formula Formula::pow(int n) const
{
  if (n == 0) {
    throw NonPositiveExponent();
  }
  if (n == 1) {
    return *this;
  }
  return this->pow(n - 1).multiply(*this);
}
