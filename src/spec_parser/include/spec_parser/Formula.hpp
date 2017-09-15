#ifndef FORMULA_HPP
#define FORMULA_HPP

#include <string>
#include <unordered_set>

#include <boost/functional/hash.hpp>

namespace spec_parser {

struct NonPositiveExponent : public std::invalid_argument {
  NonPositiveExponent()
      : invalid_argument("Error: exponent must be greater than 1.")
  {
  }
};

struct FormulaTerm : public std::unordered_set<std::string> {
  FormulaTerm();
  FormulaTerm(std::initializer_list<std::string> init);
};

} // must do hash specialization of FormulaTerm outside of the spec_parser
  // namespace and before the declaration of FormulaTerms, as its declaration
  // (implicitly) instantiates FormulaTerm.

namespace std {
template <>
struct hash<spec_parser::FormulaTerm> {
  size_t operator()(const spec_parser::FormulaTerm& xs) const {
    size_t h = 0;
    for (auto& x : xs) {
      boost::hash_combine(h, boost::hash_value(x));
    }
    return h;
  }
};
}

namespace spec_parser {

struct FormulaTerms : public std::unordered_set<FormulaTerm> {
  FormulaTerms(std::initializer_list<FormulaTerm> init);
};

struct Formula {
  using Term = FormulaTerm;
  using Terms = FormulaTerms;

  Terms terms;

  Formula();
  Formula(std::initializer_list<Term> terms);

  Formula add(const Formula& other) const;
  Formula interact(const Formula& other) const;
  Formula multiply(const Formula& other) const;
  Formula subtract(const Formula& other) const;
  Formula pow(int n) const;
};

} // namespace spec_parser

#endif // FORMULA_HPP
