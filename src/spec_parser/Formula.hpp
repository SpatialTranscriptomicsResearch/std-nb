#ifndef FORMULA_HPP
#define FORMULA_HPP

#include <string>
#include <unordered_set>

// TODO: Look into creating a better hash function
namespace std {
template <typename T>
struct hash<unordered_set<T>> {
  size_t operator()(const unordered_set<T>& xs) const {
    size_t h = 0;
    for (auto& x : xs) {
      h ^= hash<T>()(x);
    }
    return h;
  }
};
}

namespace spec_parser {

struct NonPositiveExponent : public std::invalid_argument {
  NonPositiveExponent()
      : invalid_argument("Error: exponent must be greater than 1.")
  {
  }
};

struct Formula {
  using Term = std::unordered_set<std::string>;
  using Terms = std::unordered_set<Term>;
  Terms terms;

  Formula() = default;
  Formula(const std::string& covariate);

  Formula add(const Formula& other) const;
  Formula interact(const Formula& other) const;
  Formula multiply(const Formula& other) const;
  Formula subtract(const Formula& other) const;
  Formula pow(int n) const;
};

} // namespace spec_parser

#endif // FORMULA_HPP
