#ifndef SPECPARSER_EXPRESSION_CLONE_HPP
#define SPECPARSER_EXPRESSION_CLONE_HPP

#include "transform.hpp"

namespace spec_parser {
namespace expression {

template <typename T>
ExpPtr<T> clone(const ExpPtr<T>& e) {
  return transform<T, T>([](const T& x) { return x; }, e);
}

}  // namespace expression
}  // namespace spec_parser

#endif  // SPECPARSER_EXPRESSION_CLONE_HPP
