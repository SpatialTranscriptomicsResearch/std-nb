#ifndef SPECPARSER_EXPRESSION_COLLECT_HPP
#define SPECPARSER_EXPRESSION_COLLECT_HPP

#include <sstream>

#include "visitor.hpp"

namespace spec_parser {
namespace expression {

template <class T>
struct lessP : std::binary_function<T, T, bool> {
  bool operator()(const T& x, const T& y) const { return *x < *y; }
};

template <typename T>
struct Collect : public Visitor<T> {
  Collect(const ExpPtr<T>& e) { visit(*e); }
  using Visitor<T>::visit;
  virtual void visit(Num<T>& e) override {}
  virtual void visit(Var<T>& e) override { set.insert(e.value); }
  virtual void visit(Bin<T>& e) override {
    visit(*e.lhs);
    visit(*e.rhs);
  }
  virtual void visit(Unr<T>& e) override { visit(*e.operand); }

  using set_type = std::set<T, lessP<T>>;
  std::set<T> operator()() const { return set; }

private:
  std::set<T> set;
};

template <typename T>
std::set<T> collect_variables(const ExpPtr<T>& e) {
  return Collect<T>(e)();
}

}  // namespace expression
}  // namespace spec_parser

#endif  // SPECPARSER_EXPRESSION_COLLECT_HPP
