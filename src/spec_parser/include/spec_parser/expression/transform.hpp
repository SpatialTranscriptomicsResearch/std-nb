#ifndef SPECPARSER_EXPRESSION_TRANSFORM_HPP
#define SPECPARSER_EXPRESSION_TRANSFORM_HPP

#include <functional>

#include "./visitor.hpp"

namespace spec_parser {
namespace expression {

template <typename T, typename U>
struct Transform : public Visitor<T> {
  Transform(const std::function<U(const T&)>& _t, const ExpPtr<T>& e) : t(_t) {
    visit(*e);
  }

  using Visitor<T>::visit;
  virtual void visit(Num<T>& e) override { res = num<U>(e.value); }
  virtual void visit(Var<T>& e) override { res = var<U>(t(e.value)); }
  virtual void visit(Bin<T>& e) override {
    res = std::make_shared<Bin<U>>(e.op, Transform<T, U>(t, e.lhs)(),
                                   Transform<T, U>(t, e.rhs)());
  }
  virtual void visit(Unr<T>& e) override {
    res = std::make_shared<Unr<U>>(e.op, Transform<T, U>(t, e.operand)());
  }

  ExpPtr<U> operator()() const { return res; }

private:
  ExpPtr<U> res;
  const std::function<U(const T&)>& t;
};

template <typename T, typename U>
ExpPtr<U> transform(const std::function<U(const T&)>& t, const ExpPtr<T>& e) {
  return Transform<T, U>(t, e)();
}

}  // namespace expression
}  // namespace spec_parser

#endif  // SPECPARSER_EXPRESSION_TRANSFORM_HPP
