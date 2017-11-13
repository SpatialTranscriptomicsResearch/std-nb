#ifndef SPECPARSER_EXPRESSION_EVAL_HPP
#define SPECPARSER_EXPRESSION_EVAL_HPP

#include <cmath>
#include <functional>

#include "./visitor.hpp"

namespace spec_parser {
namespace expression {

template <typename T>
struct Eval : public Visitor<T> {
  Eval(const std::function<double(const T&)>& _f, const ExpPtr<T>& e) : f(_f) {
    visit(*e);
  }
  using Visitor<T>::visit;
  virtual void visit(Num<T>& e) override { x = e.value; }
  virtual void visit(Var<T>& e) override { x = f(e.value); }
  virtual void visit(Bin<T>& e) override {
    auto a = Eval<T>(f, e.lhs)(), b = Eval<T>(f, e.rhs)();
    switch (e.op) {
      case Bin<T>::Type::ADD:
        x = a + b;
        break;
      case Bin<T>::Type::MUL:
        x = a * b;
        break;
    }
  }
  virtual void visit(Unr<T>& e) override {
    auto a = Eval<T>(f, e.operand)();
    switch (e.op) {
      case Unr<T>::Type::NEG:
        x = -a;
        break;
      case Unr<T>::Type::INV:
        x = 1 / a;
        break;
      case Unr<T>::Type::LOG:
        x = std::log(a);
        break;
      case Unr<T>::Type::EXP:
        x = std::exp(a);
        break;
    }
  }

  double operator()() const { return x; }

private:
  const std::function<double(const T&)> f;
  double x;
};

template <typename T>
double eval(const std::function<double(const T&)>& f, const ExpPtr<T>& e) {
  return Eval<T>(f, e)();
}

}  // namespace expression
}  // namespace spec_parser

#endif  // SPECPARSER_EXPRESSION_EVAL_HPP
