#ifndef SPECPARSER_EXPRESSION_SIMPLIFY_HPP
#define SPECPARSER_EXPRESSION_SIMPLIFY_HPP

#include "clone.hpp"
#include "visitor.hpp"

namespace spec_parser {
namespace expression {

template <typename T>
struct IsValue : public Visitor<T> {
  IsValue(const ExpPtr<T> e, double v) : value(v), identical(false) {
    visit(*clone(e));
  }
  using Visitor<T>::visit;
  virtual void visit(Num<T>& x) override { identical = x.value == value; }
  virtual void visit(Var<T>&) override {}
  virtual void visit(Unr<T>&) override {}
  virtual void visit(Bin<T>&) override {}

  bool operator()() const { return identical; }

private:
  const double value;
  bool identical;
};

template <typename T>
bool is_zero(const ExpPtr<T>& e) {
  return IsValue<T>(e, 0.0)();
}

template <typename T>
bool is_unit(const ExpPtr<T>& e) {
  return IsValue<T>(e, 1.0)();
}

template <typename T>
struct Simplify : public Visitor<T> {
  Simplify(const ExpPtr<T>& e) : x(e) { visit(*clone(e)); }
  using Visitor<T>::visit;
  virtual void visit(Num<T>& e) override {}
  virtual void visit(Var<T>& e) override {}
  virtual void visit(Bin<T>& e) override {
    auto &a = e.lhs, &b = e.rhs;
    auto sa = Simplify<T>(a)(), sb = Simplify<T>(b)();
    switch (e.op) {
      case Bin<T>::Type::ADD:
        if (is_zero(sa))
          x = sb;
        else if (is_zero(sb))
          x = sa;
        else
          x = sa + sb;
        break;
      case Bin<T>::Type::MUL:
        if (is_zero(sa) or is_zero(sb))
          x = num<T>(0);
        else if (is_unit(sa))
          x = sb;
        else if (is_unit(sb))
          x = sa;
        else
          x = sa * sb;
        break;
    }
  }
  virtual void visit(Unr<T>& e) override {
    auto& a = e.operand;
    auto sa = Simplify<T>(a)();
    switch (e.op) {
      case Unr<T>::Type::NEG:
        if (is_zero(sa))
          x = num<T>(0);
        else
          x = neg(sa);
        break;
      case Unr<T>::Type::INV:
        if (is_unit(sa))
          x = num<T>(1);
        else
          x = inv(sa);
        break;
      case Unr<T>::Type::LOG:
        if (is_unit(sa))
          x = num<T>(0);
        else
          x = log(sa);
        break;
      case Unr<T>::Type::EXP:
        if (is_zero(sa))
          x = num<T>(1);
        else
          x = exp(sa);
        break;
    }
  }

  ExpPtr<T> operator()() const { return x; }

private:
  ExpPtr<T> x;
};

template <typename T>
ExpPtr<T> simplify(const ExpPtr<T>& e) {
  LOG(debug) << "Simplifying expression " << show(e);
  auto res = Simplify<T>(e)();
  LOG(debug) << "Simplified expression " << show(res);
  return res;
}

}  // namespace expression
}  // namespace spec_parser

#endif  // SPECPARSER_EXPRESSION_SIMPLIFY_HPP
