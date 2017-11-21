#ifndef SPECPARSER_EXPRESSION_DERIV_HPP
#define SPECPARSER_EXPRESSION_DERIV_HPP

#include "clone.hpp"
#include "visitor.hpp"

namespace spec_parser {
namespace expression {

template <typename T>
struct Deriv : public Visitor<T> {
  Deriv(const T& _v, const ExpPtr<T>& e) : v(_v) { visit(*clone(e)); }
  using Visitor<T>::visit;
  virtual void visit(Num<T>&) override { x = num<T>(0); }
  virtual void visit(Var<T>& e) override {
    x = e.value == v ? num<T>(1) : num<T>(0);
  };
  virtual void visit(Bin<T>& e) override {
    auto &a = e.lhs, &b = e.rhs;
    auto da = Deriv<T>(v, a)(), db = Deriv<T>(v, b)();
    switch (e.op) {
      case Bin<T>::Type::ADD:
        x = da + db;
        break;
      case Bin<T>::Type::MUL:
        x = (da * b) + (a * db);
        break;
    }
  }
  virtual void visit(Unr<T>& e) override {
    auto& a = e.operand;
    auto da = Deriv<T>(v, a)();
    switch (e.op) {
      case Unr<T>::Type::NEG:
        x = -da;
        break;
      case Unr<T>::Type::INV:
        x = -(da / (a * a));
        break;
      case Unr<T>::Type::LOG:
        x = da / a;
        break;
      case Unr<T>::Type::EXP:
        x = exp(a) * da;
        break;
    }
  }

  ExpPtr<T> operator()() const { return x; }

private:
  const T& v;
  ExpPtr<T> x;
};

template <typename T>
ExpPtr<T> deriv(const T& x, const ExpPtr<T>& e) {
  LOG(verbose) << "Computing derivate w.r.t. " << to_string(*x)
               << " of expression " << show(e);
  auto res = Deriv<T>(x, e)();
  LOG(verbose) << "Computed derivate w.r.t. " << to_string(*x)
               << " of expression " << show(e) << " = " << show(res);
  return Deriv<T>(x, e)();
}

}  // namespace expression
}  // namespace spec_parser

#endif  // SPECPARSER_EXPRESSION_DERIV_HPP
