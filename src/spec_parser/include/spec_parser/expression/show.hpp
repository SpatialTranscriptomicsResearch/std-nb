#ifndef SPECPARSER_EXPRESSION_SHOW_HPP
#define SPECPARSER_EXPRESSION_SHOW_HPP

#include <sstream>
#include <utility>

#include "precedence.hpp"
#include "visitor.hpp"

namespace spec_parser {
namespace expression {

template <typename T>
struct Show : public Visitor<T> {
  Show(const ExpPtr<T>& e) : _precedence(precedence(e)) { visit(*e); }
  using Visitor<T>::visit;
  virtual void visit(Num<T>& e) override { ss << e.value; }
  virtual void visit(Var<T>& e) override { ss << e.value; }
  virtual void visit(Bin<T>& e) override {
    auto[a, apred] = Show<T>(e.lhs)();
    auto[b, bpred] = Show<T>(e.rhs)();
    if (apred < _precedence) {
      a = "(" + a + ")";
    }
    if (bpred < _precedence) {
      b = "(" + b + ")";
    }
    switch (e.op) {
      case Bin<T>::Type::ADD:
        ss << a << "+" << b;
        break;
      case Bin<T>::Type::MUL:
        ss << a << "*" << b;
        break;
    }
  }
  virtual void visit(Unr<T>& e) override {
    auto[a, apred] = Show<T>(e.operand)();
    if (apred <= _precedence) {
      a = "(" + a + ")";
    }
    switch (e.op) {
      case Unr<T>::Type::NEG:
        ss << "-" << a;
        break;
      case Unr<T>::Type::INV:
        ss << "1/" << a;
        break;
      case Unr<T>::Type::LOG:
        ss << "log" << a;
        break;
      case Unr<T>::Type::EXP:
        ss << "exp" << a;
        break;
    }
  }

  std::pair<std::string, decltype(precedence(ExpPtr<T>()))> operator()() const {
    return {ss.str(), _precedence};
  }

private:
  std::stringstream ss;
  decltype(precedence(ExpPtr<T>())) _precedence;
};

template <typename T>
std::string show(const ExpPtr<T>& e) {
  return Show<T>(e)().first;
}

}  // namespace expression
}  // namespace spec_parser

#endif  // SPECPARSER_EXPRESSION_SHOW_HPP
