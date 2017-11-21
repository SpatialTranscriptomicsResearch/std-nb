#ifndef SPECPARSER_EXPRESSION_SHOW_HPP
#define SPECPARSER_EXPRESSION_SHOW_HPP

#include <sstream>

#include "visitor.hpp"

namespace spec_parser {
namespace expression {

template <typename T>
struct Show : public Visitor<T> {
  Show(const ExpPtr<T>& e) { visit(*e); }
  using Visitor<T>::visit;
  virtual void visit(Num<T>& e) override { ss << e.value; }
  virtual void visit(Var<T>& e) override { ss << e.value; }
  virtual void visit(Bin<T>& e) override {
    auto a = Show<T>(e.lhs)(), b = Show<T>(e.rhs)();
    switch (e.op) {
      case Bin<T>::Type::ADD:
        ss << "(" << a << "+" << b << ")";
        break;
      case Bin<T>::Type::MUL:
        ss << "(" << a << "*" << b << ")";
        break;
    }
  }
  virtual void visit(Unr<T>& e) override {
    auto a = Show<T>(e.operand)();
    switch (e.op) {
      case Unr<T>::Type::NEG:
        ss << "-(" << a << ")";
        break;
      case Unr<T>::Type::INV:
        ss << "1/(" << a << ")";
        break;
      case Unr<T>::Type::LOG:
        ss << "log(" << a << ")";
        break;
      case Unr<T>::Type::EXP:
        ss << "exp(" << a << ")";
        break;
    }
  }

  std::string operator()() const { return ss.str(); }

private:
  std::stringstream ss;
};

template <typename T>
std::string show(const ExpPtr<T>& e) {
  return Show<T>(e)();
}

}  // namespace expression
}  // namespace spec_parser

#endif  // SPECPARSER_EXPRESSION_SHOW_HPP
