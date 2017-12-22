#ifndef SPECPARSER_EXPRESSION_PRECEDENCE_HPP
#define SPECPARSER_EXPRESSION_PRECEDENCE_HPP

#include "visitor.hpp"

namespace spec_parser {
namespace expression {

constexpr unsigned char precedence_min = 0;
constexpr unsigned char precedence_max = ~0;

constexpr unsigned char precedence_neg = precedence_min + 1;
constexpr unsigned char precedence_add = precedence_min + 2;
constexpr unsigned char precedence_mul = precedence_min + 3;
constexpr unsigned char precedence_inv = precedence_min + 4;
constexpr unsigned char precedence_log = precedence_max;
constexpr unsigned char precedence_exp = precedence_max;

template <typename T>
struct Precedence : public Visitor<T> {
  Precedence(const ExpPtr<T>& e) { visit(*e); }
  using Visitor<T>::visit;
  virtual void visit(Num<T>&) override { precedence = precedence_max; }
  virtual void visit(Var<T>&) override { precedence = precedence_max; }
  virtual void visit(Bin<T>& e) override {
    switch (e.op) {
      case Bin<T>::Type::ADD:
        precedence = precedence_add;
        break;
      case Bin<T>::Type::MUL:
        precedence = precedence_mul;
        break;
    }
  }
  virtual void visit(Unr<T>& e) override {
    switch (e.op) {
      case Unr<T>::Type::NEG:
        precedence = precedence_neg;
        break;
      case Unr<T>::Type::INV:
        precedence = precedence_inv;
        break;
      case Unr<T>::Type::LOG:
        precedence = precedence_log;
        break;
      case Unr<T>::Type::EXP:
        precedence = precedence_exp;
        break;
    }
  }

  unsigned char operator()() const { return precedence; }

private:
  unsigned char precedence;
};

template <typename T>
unsigned char precedence(const ExpPtr<T>& e) {
  return Precedence<T>(e)();
}

}  // namespace expression
}  // namespace spec_parser

#endif  // SPECPARSER_EXPRESSION_PRECEDENCE_HPP
