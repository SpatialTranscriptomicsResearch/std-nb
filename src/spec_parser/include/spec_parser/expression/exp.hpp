#ifndef SPECPARSER_EXPRESSION_EXP_HPP
#define SPECPARSER_EXPRESSION_EXP_HPP

#include <memory>

namespace spec_parser {
namespace expression {

template <typename T>
struct Visitor;

template <typename T>
struct Exp {
  virtual ~Exp() {}
  virtual void accept(Visitor<T>& v) = 0;
};

template <typename T>
using ExpPtr = std::shared_ptr<Exp<T>>;

/**
 * Binary operators
 */
enum class Bin_ { ADD, MUL };

template <typename T>
struct Bin : public Exp<T> {
  using Type = Bin_;
  Type op;
  ExpPtr<T> lhs, rhs;
  Bin(Type _op, const ExpPtr<T>& _lhs, const ExpPtr<T>& _rhs)
      : op(_op), lhs(_lhs), rhs(_rhs) {}
  virtual void accept(Visitor<T>& v) override { v.visit(*this); }
};

template <typename T>
ExpPtr<T> add(const ExpPtr<T>& lhs, const ExpPtr<T>& rhs) {
  return std::make_shared<Bin<T>>(Bin<T>::Type::ADD, lhs, rhs);
}
template <typename T>
ExpPtr<T> mul(const ExpPtr<T>& lhs, const ExpPtr<T>& rhs) {
  return std::make_shared<Bin<T>>(Bin<T>::Type::MUL, lhs, rhs);
}

/**
 * Unary operators
 */
enum class Unr_ {
  NEG,
  INV,
  LOG,
  EXP,
};

template <typename T>
struct Unr : public Exp<T> {
  using Type = Unr_;
  Type op;
  ExpPtr<T> operand;
  Unr(Type _op, const ExpPtr<T>& _operand) : op(_op), operand(_operand) {}
  ~Unr() {}
  virtual void accept(Visitor<T>& v) override { v.visit(*this); }
};

template <typename T>
ExpPtr<T> neg(const ExpPtr<T>& operand) {
  return std::make_shared<Unr<T>>(Unr_::NEG, operand);
}
template <typename T>
ExpPtr<T> inv(const ExpPtr<T>& operand) {
  return std::make_shared<Unr<T>>(Unr_::INV, operand);
}
template <typename T>
ExpPtr<T> log(const ExpPtr<T>& operand) {
  return std::make_shared<Unr<T>>(Unr_::LOG, operand);
}
template <typename T>
ExpPtr<T> exp(const ExpPtr<T>& operand) {
  return std::make_shared<Unr<T>>(Unr_::EXP, operand);
}

/**
 * Numeric data
 */
template <typename T>
struct Num : public Exp<T> {
  double value;
  Num(double _value) : value(_value) {}
  virtual void accept(Visitor<T>& v) override { v.visit(*this); }
};

template <typename T>
ExpPtr<T> num(double val) {
  return std::make_shared<Num<T>>(val);
}

/**
 * Variable data
 */
template <typename T>
struct Var : public Exp<T> {
  T value;
  Var(const T& _value) : value(_value) {}
  virtual void accept(Visitor<T>& v) override { v.visit(*this); }
};

template <typename T>
ExpPtr<T> var(const T& val) {
  return std::make_shared<Var<T>>(val);
}

/**
 * Operators
 */
template <typename T>
ExpPtr<T> operator+(const ExpPtr<T>& a, const ExpPtr<T>& b) {
  return add(a, b);
}
template <typename T>
ExpPtr<T> operator-(const ExpPtr<T>& a, const ExpPtr<T>& b) {
  return add(a, neg(b));
}
template <typename T>
ExpPtr<T> operator*(const ExpPtr<T>& a, const ExpPtr<T>& b) {
  return mul(a, b);
}
template <typename T>
ExpPtr<T> operator/(const ExpPtr<T>& a, const ExpPtr<T>& b) {
  return mul(a, inv(b));
}
template <typename T>
ExpPtr<T> operator-(const ExpPtr<T>& a) {
  return neg(a);
}

}  // namespace expression
}  // namespace spec_parser

#endif  // SPECPARSER_EXPRESSION_EXP_HPP
