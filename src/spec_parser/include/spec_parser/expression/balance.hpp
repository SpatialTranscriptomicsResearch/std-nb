#ifndef SPECPARSER_EXPRESSION_BALANCEE_HPP
#define SPECPARSER_EXPRESSION_BALANCEE_HPP

#include "clone.hpp"
#include "visitor.hpp"

namespace spec_parser {
namespace expression {

template <typename T>
ExpPtr<T> balance(const ExpPtr<T>& e);

namespace detail {
enum struct Types { Num, Var, Add, Mul, Neg, Inv, Exp, Log };

template <typename T>
struct WhichType : public Visitor<T> {
  WhichType(const ExpPtr<T> e) : type(Types::Num) { visit(*clone(e)); }
  using Visitor<T>::visit;
  virtual void visit(Num<T>&) override { type = Types::Num; }
  virtual void visit(Var<T>&) override { type = Types::Var; }
  virtual void visit(Bin<T>& x) override {
    switch (x.op) {
      case Bin<T>::Type::ADD:
        type = Types::Add;
        break;
      case Bin<T>::Type::MUL:
        type = Types::Mul;
        break;
    }
  }
  virtual void visit(Unr<T>& x) override {
    switch (x.op) {
      case Unr<T>::Type::NEG:
        type = Types::Neg;
        break;
      case Unr<T>::Type::INV:
        type = Types::Inv;
        break;
      case Unr<T>::Type::EXP:
        type = Types::Exp;
        break;
      case Unr<T>::Type::LOG:
        type = Types::Log;
        break;
    }
  }

  Types operator()() const { return type; }

private:
  Types type;
};

template <typename T>
bool is_type(const ExpPtr<T>& e, Types type) {
  return WhichType<T>(e)() == type;
}

template <typename T>
bool is_sum(const ExpPtr<T>& e) {
  return is_type<T>(e, Types::Add);
}

template <typename T>
bool is_mul(const ExpPtr<T>& e) {
  return is_type<T>(e, Types::Mul);
}

template <typename T>
struct Terms : public Visitor<T> {
  Terms(const ExpPtr<T>& e, Types type_) : type(type_) {
    if (is_type(e, type))
      visit(*clone(e));
  }
  using Visitor<T>::visit;
  virtual void visit(Num<T>& e) override {}
  virtual void visit(Var<T>& e) override {}
  virtual void visit(Bin<T>& e) override {
    for (auto& operand : {e.lhs, e.rhs})
      if (is_type(operand, type)) {
        for (auto& term : Terms<T>(operand, type)())
          v.push_back(term);
      } else {
        v.push_back(operand);
      }
  }
  virtual void visit(Unr<T>& e) override {}

  std::vector<ExpPtr<T>> operator()() const { return v; }

private:
  Types type;
  std::vector<ExpPtr<T>> v;
};

template <typename T>
std::vector<ExpPtr<T>> terms(const ExpPtr<T>& e, Types type) {
  return Terms<T>(e, type)();
}

template <typename T, typename Iter>
ExpPtr<T> balance_terms(
    const Iter begin, const Iter end,
    std::function<ExpPtr<T>(const ExpPtr<T>&, const ExpPtr<T>&)> fnc) {
  size_t n = std::distance(begin, end);
  switch (n) {
    case 0:
      // impossible
      return num<T>(0);
    case 1:
      return *begin;
    case 2:
      return fnc(*begin, *(begin + 1));
    case 3:
      return fnc(balance_terms<T>(begin, begin + 2, fnc), *(begin + 2));
    default: {
      size_t k = (n + 1) / 2;
      return fnc(balance_terms<T>(begin, begin + k, fnc),
                 balance_terms<T>(begin + k, end, fnc));
    }
  }
}

template <typename T>
ExpPtr<T> balance_terms_sub(
    const ExpPtr<T>& e, Types type,
    std::function<ExpPtr<T>(const ExpPtr<T>&, const ExpPtr<T>&)> fnc) {
  auto ts = terms(e, type);
  std::transform(begin(ts), end(ts), begin(ts), balance<T>);
  return balance_terms<T>(begin(ts), end(ts), fnc);
}

template <typename T>
ExpPtr<T> balance_add(const ExpPtr<T>& e) {
  return balance_terms_sub<T>(e, Types::Add, add<T>);
}

template <typename T>
ExpPtr<T> balance_mul(const ExpPtr<T>& e) {
  return balance_terms_sub<T>(e, Types::Mul, mul<T>);
}

template <typename T>
struct Balance : public Visitor<T> {
  Balance(const ExpPtr<T>& e) : x(e) { visit(*clone(e)); }
  using Visitor<T>::visit;
  virtual void visit(Num<T>& e) override {}
  virtual void visit(Var<T>& e) override {}
  virtual void visit(Bin<T>& e) override {
    switch (e.op) {
      case Bin<T>::Type::ADD:
        x = balance_add<T>(x);
        break;
      case Bin<T>::Type::MUL:
        x = balance_mul<T>(x);
        break;
    }
  }
  virtual void visit(Unr<T>& e) override {
    switch (e.op) {
      case Unr<T>::Type::NEG:
        x = neg(balance(e.operand));
        break;
      case Unr<T>::Type::INV:
        x = inv(balance(e.operand));
        break;
      case Unr<T>::Type::EXP:
        x = exp(balance(e.operand));
        break;
      case Unr<T>::Type::LOG:
        x = log(balance(e.operand));
        break;
    }
  }

  ExpPtr<T> operator()() const { return x; }

private:
  ExpPtr<T> x;
};

}  // namespace detail

template <typename T>
ExpPtr<T> balance(const ExpPtr<T>& e) {
  return detail::Balance<T>(e)();
}

}  // namespace expression
}  // namespace spec_parser

#endif  // SPECPARSER_EXPRESSION_BALANCEE_HPP
