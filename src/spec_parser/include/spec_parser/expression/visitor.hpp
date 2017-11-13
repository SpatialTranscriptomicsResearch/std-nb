#ifndef SPECPARSER_EXPRESSION_VISITOR_HPP
#define SPECPARSER_EXPRESSION_VISITOR_HPP

#include "./exp.hpp"

namespace spec_parser {
namespace expression {

template <typename T>
struct Visitor {
  Visitor() = default;
  Visitor(Exp<T>& e) { visit(e); }
  virtual ~Visitor() {}
  virtual void visit(Exp<T>& e) { e.accept(*this); }
  virtual void visit(Num<T>& e) = 0;
  virtual void visit(Var<T>& e) = 0;
  virtual void visit(Bin<T>& e) = 0;
  virtual void visit(Unr<T>& e) = 0;
};

}  // namespace expression
}  // namespace spec_parser

#endif  // SPECPARSER_EXPRESSION_VISITOR_HPP
