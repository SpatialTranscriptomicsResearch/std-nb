#ifndef SPECPARSER_EXPRESSION_HPP
#define SPECPARSER_EXPRESSION_HPP

#include "spec_parser/expression/deriv.hpp"
#include "spec_parser/expression/simplify.hpp"
#include "spec_parser/expression/eval.hpp"
#include "spec_parser/expression/show.hpp"
#include "spec_parser/expression/transform.hpp"

namespace spec_parser {

template <typename T>
using Expression = expression::Exp<T>;
template <typename T>
using ExpressionPtr = expression::ExpPtr<T>;

using expression::var;
using expression::num;
using expression::log;
using expression::exp;

using expression::eval;
using expression::deriv;
using expression::simplify;
using expression::show;
using expression::transform;

}  // namespace spec_parser

#endif  // SPECPARSER_EXPRESSION_HPP
