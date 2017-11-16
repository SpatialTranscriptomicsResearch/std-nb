%skeleton "lalr1.cc"
%require "3.0.4"

%defines
%define parser_class_name {parser}
%define api.token.constructor
%define api.value.type variant
%define parse.assert

%code requires
{
#include <memory>
#include <string>
#include <set>

#include "spec_parser/Distribution.hpp"
#include "spec_parser/Expression.hpp"
#include "spec_parser/Formula.hpp"
#include "spec_parser/RandomVariable.hpp"

namespace spec_parser {

constexpr char unit_covariate[] = "1";

class Driver;

} // namespace_spec_parser

}

// The parsing context.
%param { spec_parser::Driver& driver }

%locations

%define parse.trace
%define parse.error verbose

%code
{
#include "spec_parser/Driver.hpp"

static void assert_unit(const std::string& str) {
  if (str != spec_parser::unit_covariate) {
    std::stringstream error_msg;
    error_msg << "Covariates must either be the unit covariate "
              << "(" << spec_parser::unit_covariate << ")"
              << " or non-numeric.";
    throw std::invalid_argument(error_msg.str());
  }
}
}

%define api.token.prefix {TOK_}
%token
  END  0  "end of file"
  ASSIGN  ":="
  EQUAL   "="
  TILDE   "~"
  COMMA   ","
  MINUS   "-"
  PLUS    "+"
  STAR    "*"
  COLON   ":"
  SLASH   "/"
  EXPON   "^"
  LPAREN  "("
  RPAREN  ")"
  LOG     "log"
  EXP     "exp"
;
%token <std::string> IDENTIFIER "identifier"
%token <std::string> NUMERIC "numeric"

%type <std::string> covariate;
%type <std::set<std::string>> covariates;

%type <spec_parser::Distribution> distr;
%type <std::string> distr_arg;
%type <std::vector<std::string>> distr_args;

%type <std::string> regressand;
%type <spec_parser::VariablePtr> regressor;
%type <spec_parser::ExpressionPtr<spec_parser::VariablePtr>> regression_expr;

%type <spec_parser::Formula> formula_expr;

%%
%left "+" "-";
%left "^";
%left "*";
%left ":";
%left "exp" "log";

%start program;

program: statements;

statements: %empty | statements statement;

statement: regression_formula
         | regression_eq
         | distr_spec;

regression_formula: regressand ":=" formula_expr { driver.add_formula($1, $3); };

regressand: "identifier" { $$ = $1; };

formula_expr: covariate { $$ = spec_parser::Formula{ { $1 } }; }
            | "numeric" { assert_unit($1); $$ = spec_parser::Formula{ { } }; }
            | "(" formula_expr ")" { $$ = $2; }
            | formula_expr "+" formula_expr { $$ = $1.add($3); }
            | formula_expr "-" formula_expr { $$ = $1.subtract($3); }
            | formula_expr ":" formula_expr { $$ = $1.interact($3); }
            | formula_expr "*" formula_expr { $$ = $1.multiply($3); }
            | formula_expr "^" "numeric" { $$ = $1.pow(std::stoi($3)); };

covariates: %empty { $$ = std::set<std::string> {}; }
          | covariate { $$ = std::set<std::string> { $1 }; }
          | covariates "," covariate { $$ = $1; $$.insert($3); };

covariate: "identifier" { $$ = $1; };

regression_eq: regressand "=" regression_expr { driver.get_expr($1).swap($3); }

regression_expr: regressor { $$ = spec_parser::var($1); }
               | "numeric" { $$ = spec_parser::num<spec_parser::VariablePtr>(std::stod($1)); }
               | "(" regression_expr ")" { $$ = $2; }
               | "log" regression_expr { $$ = spec_parser::log($2); }
               | "exp" regression_expr { $$ = spec_parser::exp($2); }
               | regression_expr "+" regression_expr { $$ = $1 + $3; }
               | regression_expr "-" regression_expr { $$ = $1 - $3; }
               | regression_expr "*" regression_expr { $$ = $1 * $3; };
               | regression_expr "/" regression_expr { $$ = $1 / $3; };

regressor: "identifier" "(" covariates ")" { $$ = driver.get_variable($1, $3); }

distr_spec: regressor "~" distr { $1->set_distribution($3); };

/* TODO: perhaps introduce special tokens for distribution names */
distr: "identifier" "(" distr_args ")" { $$ = spec_parser::Distribution($1, $3); }

distr_args: %empty { $$ = std::vector<std::string> {}; }
          | distr_arg { $$ = std::vector<std::string> { $1 }; }
          | distr_args "," distr_arg { $$ = $1; $$.push_back($3); };

distr_arg: "numeric" { $$ = $1; }
         | regressor { $$ = $1->full_id(); };
%%

void yy::parser::error (const location_type& l, const std::string& m)
{
  driver.error (l, m);
}
