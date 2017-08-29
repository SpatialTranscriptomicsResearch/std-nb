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
#include <iostream>

#include "RegressionEquation.hpp"
#include "RandomVariable.hpp"

class Driver;
}

// The parsing context.
%param { Driver& driver }

%locations

%initial-action
{
  // Initialize the initial location.
  @$.begin.filename = @$.end.filename = &driver.file;
};

%define parse.trace
%define parse.error verbose

%code
{
#include "driver.hpp"

std::string unit_covariate(int number) {
  if (number != 1) {
    throw std::logic_error("covariate must be identifier or \"1\"");
  }
  return std::to_string(number);
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
  EXP     "^"
  LPAREN  "("
  RPAREN  ")"
;
%token <std::string> IDENTIFIER "identifier"
%token <int> NUMBER "number"

%type <RegressionEquation> regression_eq;
%type <RandomVariable> distr;
%type <std::string> regressand;
%type <std::pair<std::string, std::vector<std::string>>> regressor;

%%
%left "+" "-";
%left "^";
%left "*";
%left ":";

%start program;

program: statements;

statements: %empty | statements statement;

statement: regression_formula
         | regression_eq
         | distr_spec;

regression_formula: regressand ":=" formula_expr;

regressand: "identifier";

formula_expr: covariate
            | "(" formula_expr ")"
            | formula_expr "+" formula_expr
            | formula_expr ":" formula_expr
            | formula_expr "*" formula_expr
            | formula_expr "^" "number";

/* TODO: could allow empty sets of arguments */
covariates: covariate
          | covariates "," covariate;

covariate: "identifier"
         | "number";

regression_eq: regressand "=" regression_eq { driver.regression_equation[$1] = $3; }

regression_eq: regressor { $$ = RegressionEquation($1); }
               | regression_eq "*" regression_eq { $$ = $1 * $3; };

regressor: "identifier" "(" covariates ")";

distr_spec: regressor "~" distr { driver.random_variables[$1] = $3; };

/* TODO: perhaps introduce special tokens for distribution names */
distr: "identifier" "(" distr_args ")" { $$ = RandomVariable($1.first, $1.second, $3); };

distr_args: distr_arg
          | distr_args "," distr_arg;

distr_arg: "number"
         | regressor;
%%

void yy::parser::error (const location_type& l, const std::string& m)
{
  driver.error (l, m);
}
