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
#include <set>

#include "Distribution.hpp"
#include "RegressionEquation.hpp"
#include "RandomVariable.hpp"

class Driver;
}

// The parsing context.
%param { Driver& driver }

%locations

%define parse.trace
%define parse.error verbose

%code
{
#include "driver.hpp"
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

%type <std::string> covariate;
%type <std::set<std::string>> covariates;

%type <Distribution> distr;
%type <std::string> distr_arg;
%type <std::vector<std::string>> distr_args;

%type <std::string> regressand;
%type <RandomVariable> regressor;

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

regressand: "identifier" { $$ = $1; };

formula_expr: covariate
            | "(" formula_expr ")"
            | formula_expr "+" formula_expr
            | formula_expr ":" formula_expr
            | formula_expr "*" formula_expr
            | formula_expr "^" "number";

covariates: %empty { $$ = std::set<std::string> {}; }
          | covariate { $$ = std::set<std::string> { $1 }; }
          | covariates "," covariate { $$ = $1; $$.insert($3); };

covariate: "identifier" { $$ = $1; };

regression_eq: regressand "=" regression_eq { driver.regression_equations[$1] = $3; }

regression_eq: regressor { $$ = RegressionEquation($1.full_id()); }
             | regression_eq "*" regression_eq { $$ = $1 * $3; };

regressor: "identifier" "(" covariates ")" { $$ = RandomVariable($1, $3); };

distr_spec: regressor "~" distr { $1.distribution = $3; driver.random_variables[$1.full_id()] = $1; };

/* TODO: perhaps introduce special tokens for distribution names */
distr: "identifier" "(" distr_args ")" { $$ = Distribution($1, $3); }

distr_args: %empty { $$ = std::vector<std::string> {}; }
          | distr_arg { $$ = std::vector<std::string> { $1 }; }
          | distr_args "," distr_arg { $$ = $1; $$.push_back($3); };

distr_arg: "number" { $$ = std::to_string($1); }
         | regressor { $$ = $1.full_id(); };
%%

void yy::parser::error (const location_type& l, const std::string& m)
{
  driver.error (l, m);
}
