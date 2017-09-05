%{
#include <cerrno>
#include <climits>
#include <cstdlib>
#include <string>
#include "driver.hpp"
#include "parser.tab.hpp"

// The location of the current token.
static yy::location loc;
%}
%option noyywrap nounput batch debug noinput
id      [a-zA-Z][a-zA-Z_0-9]*
numeric [0-9\.]+
blank   [ \t]
comment #[^\n]*

%{
  // Code run each time a pattern is matched.
  #define YY_USER_ACTION  loc.columns (yyleng);
%}

%%

%{
  // Code run each time yylex is called.
  loc.step ();
%}

":="       return yy::parser::make_ASSIGN(loc);
"="        return yy::parser::make_EQUAL(loc);
"~"        return yy::parser::make_TILDE(loc);
","        return yy::parser::make_COMMA(loc);
"-"        return yy::parser::make_MINUS(loc);
"+"        return yy::parser::make_PLUS(loc);
"*"        return yy::parser::make_STAR(loc);
":"        return yy::parser::make_COLON(loc);
"^"        return yy::parser::make_EXP(loc);
"("        return yy::parser::make_LPAREN(loc);
")"        return yy::parser::make_RPAREN(loc);

{blank}+   loc.step();
{comment}  loc.step();
[\n]+      loc.lines(yyleng); loc.step();
{id}       return yy::parser::make_IDENTIFIER(yytext, loc);
{numeric}  return yy::parser::make_NUMERIC(yytext, loc);
<<EOF>>    return yy::parser::make_END(loc);
.          Driver.error(loc, "invalid character");
%%

int spec_parser::Driver::parse(const std::string& s)
{
  cur_line = s;
  yy_flex_debug = trace_scanning;
  YY_BUFFER_STATE buf = yy_scan_string(cur_line.c_str());
  yy::parser parser(*this);
  parser.set_debug_level(trace_parsing);
  int res = parser.parse();
  yy_delete_buffer(buf);
  return res;
}

yy::location& spec_parser::Driver::location() const {
  return loc;
}
