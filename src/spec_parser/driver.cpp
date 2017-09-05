#include <exception>
#include <sstream>

#include "driver.hpp"
#include "parser.tab.hpp"

using namespace spec_parser;

ParseError::ParseError(const std::string& _line, const yy::location& _where,
    const std::string& what)
    : runtime_error(what)
    , line(_line)
    , where(_where)
{
}

Driver::~Driver() {}

Driver::Driver()
    : trace_scanning(false)
    , trace_parsing(false)
{
}

void Driver::error(const yy::location& l, const std::string& m)
{
  throw ParseError(cur_line, l, m);
}
