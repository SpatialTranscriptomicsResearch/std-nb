#include "driver.hpp"
#include "parser.tab.hh"

Driver::Driver()
    : trace_scanning(false)
    , trace_parsing(false)
{
}

Driver::~Driver() {}

int Driver::parse(const std::string& f)
{
  file = f;
  scan_begin();
  yy::parser parser(*this);
  parser.set_debug_level(trace_parsing);
  int res = parser.parse();
  scan_end();
  return res;
}

void Driver::error(const yy::location& l, const std::string& m)
{
  std::cerr << l << ": " << m << std::endl;
}

void Driver::error(const std::string& m) { std::cerr << m << std::endl; }
