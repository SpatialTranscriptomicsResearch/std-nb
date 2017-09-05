#include <exception>
#include <sstream>

#include "driver.hpp"
#include "parser.tab.hpp"

using namespace spec_parser;

Driver::Driver()
    : trace_scanning(false)
    , trace_parsing(false)
{
}

Driver::~Driver() {}

static std::string location_to_string(const yy::location& l)
{
  std::stringstream ss;
  ss << l;
  return ss.str();
}

void Driver::error(const yy::location& l, const std::string& m)
{
  throw std::runtime_error(
      "'" + cur_line + "' (" + location_to_string(l) + "): " + m);
}

void Driver::error(const std::string& m)
{
  throw std::runtime_error("'" + cur_line + "' : " + m);
}
