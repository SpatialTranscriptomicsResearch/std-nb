#ifndef DISTRIBUTION_HPP
#define DISTRIBUTION_HPP

#include <exception>
#include <unordered_map>
#include <string>

#include "coefficient.hpp"

namespace spec_parser {

struct Distribution {
  // TODO: consider using separate Distribution type to increase modularity
  using Type = Coefficient::Type;

  static Type stodistr(const std::string& s);
  static std::string distrtos(const Type& d);

  Type type;
  std::vector<std::string> arguments;

  Distribution();
  Distribution(Type type, const std::vector<std::string> arguments);
  Distribution(const std::string& type, const std::vector<std::string> arguments);
  size_t desired_argument_number();
};

std::string to_string(const Distribution& d);

} // namespace spec_parser

#endif // DISTRIBUTION_HPP
