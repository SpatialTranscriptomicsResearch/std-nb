#ifndef DISTRIBUTION_HPP
#define DISTRIBUTION_HPP

#include <exception>
#include <unordered_map>

#include "coefficient.hpp"

namespace spec_parser {

struct Distribution {
  // TODO: consider using separate Distribution type to increase modularity
  using Type = Coefficient::Distribution;

  static Type stodistr(const std::string& s);
  static std::string distrtos(const Type& d);

  Type type;
  std::vector<std::string> arguments;

  Distribution();
  Distribution(const std::string& type, const std::vector<std::string> arguments);
};

} // namespace spec_parser

#endif // DISTRIBUTION_HPP
