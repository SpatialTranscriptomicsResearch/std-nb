#ifndef DESIGN_HPP
#define DESIGN_HPP

#include <iostream>
#include <stdexcept>
#include "covariate.hpp"

namespace Exception {
namespace Design {
struct NoPathColumn : public std::runtime_error {
  NoPathColumn()
      : std::runtime_error("Error: no path column in specification."){};
};
struct MultiplePathColumns : public std::runtime_error {
  MultiplePathColumns()
      : std::runtime_error("Error: multiple path columns in specification."){};
};
struct MultipleNameColumns : public std::runtime_error {
  MultipleNameColumns()
      : std::runtime_error("Error: multiple name columns in specification."){};
};
struct RepeatedColumnName : public std::runtime_error {
  RepeatedColumnName(const std::string &str)
      : std::runtime_error("Error: repeated column in specification: '" + str
                           + "'."){};
};
}
}

struct Specification {
  std::string path;
  std::string name;
  std::vector<size_t> covariate_values;
};

using Design = std::vector<Specification>;

void read_design(std::istream &is, Design &design, Covariates &covariates);

#endif
