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
struct ReservedCovariateName : public std::runtime_error {
  ReservedCovariateName(const std::string &str)
      : std::runtime_error("Error: reserved covariate name in specification: '"
                           + str + "'."){};
};
}
}

struct Specification {
  std::string path;
  std::string name;
  std::vector<size_t> covariate_values;
};

using Specifications = std::vector<Specification>;

struct Design {
  void from_string(const std::string &str);
  void from_stream(std::istream &is);
  std::string to_string() const;
  Specifications dataset_specifications;
  Covariates covariates;
  void add_covariate_section();
  void add_covariate_unit();
  bool is_reserved_name(const std::string &s) const;
  std::string path_label = "path";
  std::string name_label = "name";
  std::string section_label = "section";
  std::string unit_label = "1";
};

std::istream &operator>>(std::istream &is, Design &design);
std::ostream &operator<<(std::ostream &os, const Design &design);

#endif
