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

namespace DesignNS {
const std::string path_label = "path";
const std::string name_label = "name";
const std::string section_label = "section";
const std::string coordsys_label = "coordsys";
const std::string unit_label = "1";
};

struct Design {
  void from_string(const std::string &str);
  void from_stream(std::istream &is);
  std::string to_string() const;
  Specifications dataset_specifications;
  Covariates covariates;
  void add_covariate_section();
  void add_covariate_coordsys();
  void add_covariate_unit();
  bool is_reserved_name(const std::string &s) const;
  void add_dataset_specification(const std::string &s);
  std::vector<size_t> determine_covariate_idxs(
      const std::vector<std::string> &term) const;
  std::vector<size_t> get_covariate_value_idxs(size_t e, const std::vector<size_t> &covariate_idxs) const;
};

std::istream &operator>>(std::istream &is, Design &design);
std::ostream &operator<<(std::ostream &os, const Design &design);

#endif
