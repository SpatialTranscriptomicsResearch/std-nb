#ifndef COVARIATE_HPP
#define COVARIATE_HPP

#include <iostream>

struct Covariate {
  enum struct Kind { Gene, Type, Section, Spot, Custom };
  Kind kind;
  std::string label;
  Covariate(Kind &kind_, const std::string &label_);
};

std::istream &operator>>(std::istream &is, Covariate::Kind &kind);

std::string to_string(const Covariate::Kind &kind);

std::ostream &operator<<(std::ostream &os, const Covariate::Kind &kind);

#endif
