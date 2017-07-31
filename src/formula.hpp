#ifndef FORMULA_HPP
#define FORMULA_HPP

#include <iostream>
#include <string>
#include <vector>

struct Formula {
  using Term = std::vector<std::string>;
  using Terms = std::vector<Term>;
  Formula(const std::string &str = "");

  // TODO : Consider emitting syntactic error on space inside of covariates
  void from_string(const std::string &str);

  std::string to_string() const;
  void add_section_to_spots();

  Terms terms;
};

std::ostream &operator<<(std::ostream &os, const Formula &formula);
std::istream &operator>>(std::istream &is, Formula &formula);

Formula DefaultRateFormula();
Formula DefaultVarianceFormula();
bool check_formula(const Formula &formula);

#endif
