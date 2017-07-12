#ifndef FORMULA_HPP
#define FORMULA_HPP

#include <iostream>
#include <string>
#include <vector>

struct Formula {
  Formula(const std::string &str = "");

  // TODO : Consider emitting syntactic error on space inside of covariates
  void from_string(const std::string &str);

  std::string to_string() const;

  std::vector<std::vector<std::string>> formula;
};

std::ostream &operator<<(std::ostream &os, const Formula &formula);
std::istream &operator>>(std::istream &is, Formula &formula);

Formula DefaultFormula();
bool check_formula(const Formula &formula);

#endif
