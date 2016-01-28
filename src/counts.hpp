#ifndef COUNTS_HPP
#define COUNTS_HPP
#include <string>
#include <vector>
#include "FactorAnalysis.hpp"

struct Counts {
  Counts(const std::string &path, const std::string &label = "",
         const std::string &separator = "\t");
  Counts(const std::vector<std::string> &rnames,
         const std::vector<std::string> &cnames,
         const FactorAnalysis::IMatrix &cnts, const std::vector<size_t> &exps);
  std::vector<std::string> row_names;
  std::vector<std::string> col_names;
  FactorAnalysis::IMatrix counts;
  std::vector<size_t> experiments;
  Counts operator+(const Counts &other) const;
  Counts operator*(const Counts &other) const;
  Counts &operator=(const Counts &other);
  void select_top(size_t n);
};

#endif
