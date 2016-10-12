#ifndef COUNTS_HPP
#define COUNTS_HPP
#include <string>
#include <vector>
#include "types.hpp"

struct Counts {
  Counts(const std::string &path, const std::string &separator = "\t");
  Counts(const std::vector<std::string> &rnames,
         const std::vector<std::string> &cnames,
         const PoissonFactorization::IMatrix &cnts,
         const std::vector<size_t> &exps,
         const std::vector<std::string> &exp_names);
  std::vector<std::string> row_names;
  std::vector<std::string> col_names;
  PoissonFactorization::IMatrix counts;
  std::vector<size_t> experiments;
  std::vector<std::string> experiment_names;
  Counts operator+(const Counts &other) const;
  Counts operator*(const Counts &other) const;
  Counts &operator=(const Counts &other);
  void select_top(size_t n);
  std::vector<Counts> split_experiments() const;
};

std::vector<Counts> load_data(const std::vector<std::string> &paths,
                              bool intersect, size_t top);

#endif
