#ifndef IO_HPP
#define IO_HPP
#include <string>
#include <vector>
#include "FactorAnalysis.hpp"

struct Counts {
  Counts(const std::string &path, const std::string &label = "",
         const std::string &separator = "\t");
  Counts(const std::vector<std::string> &rnames,
         const std::vector<std::string> &cnames,
         const FactorAnalysis::IMatrix &cnts);
  std::vector<std::string> row_names;
  std::vector<std::string> col_names;
  FactorAnalysis::IMatrix counts;
  Counts operator+(const Counts &other) const;
  Counts operator*(const Counts &other) const;
  Counts &operator=(const Counts &other);
};

void write_vector(
    const FactorAnalysis::Vector &v, const std::string &path,
    const std::vector<std::string> &names = std::vector<std::string>());
void write_matrix(
    const FactorAnalysis::Matrix &m, const std::string &path,
    const std::vector<std::string> &row_names = std::vector<std::string>(),
    const std::vector<std::string> &col_names = std::vector<std::string>());

#endif
