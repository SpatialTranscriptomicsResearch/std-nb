#ifndef IO_HPP
#define IO_HPP
#include <string>
#include <vector>
#include "PoissonFactorAnalysis.hpp"

struct Counts {
  Counts(const std::string &path, const std::string &separator = "\t");
  Counts(const std::vector<std::string> &rnames,
         const std::vector<std::string> &cnames,
         const PoissonFactorAnalysis::IMatrix &cnts);
  std::vector<std::string> row_names;
  std::vector<std::string> col_names;
  PoissonFactorAnalysis::IMatrix counts;
  Counts operator+(const Counts &other) const;
  Counts operator*(const Counts &other) const;
  Counts &operator=(const Counts &other);
};

void write_vector(
    const PoissonFactorAnalysis::Vector &v, const std::string &path,
    const std::vector<std::string> &names = std::vector<std::string>());
void write_matrix(
    const PoissonFactorAnalysis::Matrix &m, const std::string &path,
    const std::vector<std::string> &row_names = std::vector<std::string>(),
    const std::vector<std::string> &col_names = std::vector<std::string>());

#endif
