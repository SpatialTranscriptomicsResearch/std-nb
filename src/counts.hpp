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
  PoissonFactorization::IMatrix counts; // TODO rename to matrix
  std::vector<size_t> experiments;
  std::vector<std::string> experiment_names;
  Counts operator+(const Counts &other) const;
  Counts operator*(const Counts &other) const;
  Counts &operator=(const Counts &other);
  void select_top(size_t n);
  std::vector<Counts> split_experiments() const;
  /**
   * Function assumes the column names are of the form "AxB" with A and B
   * positive integers. It then computes the matrix of pairwise squared
   * Euclidean distances.
   */
  PoissonFactorization::Matrix compute_distances() const;
};

std::vector<Counts> load_data(const std::vector<std::string> &paths,
                              bool intersect, size_t top);

void gene_union(std::vector<Counts> &counts_v);
void gene_intersection(std::vector<Counts> &counts_v);

template <typename T>
T apply_kernel(T m, double sigma) {
  return 1 / sqrt(2 * M_PI) / sigma * exp(-m / (2 * sigma * sigma));
}

PoissonFactorization::Matrix row_normalize(
    PoissonFactorization::Matrix m_);

#endif
