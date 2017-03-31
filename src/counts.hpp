#ifndef COUNTS_HPP
#define COUNTS_HPP
#include <string>
#include <vector>
#include "types.hpp"

struct Counts {
  Counts(const std::string &path, const std::string &separator = "\t");
  std::string path;
  std::vector<std::string> row_names;
  std::vector<std::string> col_names;
  STD::IMatrix counts;  // TODO rename to matrix
  /**
   * Function assumes the column names are of the form "AxB" with A and B
   * positive integers. It then computes the matrix of pairwise squared
   * Euclidean distances.
   */
  STD::Matrix compute_distances() const;
  STD::Matrix parse_coords() const;
};

std::vector<Counts> load_data(const std::vector<std::string> &paths,
                              bool intersect, size_t top, bool discard_empty);

void gene_union(std::vector<Counts> &counts_v);
void gene_intersection(std::vector<Counts> &counts_v);

template <typename T>
T apply_kernel(T m, double sigma) {
  return 1 / sqrt(2 * M_PI) / sigma * exp(-m / (2 * sigma * sigma));
}

// compute squared Euclidean distances between all pairs of rows of a and b
STD::Matrix compute_sq_distances(const STD::Matrix &a, const STD::Matrix &b);

STD::Matrix row_normalize(STD::Matrix m_);

size_t sum_rows(const std::vector<Counts> &c);
size_t max_row_number(const std::vector<Counts> &c);

#endif
