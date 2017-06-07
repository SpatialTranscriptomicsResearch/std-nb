#ifndef COUNTS_HPP
#define COUNTS_HPP
#include <memory>
#include <string>
#include <vector>
#include "types.hpp"

struct Counts {
  Counts(const std::string &path, bool transpose,
         const std::string &separator = "\t");
  std::string path;
  std::vector<std::string> row_names;
  std::vector<std::string> col_names;
  std::shared_ptr<STD::Matrix> matrix;
  /**
   * Function assumes the column names are of the form "AxB" with A and B
   * positive integers. It then computes the matrix of pairwise squared
   * Euclidean distances.
   */
  STD::Matrix compute_distances() const;
  STD::Matrix parse_coords() const;
  size_t num_genes() const;
  size_t num_samples() const;
  size_t operator()(size_t g, size_t t) const;
  // size_t &operator()(size_t g, size_t t);
};

std::vector<Counts> load_data(const std::vector<std::string> &paths,
                              bool intersect, size_t top, size_t bottom,
                              bool discard_empty, bool transpose);

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
size_t sum_cols(const std::vector<Counts> &c);
size_t max_row_number(const std::vector<Counts> &c);

#endif
