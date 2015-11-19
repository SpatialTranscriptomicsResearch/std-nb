#ifndef IO_HPP
#define IO_HPP
#include <string>
#include <vector>
#include "PoissonFactorAnalysis.hpp"

std::vector<std::vector<PoissonFactorAnalysis::Int>> read_matrix_vec_of_vec(const std::string &path, size_t skip_lines=1, size_t skip_col=1, const std::string &separator="\t");
PoissonFactorAnalysis::IMatrix vec_of_vec_to_multi_array(const std::vector<std::vector<PoissonFactorAnalysis::Int>> &v);
PoissonFactorAnalysis::IMatrix read_matrix(const std::string &path);
void write_vector(const PoissonFactorAnalysis::Vector &v, const std::string &path);
void write_matrix(const PoissonFactorAnalysis::Matrix &m, const std::string &path);

#endif
