#ifndef IO_HPP
#define IO_HPP
#include <string>
#include <vector>
#include "FactorAnalysis.hpp"

void write_vector(const FactorAnalysis::Vector &v, const std::string &path,
                  const std::vector<std::string> &names
                  = std::vector<std::string>());
void write_matrix(const FactorAnalysis::Matrix &m, const std::string &path,
                  const std::vector<std::string> &row_names
                  = std::vector<std::string>(),
                  const std::vector<std::string> &col_names
                  = std::vector<std::string>());

#endif
