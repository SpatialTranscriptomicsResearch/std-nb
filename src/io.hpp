#ifndef IO_HPP
#define IO_HPP
#include <string>
#include <vector>
#include <iostream>
#include "FactorAnalysis.hpp"

void write_vector(const FactorAnalysis::Vector &v, const std::string &path,
                  const std::vector<std::string> &names
                  = std::vector<std::string>());
void write_matrix(const FactorAnalysis::Matrix &m, const std::string &path,
                  const std::vector<std::string> &row_names
                  = std::vector<std::string>(),
                  const std::vector<std::string> &col_names
                  = std::vector<std::string>());

FactorAnalysis::Matrix read_matrix(std::istream &os);
FactorAnalysis::Vector read_vector(std::istream &os);

FactorAnalysis::IMatrix read_counts(std::istream &ifs,
                                    const std::string &separator,
                                    std::vector<std::string> &row_names,
                                    std::vector<std::string> &col_names,
                                    const std::string &label);

#endif
