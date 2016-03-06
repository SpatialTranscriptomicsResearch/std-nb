#ifndef IO_HPP
#define IO_HPP
#include <string>
#include <vector>
#include <iostream>
#include "FactorAnalysis.hpp"

void write_vector(const FactorAnalysis::Vector &v, const std::string &path,
                  const std::vector<std::string> &names
                  = std::vector<std::string>(),
                  const std::string &separator = "\t");
void write_matrix(const FactorAnalysis::Matrix &m, const std::string &path,
                  const std::vector<std::string> &row_names
                  = std::vector<std::string>(),
                  const std::vector<std::string> &col_names
                  = std::vector<std::string>(),
                  const std::string &separator = "\t");

FactorAnalysis::Matrix read_matrix(std::istream &os,
                                   const std::string &separator,
                                   const std::string &label);
FactorAnalysis::Vector read_vector(std::istream &os,
                                   const std::string &separator);

FactorAnalysis::Matrix read_floats(std::istream &ifs,
                                   const std::string &separator,
                                   std::vector<std::string> &row_names,
                                   std::vector<std::string> &col_names,
                                   const std::string &label);

FactorAnalysis::IMatrix read_counts(std::istream &ifs,
                                    const std::string &separator,
                                    std::vector<std::string> &row_names,
                                    std::vector<std::string> &col_names,
                                    const std::string &label);

#endif
