#ifndef IO_HPP
#define IO_HPP
#include <string>
#include <exception>
#include <vector>
#include <iostream>
#include "FactorAnalysis.hpp"

template <typename V>
void write_vector(const V &v, const std::string &path,
                  const std::vector<std::string> &names
                  = std::vector<std::string>(),
                  const std::string &separator = "\t") {
  size_t X = v.n_rows;

  bool names_given = not names.empty();

  if (names_given) {
    if (names.size() != X)
      throw(std::runtime_error(
          "Error: length of names (" + std::to_string(names.size())
          + ") does not match length of vector (" + std::to_string(X) + ")."));
  }

  std::ofstream ofs(path);
  for (size_t x = 0; x < X; ++x)
    ofs << (names_given ? names[x] + separator : "") << v[x] << '\n';
}

template <typename M>
void write_matrix(const M &m, const std::string &path,
                  const std::vector<std::string> &row_names
                  = std::vector<std::string>(),
                  const std::vector<std::string> &col_names
                  = std::vector<std::string>(),
                  const std::string &separator = "\t") {
  size_t X = m.n_rows;
  size_t Y = m.n_cols;

  bool row_names_given = not row_names.empty();
  bool col_names_given = not col_names.empty();

  if (row_names_given) {
    if (row_names.size() != X)
      throw(std::runtime_error(
          "Error: length of row names (" + std::to_string(row_names.size())
          + ") does not match number of rows (" + std::to_string(X) + ")."));
  }

  if (col_names_given) {
    if (col_names.size() != Y)
      throw(std::runtime_error(
          "Error: length of col names (" + std::to_string(col_names.size())
          + ") does not match number of cols (" + std::to_string(Y) + ")."));
  }

  std::ofstream ofs(path);
  if (col_names_given) {
    for (size_t y = 0; y < Y; ++y)
      ofs << separator << col_names[y];
    ofs << '\n';
  }
  for (size_t x = 0; x < X; ++x) {
    if (row_names_given)
      ofs << row_names[x] + separator;
    for (size_t y = 0; y < Y; ++y)
      ofs << (y != 0 ? separator : "") << m(x, y);
    ofs << '\n';
  }
}

PoissonFactorization::Matrix read_matrix(std::istream &os,
                                   const std::string &separator,
                                   const std::string &label);

PoissonFactorization::IMatrix read_imatrix(std::istream &os,
                                     const std::string &separator,
                                     const std::string &label);

template <typename V>
V read_vector(std::istream &is, const std::string &separator) {
  // TODO improve / factor / simplify implementation
  std::string line;
  std::vector<double> v;
  while (getline(is, line)) {
    auto here = line.find(separator);
    if (here != std::string::npos) {
      line = line.substr(here + 1);
      v.push_back(atof(line.c_str()));
    }
  }
  V v_(v.size());
  for (size_t i = 0; i < v.size(); ++i)
    v_[i] = v[i];

  return v_;
}

PoissonFactorization::Matrix read_floats(std::istream &ifs,
                                   const std::string &separator,
                                   std::vector<std::string> &row_names,
                                   std::vector<std::string> &col_names,
                                   const std::string &label);

PoissonFactorization::IMatrix read_counts(std::istream &ifs,
                                    const std::string &separator,
                                    std::vector<std::string> &row_names,
                                    std::vector<std::string> &col_names,
                                    const std::string &label);

#endif
