#ifndef IO_HPP
#define IO_HPP
#include <exception>
#include <iostream>
#include <string>
#include <vector>
#include "compression_mode.hpp"
#include "types.hpp"

template <typename V>
void write_vector(const V &v, const std::string &path, CompressionMode mode,
                  const std::vector<std::string> &names
                  = std::vector<std::string>(),
                  const std::string &separator = "\t") {
  size_t X = v.size();

  bool names_given = not names.empty();

  if (names_given) {
    if (names.size() != X)
      throw(std::runtime_error(
          "Error: length of names (" + std::to_string(names.size())
          + ") does not match length of vector (" + std::to_string(X) + ")."));
  }

  write_file(path, mode, [&](std::ostream &ofs) {
    for (size_t x = 0; x < X; ++x)
      ofs << (names_given ? names[x] + separator : "") << v[x] << '\n';
  });
}

template <typename M>
void write_matrix(const M &m, const std::string &path, CompressionMode mode,
                  const std::vector<std::string> &row_names
                  = std::vector<std::string>(),
                  const std::vector<std::string> &col_names
                  = std::vector<std::string>(),
                  std::vector<size_t> col_order = std::vector<size_t>(),
                  std::vector<size_t> row_order = std::vector<size_t>(),
                  const std::string &separator = "\t") {
  size_t X = m.rows();
  size_t Y = m.cols();

  if (row_order.empty())
    for (size_t x = 0; x < X; ++x)
      row_order.push_back(x);

  if (col_order.empty())
    for (size_t y = 0; y < Y; ++y)
      col_order.push_back(y);

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

  if (col_order.size() != Y)
    throw(std::runtime_error("Error: length of column order index vector ("
                             + std::to_string(col_order.size())
                             + ") does not match number of cols ("
                             + std::to_string(Y) + ")."));

  write_file(path, mode, [&](std::ostream &ofs) {
    if (col_names_given) {
      for (size_t y = 0; y < Y; ++y)
        // TODO decide if the old behavior might be preferable
        // ofs << separator << col_names[col_order[y]];
        ofs << separator << col_names[y];
      ofs << '\n';
    }
    for (size_t x = 0; x < X; ++x) {
      if (row_names_given)
        ofs << row_names[x] + separator;
      for (size_t y = 0; y < Y; ++y)
        ofs << (y != 0 ? separator : "") << m(row_order[x], col_order[y]);
      ofs << '\n';
    }
  });
}

STD::Matrix read_matrix(std::istream &os, const std::string &separator);

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

STD::Matrix read_floats(std::istream &ifs, const std::string &separator,
                        std::vector<std::string> &row_names,
                        std::vector<std::string> &col_names);

void print_matrix_head(std::ostream &os, const STD::Matrix &m,
                       const std::string &label = "", size_t n = 10);

template <typename V>
void print_vector_head(std::ostream &os, const V &v,
                       const std::string &label = "", size_t n = 10) {
  if (label != "")
    os << label << "\n";
  size_t X = v.rows();
  for (size_t x = 0; x < std::min(n, X); ++x)
    os << (x > 0 ? "\t" : "") << v[x];
  os << "\n";

  size_t zeros = 0;
  for (auto &x : v)
    if (x == 0)
      zeros++;

  os << "There are " << zeros << " zeros";
  if (label != "")
    os << " in " << label;
  os << ". This corresponds to " << (100.0 * zeros / X) << "%."
     << "\n";

  // os << Stats::summary(v) << "\n"; // TODO reactivate this
}

#endif
