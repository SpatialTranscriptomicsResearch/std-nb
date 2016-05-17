#include <algorithm>
#include <fstream>
#include <exception>
#include <unordered_map>
#include <boost/tokenizer.hpp>
#include "compression.hpp"
#include "io.hpp"

using namespace std;
using Int = FactorAnalysis::Int;
using Float = FactorAnalysis::Float;
using Matrix = FactorAnalysis::Matrix;
using IMatrix = FactorAnalysis::IMatrix;
using Vector = FactorAnalysis::Vector;

Matrix read_matrix(istream &is, const string &separator, const string &label) {
  // TODO improve / factor / simplify implementation
  vector<string> row_names, col_names;
  Matrix m = read_floats(is, separator, row_names, col_names, label);
  return m;
}

IMatrix read_imatrix(istream &is, const string &separator,
                     const string &label) {
  // TODO improve / factor / simplify implementation
  vector<string> row_names, col_names;
  IMatrix m = read_counts(is, separator, row_names, col_names, label);
  return m;
}

IMatrix vec_of_vec_to_multi_array(const vector<vector<Int>> &v) {
  const size_t s1 = v.size();
  const size_t s2 = v[0].size();
  IMatrix A(s1, s2);
  for (size_t i = 0; i < s1; ++i)
    for (size_t j = 0; j < s2; ++j)
      A(i, j) = v[i][j];
  return A;
}

Matrix vec_of_vec_to_multi_array_float(const vector<vector<Float>> &v) {
  const size_t s1 = v.size();
  const size_t s2 = v[0].size();
  Matrix A(s1, s2);
  for (size_t i = 0; i < s1; ++i)
    for (size_t j = 0; j < s2; ++j)
      A(i, j) = v[i][j];
  return A;
}

Matrix read_floats(istream &ifs, const string &separator,
                   vector<string> &row_names, vector<string> &col_names,
                   const string &label) {
  using tokenizer = boost::tokenizer<boost::char_separator<char>>;
  boost::char_separator<char> sep(separator.c_str());
  vector<vector<Float>> m;

  string line;

  getline(ifs, line);
  tokenizer tok(line, sep);
  for (auto token : tok)
    col_names.push_back((label.empty() ? "" : label + " ") + token.c_str());

  while (getline(ifs, line)) {
    tok = tokenizer(line, sep);
    size_t col = 0;
    vector<Float> v;
    for (auto token : tok)
      if (col++ == 0)
        row_names.push_back(token);
      else
        v.push_back(atof(token.c_str()));
    m.push_back(v);
  }

  auto matrix = vec_of_vec_to_multi_array_float(m);

  size_t ncol = matrix.n_cols;

  if (ncol == col_names.size())
    return matrix;
  else if (ncol == col_names.size() - 1) {
    vector<string> new_col_names(begin(col_names) + 1, end(col_names));
    col_names = new_col_names;
    return matrix;
  } else
    throw std::runtime_error(
        "Mismatch between number of columns and number of column labels.");
}

IMatrix read_counts(istream &ifs, const string &separator,
                    vector<string> &row_names, vector<string> &col_names,
                    const string &label) {
  using tokenizer = boost::tokenizer<boost::char_separator<char>>;
  boost::char_separator<char> sep(separator.c_str());
  vector<vector<Int>> m;

  string line;

  getline(ifs, line);
  tokenizer tok(line, sep);
  for (auto token : tok)
    col_names.push_back((label.empty() ? "" : label + " ") + token.c_str());

  while (getline(ifs, line)) {
    tok = tokenizer(line, sep);
    size_t col = 0;
    vector<Int> v;
    for (auto token : tok)
      if (col++ == 0)
        row_names.push_back(token);
      else
        v.push_back(atoi(token.c_str()));
    m.push_back(v);
  }

  auto matrix = vec_of_vec_to_multi_array(m);

  size_t ncol = matrix.n_cols;

  if (ncol == col_names.size())
    return matrix;
  else if (ncol == col_names.size() - 1) {
    vector<string> new_col_names(begin(col_names) + 1, end(col_names));
    col_names = new_col_names;
    return matrix;
  } else
    throw std::runtime_error(
        "Mismatch between number of columns and number of column labels.");
}
