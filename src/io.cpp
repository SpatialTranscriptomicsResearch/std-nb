#include "io.hpp"
#include <algorithm>
#include <boost/tokenizer.hpp>
#include <exception>
#include <fstream>
#include <regex>
#include <string>
#include <unordered_map>
#include "compression.hpp"

using namespace std;
using Int = STD::Int;
using Float = STD::Float;
using Matrix = STD::Matrix;

Matrix read_matrix(istream &is, const string &separator) {
  // TODO improve / factor / simplify implementation
  vector<string> row_names, col_names;
  Matrix m = read_floats(is, separator, row_names, col_names);
  return m;
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
                   vector<string> &row_names, vector<string> &col_names) {
  using tokenizer = boost::tokenizer<boost::char_separator<char>>;
  boost::char_separator<char> sep(separator.c_str());
  vector<vector<Float>> m;

  string line;

  getline(ifs, line);
  tokenizer tok(line, sep);
  for (auto token : tok)
    col_names.push_back(token.c_str());

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

  size_t ncol = matrix.cols();

  if (ncol == col_names.size())
    return matrix;
  else if (ncol == col_names.size() - 1) {
    vector<string> new_col_names(begin(col_names) + 1, end(col_names));
    col_names = new_col_names;
    return matrix;
  } else
    throw std::runtime_error(
        string()
        + "Mismatch between number of columns and number of column labels.\n"
        + "There are " + std::to_string(ncol) + " columns and "
        + std::to_string(col_names.size()) + " column names.");
}

void print_matrix_head(ostream &os, const Matrix &m, const std::string &label,
                       size_t n) {
  if (label != "")
    os << label << std::endl;
  size_t X = m.rows();
  size_t Y = m.cols();
  for (size_t x = 0; x < std::min<size_t>(X, n); ++x) {
    for (size_t y = 0; y < Y; ++y)
      os << (y > 0 ? "\t" : "") << m(x, y);
    os << std::endl;
  }

  if (label != "")
    os << label << " ";
  os << "column sums" << std::endl;

  size_t zeros = 0;
  for (size_t y = 0; y < Y; ++y) {
    double sum = 0;
    for (size_t x = 0; x < X; ++x) {
      if (m(x, y) == 0)
        zeros++;
      sum += m(x, y);
    }
    os << (y > 0 ? "\t" : "") << sum;
  }
  os << std::endl;

  os << "There are " << zeros << " zeros";
  if (label != "")
    os << " in " << label;
  os << ". This corresponds to " << (100.0 * zeros / X / Y) << "%."
     << std::endl;
}
