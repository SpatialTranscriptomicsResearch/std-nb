#include <algorithm>
#include <fstream>
#include <exception>
#include <unordered_map>
#include <boost/tokenizer.hpp>
#include "compression.hpp"
#include "io.hpp"

using namespace std;
using Int = FactorAnalysis::Int;
using Matrix = FactorAnalysis::Matrix;
using IMatrix = FactorAnalysis::IMatrix;
using Vector = FactorAnalysis::Vector;

void write_vector(const Vector &v, const string &path,
                  const vector<string> &names) {
  auto shape = v.shape();
  size_t X = shape[0];

  bool names_given = not names.empty();

  if (names_given) {
    if (names.size() != X)
      throw(runtime_error("Error: length of names (" + to_string(names.size())
                          + ") does not match length of vector (" + to_string(X)
                          + ")."));
  }

  ofstream ofs(path);
  for (size_t x = 0; x < X; ++x)
    ofs << (names_given ? names[x] + "\t" : "") << v[x] << endl;
}

void write_matrix(const Matrix &m, const string &path,
                  const vector<string> &row_names,
                  const vector<string> &col_names) {
  auto shape = m.shape();
  size_t X = shape[0];
  size_t Y = shape[1];

  bool row_names_given = not row_names.empty();
  bool col_names_given = not col_names.empty();

  if (row_names_given) {
    if (row_names.size() != X)
      throw(runtime_error(
          "Error: length of row names (" + to_string(row_names.size())
          + ") does not match number of rows (" + to_string(X) + ")."));
  }

  if (col_names_given) {
    if (col_names.size() != Y)
      throw(runtime_error(
          "Error: length of col names (" + to_string(col_names.size())
          + ") does not match number of cols (" + to_string(Y) + ")."));
  }

  ofstream ofs(path);
  if (col_names_given) {
    for (size_t y = 0; y < Y; ++y)
      ofs << "\t" << col_names[y];
    ofs << endl;
  }
  for (size_t x = 0; x < X; ++x) {
    if (row_names_given)
      ofs << row_names[x] + "\t";
    for (size_t y = 0; y < Y; ++y)
      ofs << (y != 0 ? "\t" : "") << m[x][y];
    ofs << endl;
  }
}

Matrix read_matrix(istream &os) {
  // TODO write implementation
  Matrix m;
  return m;
}

Vector read_vector(istream &is) {
  // TODO write implementation
  string line;
  while(getline(is, line)) {
  }
  Vector v;
  return v;
}

IMatrix vec_of_vec_to_multi_array(const vector<vector<Int>> &v) {
  const size_t s1 = v.size();
  const size_t s2 = v[0].size();
  using index = IMatrix::index;
  IMatrix A(boost::extents[s1][s2]);
  for (size_t i = 0; i < s1; ++i)
    for (size_t j = 0; j < s2; ++j)
      A[i][j] = v[i][j];
  return A;
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

  auto shape = matrix.shape();
  size_t ncol = shape[1];

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
