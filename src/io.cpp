#include <algorithm>
#include <fstream>
#include <exception>
#include <unordered_map>
#include <boost/tokenizer.hpp>
#include "compression.hpp"
#include "io.hpp"

using namespace std;
using Matrix = FactorAnalysis::Matrix;
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
