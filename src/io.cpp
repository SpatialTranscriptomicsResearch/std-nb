#include <algorithm>
#include <fstream>
#include <exception>
#include <unordered_map>
#include <boost/tokenizer.hpp>
#include "compression.hpp"
#include "io.hpp"

using namespace std;
using PFA = PoissonFactorAnalysis;
using Int = PFA::Int;

PFA::IMatrix vec_of_vec_to_multi_array(const vector<vector<Int>> &v) {
  const size_t s1 = v.size();
  const size_t s2 = v[0].size();
  using index = PoissonFactorAnalysis::IMatrix::index;
  PFA::IMatrix A(boost::extents[s1][s2]);
  for (size_t i = 0; i < s1; ++i)
    for (size_t j = 0; j < s2; ++j) A[i][j] = v[i][j];
  return A;
}

PFA::IMatrix read_counts(istream &ifs, const string &separator,
                         vector<string> &row_names, vector<string> &col_names) {
  using tokenizer = boost::tokenizer<boost::char_separator<char>>;
  boost::char_separator<char> sep(separator.c_str());
  vector<vector<Int>> m;

  string line;

  size_t col = 0;
  getline(ifs, line);
  tokenizer tok(line, sep);
  for (auto token : tok)
    if (col++ > 0) col_names.push_back(token.c_str());

  while (getline(ifs, line)) {
    tok = tokenizer(line, sep);
    col = 0;
    vector<Int> v;
    for (auto token : tok)
      if (col++ == 0)
        row_names.push_back(token);
      else
        v.push_back(atoi(token.c_str()));
    m.push_back(v);
  }

  return vec_of_vec_to_multi_array(m);
}

Counts::Counts(const string &path, const string &separator)
    : row_names(),
      col_names(),
      counts(parse_file<PFA::IMatrix>(path, read_counts, separator, row_names,
                                      col_names)) {}

Counts::Counts(const vector<string> &rnames, const vector<string> &cnames,
               const PFA::IMatrix &cnts)
    : row_names(rnames), col_names(cnames), counts(cnts) {}

template <typename T>
unordered_map<T, size_t> generate_index_map(const vector<T> &v) {
  const size_t n = v.size();
  unordered_map<T, size_t> m;
  for (size_t i = 0; i < n; ++i) m[v[i]] = i;
  return m;
}

Counts Counts::operator+(const Counts &other) const {
  vector<string> rnames;
  set_union(begin(row_names), end(row_names), begin(other.row_names),
            end(other.row_names), begin(rnames));

  vector<string> cnames = col_names;
  for (auto &name : other.col_names) cnames.push_back(name);

  auto m1 = generate_index_map(row_names);
  auto m2 = generate_index_map(other.row_names);

  const size_t nrow = rnames.size();
  const size_t ncol = cnames.size();

  const size_t ncol1 = col_names.size();

  PFA::IMatrix cnt(boost::extents[nrow][ncol]);
  size_t col_idx = 0;
  for (; col_idx < ncol1; ++col_idx) {
    size_t row_idx = 0;
    for (auto &name : rnames) {
      auto iter = m1.find(name);
      if (iter != end(m1))
        cnt[row_idx][col_idx] = counts[iter->second][col_idx];
      row_idx++;
    }
  }
  for (; col_idx < ncol; ++col_idx) {
    size_t row_idx = 0;
    for (auto &name : rnames) {
      auto iter = m2.find(name);
      if (iter != end(m2))
        cnt[row_idx][col_idx] = other.counts[iter->second][col_idx - ncol1];
      row_idx++;
    }
  }
  return {rnames, cnames, cnt};
}

Counts Counts::operator*(const Counts &other) const {
  vector<string> rnames;
  set_intersection(begin(row_names), end(row_names), begin(other.row_names),
                   end(other.row_names), begin(rnames));

  vector<string> cnames = col_names;
  for (auto &name : other.col_names) cnames.push_back(name);

  auto m1 = generate_index_map(row_names);
  auto m2 = generate_index_map(other.row_names);

  const size_t nrow = rnames.size();
  const size_t ncol = cnames.size();

  const size_t ncol1 = col_names.size();

  PFA::IMatrix cnt(boost::extents[nrow][ncol]);
  size_t col_idx = 0;
  for (; col_idx < ncol1; ++col_idx) {
    size_t row_idx = 0;
    for (auto &name : rnames) {
      auto iter = m1.find(name);
      if (iter != end(m1))
        cnt[row_idx][col_idx] = counts[iter->second][col_idx];
      row_idx++;
    }
  }
  for (; col_idx < ncol; ++col_idx) {
    size_t row_idx = 0;
    for (auto &name : rnames) {
      auto iter = m2.find(name);
      if (iter != end(m2))
        cnt[row_idx][col_idx] = other.counts[iter->second][col_idx - ncol1];
      row_idx++;
    }
  }
  return {rnames, cnames, cnt};
}

void write_vector(const PFA::Vector &v, const string &path,
                  const vector<string> &names) {
  auto shape = v.shape();
  size_t X = shape[0];

  bool names_given = not names.empty();

  if (names_given) {
    if (names.size() != X)
      throw(runtime_error("Error: length of names (" + to_string(names.size()) +
                          ") does not match length of vector (" + to_string(X) +
                          ")."));
  }

  ofstream ofs(path);
  for (size_t x = 0; x < X; ++x)
    ofs << (names_given ? names[x] + "\t" : "") << v[x] << endl;
}

void write_matrix(const PFA::Matrix &m, const string &path,
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
          "Error: length of row names (" + to_string(row_names.size()) +
          ") does not match number of rows (" + to_string(X) + ")."));
  }

  if (col_names_given) {
    if (col_names.size() != Y)
      throw(runtime_error(
          "Error: length of col names (" + to_string(col_names.size()) +
          ") does not match number of cols (" + to_string(Y) + ")."));
  }

  ofstream ofs(path);
  if (col_names_given) {
    for (size_t y = 0; y < Y; ++y) ofs << "\t" << col_names[y];
    ofs << endl;
  }
  for (size_t x = 0; x < X; ++x) {
    if (row_names_given) ofs << row_names[x] + "\t";
    for (size_t y = 0; y < Y; ++y) ofs << (y != 0 ? "\t" : "") << m[x][y];
    ofs << endl;
  }
}
