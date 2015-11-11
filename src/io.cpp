#include <fstream>
#include <boost/tokenizer.hpp>
#include "io.hpp"

using namespace std;
using PFA = PoissonFactorAnalysis;
using Int = PFA::Int;

vector<vector<Int>> read_matrix_vec_of_vec(const string &path) {
  vector<vector<Int>> m;
  ifstream ifs(path);
  string line;
  while(getline(ifs, line)) {
    boost::tokenizer<> tok(line);
    vector<Int> v;
    for(auto token: tok)
      v.push_back(atoi(token.c_str()));
    m.push_back(v);
  }

  return m;
}

PFA::IMatrix vec_of_vec_to_multi_array(const vector<vector<Int>> &v) {
  const size_t s1 = v.size();
  const size_t s2 = v[0].size();
  using index = PoissonFactorAnalysis::IMatrix::index;
  PFA::IMatrix A(boost::extents[s1][s2]);
  for(size_t i = 0; i < s1; ++i)
    for(size_t j = 0; j < s2; ++j)
      A[i][j] = v[i][j];
  return A;
}

PFA::IMatrix read_matrix(const string &path) {
  return vec_of_vec_to_multi_array(read_matrix_vec_of_vec(path));
}

void write_vector(const PFA::Vector &v, const string &path) {
  ofstream ofs(path);
  auto shape = v.shape();
  size_t X = shape[0];
  for(size_t x = 0; x < X; ++x)
    ofs << v[x] << endl;
}

void write_matrix(const PFA::Matrix &m, const string &path) {
  ofstream ofs(path);
  auto shape = m.shape();
  size_t X = shape[0];
  size_t Y = shape[1];
  for(size_t x = 0; x < X; ++x) {
    for(size_t y = 0; y < Y; ++y)
      ofs << (y != 0 ? " " : "") << m[x][y];
    ofs << endl;
  }
}
