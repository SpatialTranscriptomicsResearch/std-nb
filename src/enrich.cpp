#include <iostream>
#include <string>
#include "compression.hpp"
#include "entropy.hpp"
#include "io.hpp"
#include "log.hpp"
#include "parallel.hpp"
#include "types.hpp"

using namespace std;
using namespace STD;

const string separator = "\t";

struct Data {
  vector<string> gene_names;
  vector<string> func_names;
  vector<string> type_names;
  Matrix expression;
  Matrix function;
  size_t T, G, F;
  Data(const string &exp_path, const string &func_path)
      : expression(parse_file<Matrix>(exp_path, read_floats, separator,
                                      gene_names, type_names, "")),
        function(parse_file<Matrix>(func_path, read_floats, separator,
                                    gene_names, func_names, "")),
        T(expression.n_cols),
        G(expression.n_rows),
        F(function.n_cols) {
    // intersect gene_ids
    // reduce to shared genes
    // set T, G, and F
    LOG(info) << "T = " << T;
    LOG(info) << "G = " << G;
    LOG(info) << "F = " << F;
    LOG(info) << "function.n_rows = " << function.n_rows;
  }
  Matrix compute_expectation() const {
    LOG(info) << "Computing expectation";
    Matrix m(T, F, arma::fill::zeros);
#pragma omp parallel for if (DO_PARALLEL)
    for (size_t f = 0; f < F; ++f)
      for (size_t t = 0; t < T; ++t)
        for (size_t g = 0; g < G; ++g)
          m(t, f) += expression(g, t) * function(g, f);
    return m;
  }
  pair<Matrix, Matrix> compute_shuffle_expectation(size_t N) const {
    // LOG(info) << "Computing expectation for shuffles";
    cerr << "Computing expectation for shuffles" << endl;
    Matrix M(T, F, arma::fill::zeros);
    Matrix Msd(T, F, arma::fill::zeros);
    vector<size_t> idx(G);
    for (size_t g = 0; g < G; ++g)
      idx[g] = g;
    for (size_t n = 0; n < N; ++n) {
      Matrix m(T, F, arma::fill::zeros);
      if (n % 100 == 0)
        cerr << "Computing expectation for shuffle " << n << endl;
      shuffle(begin(idx), end(idx), EntropySource::rng);
#pragma omp parallel for if (DO_PARALLEL)
      for (size_t f = 0; f < F; ++f)
        for (size_t g = 0; g < G; ++g)
          for (size_t t = 0; t < T; ++t)
            m(t, f) += expression(idx[g], t) * function(g, f);
      M += m;
      for (size_t f = 0; f < F; ++f)
        for (size_t t = 0; t < T; ++t)
          Msd(t, f) += m(t, f) * m(t, f);
    }
    for (size_t f = 0; f < F; ++f)
      for (size_t t = 0; t < T; ++t)
        Msd(t, f) = sqrt(N * Msd(t, f) - M(t, f) * M(t, f)) / N;
    M = M / N;
    return pair<Matrix, Matrix>(M, Msd);
  }
};

ostream &print(ostream &os, const Matrix &m,
           const vector<string> &row_names = vector<string>(),
           const vector<string> &col_names = vector<string>()) {
  for (auto name : col_names)
    os << "\t" << name;
  os << endl;
  for (size_t i = 0; i < m.n_rows; ++i) {
    os << row_names[i];
    for (size_t j = 0; j < m.n_cols; ++j)
      os << "\t" << m(i, j);
    os << endl;
  }
  return os;
}

int main(int argc, char **argv) {
  string expression_path = argv[1];
  string func_path = argv[2];

  Data data(expression_path, func_path);

  auto observed = data.compute_expectation();
  // LOG(info) << observed;

  auto expected = data.compute_shuffle_expectation(1000);
  // LOG(info) << expected;

  print(cerr, observed, data.type_names, data.func_names);
  print(cout, (observed - expected.first) / expected.second, data.type_names,
        data.func_names);

  return EXIT_SUCCESS;
}
