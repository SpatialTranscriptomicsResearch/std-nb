#include <iostream>
#include <string>
#include "compression.hpp"
#include "counts.hpp"
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
  Counts expression;
  Counts function;
  size_t T, G, F;
  Data(const string &exp_path, const string &func_path)
      : expression(exp_path, false),
        function(func_path, false),
        T(expression.matrix->cols()),
        G(expression.matrix->rows()),
        F(function.matrix->cols()) {
    // intersect gene_ids
    vector<Counts> cnts;
    cnts.push_back(expression);
    cnts.push_back(function);
    gene_intersection(cnts);

    // reduce to shared genes
    expression = cnts[0];
    function = cnts[1];

    // set T, G, and F
    LOG(info) << "T = " << T;
    LOG(info) << "G = " << G;
    LOG(info) << "F = " << F;
    LOG(info) << "function.n_rows = " << function.matrix->rows();
  }

  Matrix compute_expectation() const {
    LOG(info) << "Computing expectation";
    Matrix m = Matrix::Zero(T, F);
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
    Matrix M = Matrix::Zero(T, F);
    Matrix Msd = Matrix::Zero(T, F);
    vector<size_t> idx(G);
    for (size_t g = 0; g < G; ++g)
      idx[g] = g;
    for (size_t n = 0; n < N; ++n) {
      Matrix m = Matrix::Zero(T, F);
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
  for (int i = 0; i < m.rows(); ++i) {
    os << row_names[i];
    for (int j = 0; j < m.cols(); ++j)
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
  print(cout, (observed - expected.first).array() / expected.second.array(),
        data.type_names, data.func_names);

  return EXIT_SUCCESS;
}
