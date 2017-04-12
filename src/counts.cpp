#include "counts.hpp"
#include <algorithm>
#include <boost/tokenizer.hpp>
#include <exception>
#include <fstream>
#include <unordered_map>
#include "aux.hpp"
#include "compression.hpp"
#include "io.hpp"
#include "log.hpp"
#include "parallel.hpp"

using namespace std;
namespace PF = STD;
using Int = PF::Int;
using IMatrix = PF::IMatrix;
using Matrix = PF::Matrix;
using Vector = PF::Vector;

Counts::Counts(const string &path_, const string &separator)
    : path(path_),
      row_names(),
      col_names(),
      matrix(new IMatrix(parse_file<IMatrix>(path, read_counts, separator,
                                             row_names, col_names))) {}

size_t Counts::operator()(size_t g, size_t t) const { return (*matrix)(g, t); }
// size_t &Counts::operator()(size_t g, size_t t) { return (*matrix)(g, t); }

size_t Counts::num_genes() const { return matrix->n_rows; }

size_t Counts::num_samples() const { return matrix->n_cols; }

void select_top(vector<Counts> &counts_v, size_t top) {
  if (top == 0 or counts_v.empty() or counts_v[0].row_names.size() <= top)
    return;
  LOG(verbose) << "Selecting top " << top;

  Vector gene_sums = rowSums<Vector>(*counts_v[0].matrix);
  for (size_t i = 1; i < counts_v.size(); ++i)
    gene_sums += rowSums<Vector>(*counts_v[i].matrix);

  const size_t G = gene_sums.n_elem;

  vector<size_t> order(G);
  iota(begin(order), end(order), 0);
  sort(begin(order), end(order), [&gene_sums](size_t a, size_t b) {
    return gene_sums(a) > gene_sums(b);
  });
  order.resize(top);

  vector<string> names;
  for (auto &o : order)
    names.push_back(counts_v[0].row_names[o]);

  for (auto &counts : counts_v) {
    const size_t T = counts.matrix->n_cols;
    IMatrix m(top, T, arma::fill::zeros);
    for (size_t i = 0; i < top; ++i)
      m.row(i) = counts.matrix->row(order[i]);
    *counts.matrix = m;
    counts.row_names = names;
  }
}

void discard_empty_spots(Counts &c) {
  auto cs = colSums<Vector>(*c.matrix);
  const size_t N = cs.n_elem;
  auto cnt = 0;
  for (size_t n = 0; n < N; ++n)
    if (cs(N - n - 1) == 0) {
      cnt++;
      c.matrix->shed_col(N - n - 1);
      c.col_names.erase(begin(c.col_names) + N - n - 1);
    }
  LOG(verbose) << "Discarded " << cnt << " spots with zero counts.";
}

void discard_empty_genes(vector<Counts> &cs) {
  if (cs.empty())
    return;

  const size_t G = cs[0].matrix->n_rows;

  Vector rowsums(G, arma::fill::zeros);
  for (auto &c : cs)
    rowsums = rowsums + rowSums<Vector>(*c.matrix);

  size_t cnt = 0;
  for (auto &rowsum : rowsums)
    if (rowsum == 0)
      cnt++;

  for (auto &c : cs)
    for (size_t g = 0; g < G; ++g)
      if (rowsums(G - g - 1) == 0) {
        c.matrix->shed_row(G - g - 1);
        c.row_names.erase(begin(c.row_names) + G - g - 1);
      }
  LOG(verbose) << "Discarded " << cnt << " genes with zero counts.";
}

vector<Counts> load_data(const vector<string> &paths, bool intersect,
                         size_t top, bool discard_empty) {
  vector<Counts> counts_v;
  for (auto &path : paths) {
    LOG(verbose) << "Loading " << path;
    counts_v.push_back(Counts(path));
  }

  if (intersect)
    gene_intersection(counts_v);
  else
    gene_union(counts_v);

  select_top(counts_v, top);

  if (discard_empty) {
    for (auto &counts : counts_v)
      discard_empty_spots(counts);
    discard_empty_genes(counts_v);
  }

  LOG(verbose) << "Done loading";
  return counts_v;
}

template <typename Fnc>
void match_genes(vector<Counts> &counts_v, Fnc fnc) {
  LOG(verbose) << "Matching genes";
  unordered_map<string, size_t> present;
  for (auto &counts : counts_v)
    for (auto &name : counts.row_names)
      present[name]++;

  vector<string> selected;
  for (auto &entry : present)
    if (fnc(entry.second))
      selected.push_back(entry.first);

  const size_t G = selected.size();

  sort(begin(selected), end(selected));

  unordered_map<string, size_t> gene_map;
  for (size_t g = 0; g < G; ++g)
    gene_map[selected[g]] = g;

  for (auto &counts : counts_v) {
    const size_t H = counts.matrix->n_rows;
    const size_t S = counts.matrix->n_cols;
    IMatrix new_counts(G, S, arma::fill::zeros);
    for (size_t h = 0; h < H; ++h) {
      auto iter = gene_map.find(counts.row_names[h]);
      if (iter != end(gene_map))
        new_counts.row(iter->second) = counts.matrix->row(h);
    }
    *counts.matrix = new_counts;
    counts.row_names = selected;
  }
}

void gene_union(vector<Counts> &counts_v) {
  match_genes(counts_v, [](size_t x) { return x > 0; });
}

void gene_intersection(vector<Counts> &counts_v) {
  const size_t n = counts_v.size();
  match_genes(counts_v, [n](size_t x) { return x == n; });
}

template <typename T>
vector<T> split_on_x(const string &s, const string &token = "x") {
  vector<T> v;
  size_t last_pos = 0;
  while (true) {
    auto pos = s.find(token, last_pos);
    if (pos != string::npos) {
      v.push_back(atof(s.substr(last_pos, pos - last_pos).c_str()));
    } else {
      v.push_back(atof(s.substr(last_pos).c_str()));
      break;
    }
    last_pos = pos + 1;
  }
  return v;
}

double sq_distance(const string &a, const string &b) {
  auto x = split_on_x<double>(a);
  auto y = split_on_x<double>(b);
  const size_t n = x.size();
  double d = 0;
  for (size_t i = 0; i < n; ++i) {
    double z = x[i] - y[i];
    d += z * z;
  }
  return d;
}

Matrix Counts::compute_distances() const {
  size_t n = matrix->n_cols;
  Matrix d(n, n, arma::fill::zeros);
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t i = 0; i < n; ++i)
    for (size_t j = i + 1; j < n; ++j)
      d(i, j) = d(j, i) = sq_distance(col_names[i], col_names[j]);
  return d;
}

Matrix Counts::parse_coords() const {
  if (matrix->n_rows == 0)
    return Matrix(0, 0);
  const size_t n = split_on_x<double>(col_names[0]).size();
  Matrix coords(matrix->n_cols, n);
  for (size_t i = 0; i < matrix->n_cols; ++i) {
    auto coord = split_on_x<double>(col_names[i]);
    for (size_t j = 0; j < n; ++j)
      coords(i, j) = coord[j];
  }
  return coords;
}

template <typename T>
void do_normalize(T &v) {
  double z = 0;
  for (auto x : v)
    z += x;
  if (z > 0)
    for (auto &x : v)
      x /= z;
}

Matrix compute_sq_distances(const Matrix &a, const Matrix &b) {
  assert(a.n_cols == b.n_cols);
  Matrix m(a.n_rows, b.n_rows, arma::fill::zeros);
  for (size_t i = 0; i < a.n_rows; ++i)
    for (size_t j = 0; j < b.n_rows; ++j)
      for (size_t k = 0; k < a.n_cols; ++k) {
        const double x = a(i, k) - b(j, k);
        m(i, j) += x * x;
      }
  return m;
}

Matrix row_normalize(Matrix m) {
  m.each_row(do_normalize<arma::Row<double>>);
  return m;
}

size_t sum_rows(const vector<Counts> &c) {
  size_t n = 0;
  for (auto &x : c)
    n += x.matrix->n_rows;
  return n;
}

size_t max_row_number(const vector<Counts> &c) {
  size_t x = 0;
  for (auto &m : c)
    x = max<size_t>(x, m.matrix->n_rows);
  return x;
}
