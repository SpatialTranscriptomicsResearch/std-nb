#include "counts.hpp"
#include <algorithm>
#include <boost/tokenizer.hpp>
#include <exception>
#include <fstream>
#include <numeric>
#include <unordered_map>
#include "aux.hpp"
#include "compression.hpp"
#include "io.hpp"
#include "log.hpp"
#include "parallel.hpp"

using namespace std;
using Int = STD::Int;
using Matrix = STD::Matrix;
using Vector = STD::Vector;

Counts::Counts(const string &path_, bool transpose, const string &separator)
    : path(path_),
      row_names(),
      col_names(),
      matrix(make_shared<Matrix>(parse_file<Matrix>(
          path, read_floats, separator, row_names, col_names))) {
  if (transpose) {
    std::swap(row_names, col_names);
    *matrix = matrix->transpose();
  }
}

size_t Counts::operator()(size_t g, size_t t) const { return (*matrix)(g, t); }
// size_t &Counts::operator()(size_t g, size_t t) { return (*matrix)(g, t); }

size_t Counts::num_genes() const { return matrix->rows(); }

size_t Counts::num_samples() const { return matrix->cols(); }

void select_top(vector<Counts> &counts_v, size_t n, bool select_top) {
  if (n == 0 or counts_v.empty() or counts_v[0].row_names.size() <= n)
    return;
  if (select_top)
    LOG(verbose) << "Selecting top " << n;
  else
    LOG(verbose) << "Selecting bottom " << n;

  Vector gene_sums = rowSums<Vector>(*counts_v[0].matrix);
  for (size_t i = 1; i < counts_v.size(); ++i)
    gene_sums += rowSums<Vector>(*counts_v[i].matrix);

  const size_t G = gene_sums.size();

  vector<size_t> order(G);
  std::iota(begin(order), end(order), 0);
  sort(begin(order), end(order), [&gene_sums](size_t a, size_t b) {
    return gene_sums(a) > gene_sums(b);
  });

  if (not select_top)
    std::reverse(begin(order), end(order));

  order.resize(n);

  vector<string> names;
  for (auto &o : order)
    names.push_back(counts_v[0].row_names[o]);

  for (auto &counts : counts_v) {
    const size_t T = counts.matrix->cols();
    Matrix m = Matrix::Zero(n, T);
    for (size_t i = 0; i < n; ++i)
      m.row(i) = counts.matrix->row(order[i]);
    *counts.matrix = m;
    counts.row_names = names;
  }
}

void discard_empty_spots(Counts &cnt) {
  const size_t R = cnt.matrix->rows();
  const size_t C = cnt.matrix->cols();
  auto cs = colSums<Vector>(*cnt.matrix);
  auto num_empty = 0;
  for (size_t c = 0; c < C; ++c)
    if (cs(C - c - 1) == 0) {
      num_empty++;
      cnt.col_names.erase(begin(cnt.col_names) + C - c - 1);
    }

  auto old = *cnt.matrix;
  *cnt.matrix = Matrix(R, C - num_empty);
  size_t c_ = 0;
  for (size_t c = 0; c < C; ++c)
    if (cs(c) > 0)
      cnt.matrix->col(c_++) = old.col(c);
  LOG(verbose) << "Discarded " << num_empty << " spots with zero counts.";
}

void discard_empty_genes(vector<Counts> &cnts) {
  if (cnts.empty())
    return;
  const size_t R = cnts[0].matrix->rows();
  Vector rs = Vector::Zero(R);
  for (auto &cnt : cnts)
    rs += rowSums<Vector>(*cnt.matrix);
  auto num_empty = 0;
  for (size_t r = 0; r < R; ++r)
    if (rs(R - r - 1) == 0) {
      num_empty++;
      for (auto &cnt : cnts)
        cnt.row_names.erase(begin(cnt.row_names) + R - r - 1);
    }

  for (auto &cnt : cnts) {
    const size_t C = cnt.matrix->cols();
    auto old = *cnt.matrix;
    *cnt.matrix = Matrix(R - num_empty, C);
    size_t r_ = 0;
    for (size_t r = 0; r < R; ++r)
      if (rs(r) > 0)
        cnt.matrix->row(r_++) = old.row(r);
  }
  LOG(verbose) << "Discarded " << num_empty << " genes with zero counts.";
}

vector<Counts> load_data(const vector<string> &paths, bool intersect,
                         size_t top, size_t bottom, bool discard_empty,
                         bool transpose) {
  vector<Counts> counts_v;
  for (auto &path : paths) {
    LOG(verbose) << "Loading " << path;
    counts_v.push_back(Counts(path, transpose));
  }

  if (intersect)
    gene_intersection(counts_v);
  else
    gene_union(counts_v);

  select_top(counts_v, top, true);
  select_top(counts_v, bottom, false);

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
    const size_t H = counts.matrix->rows();
    const size_t S = counts.matrix->cols();
    Matrix new_counts = Matrix::Zero(G, S);
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
  size_t n = matrix->cols();
  Matrix d = Matrix::Zero(n, n);
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t i = 0; i < n; ++i)
    for (size_t j = i + 1; j < n; ++j)
      d(i, j) = d(j, i) = sq_distance(col_names[i], col_names[j]);
  return d;
}

Matrix Counts::parse_coords() const {
  if (matrix->rows() == 0)
    return Matrix(0, 0);
  const STD::Index n = split_on_x<double>(col_names[0]).size();
  Matrix coords(matrix->cols(), n);
  for (STD::Index i = 0; i < matrix->cols(); ++i) {
    auto coord = split_on_x<double>(col_names[i]);
    for (STD::Index j = 0; j < n; ++j)
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
  assert(a.cols() == b.cols());
  Matrix m = Matrix::Zero(a.rows(), b.rows());
  for (STD::Index i = 0; i < a.rows(); ++i)
    for (STD::Index j = 0; j < b.rows(); ++j)
      for (STD::Index k = 0; k < a.cols(); ++k) {
        const double x = a(i, k) - b(j, k);
        m(i, j) += x * x;
      }
  return m;
}

Matrix row_normalize(Matrix m) {
  for (STD::Index r = 0; r < m.rows(); ++r) {
    double z = m.row(r).sum();
    if (z > 0)
      m.row(r) /= z;
  }
  return m;
}

size_t sum_rows(const vector<Counts> &c) {
  size_t n = 0;
  for (auto &x : c)
    n += x.matrix->rows();
  return n;
}

size_t sum_cols(const vector<Counts> &c) {
  size_t n = 0;
  for (auto &x : c)
    n += x.matrix->cols();
  return n;
}

size_t max_row_number(const vector<Counts> &c) {
  size_t x = 0;
  for (auto &m : c)
    x = max<size_t>(x, m.matrix->rows());
  return x;
}
