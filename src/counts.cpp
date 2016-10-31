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
namespace PF = PoissonFactorization;
using Int = PF::Int;
using IMatrix = PF::IMatrix;
using Matrix = PF::Matrix;
using Vector = PF::Vector;

Counts::Counts(const string &path, const string &separator)
    : row_names(),
      col_names(),
      counts(parse_file<IMatrix>(path, read_counts, separator, row_names,
                                 col_names, "")) {}

Counts::Counts(const vector<string> &rnames, const vector<string> &cnames,
               const IMatrix &cnts)
    : row_names(rnames),
      col_names(cnames),
      counts(cnts) {
  assert(rnames.size() == cnts.n_rows);
  assert(cnames.size() == cnts.n_cols);
}

Counts &Counts::operator=(const Counts &other) {
  row_names = other.row_names;
  col_names = other.col_names;
  counts = other.counts;
  return *this;
}

void select_top(std::vector<Counts> &counts_v, size_t top) {
  if (top == 0 or counts_v.empty() or counts_v[0].row_names.size() <= top)
    return;
  LOG(verbose) << "Selecting top " << top;

  Vector gene_sums = rowSums<Vector>(counts_v[0].counts);
  for (size_t i = 1; i < counts_v.size(); ++i)
    gene_sums += rowSums<Vector>(counts_v[i].counts);

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
    const size_t T = counts.counts.n_cols;
    IMatrix m(top, T, arma::fill::zeros);
    for (size_t i = 0; i < top; ++i)
      m.row(i) = counts.counts.row(order[i]);
    counts.counts = m;
    counts.row_names = names;
  }
}

vector<Counts> load_data(const vector<string> &paths, bool intersect,
                         size_t top) {
  vector<Counts> counts_v;
  for (auto &path : paths) {
    LOG(verbose) << "Loading " << path;
    counts_v.push_back(Counts({path}));
  }

  if (intersect)
    gene_intersection(counts_v);
  else
    gene_union(counts_v);

  select_top(counts_v, top);

  LOG(verbose) << "Done loading";
  return counts_v;
}

template <typename Fnc>
void match_genes(std::vector<Counts> &counts_v, Fnc fnc) {
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
    const size_t H = counts.counts.n_rows;
    const size_t S = counts.counts.n_cols;
    IMatrix new_counts(G, S, arma::fill::zeros);
    for (size_t h = 0; h < H; ++h) {
      auto iter = gene_map.find(counts.row_names[h]);
      if (iter != end(gene_map))
        new_counts.row(iter->second) = counts.counts.row(h);
    }
    counts.counts = new_counts;
    counts.row_names = selected;
  }
}

void gene_union(std::vector<Counts> &counts_v) {
  match_genes(counts_v, [](size_t x) { return x > 0; });
}

void gene_intersection(std::vector<Counts> &counts_v) {
  const size_t n = counts_v.size();
  match_genes(counts_v, [n](size_t x) { return x == n; });
}

template <typename T>
vector<T> split_on_x(const string &s, const std::string &token = "x") {
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
  size_t n = counts.n_cols;
  Matrix d(n, n, arma::fill::zeros);
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t i = 0; i < n; ++i)
    for (size_t j = i + 1; j < n; ++j)
      d(i, j) = d(j, i) = sq_distance(col_names[i], col_names[j]);
  return d;
}

Matrix Counts::parse_coords() const {
  if(counts.n_rows == 0)
    return Matrix(0, 0);
  const size_t n = split_on_x<double>(col_names[0]).size();
  Matrix coords(counts.n_cols, n);
  for(size_t i = 0; i < counts.n_cols; ++i) {
    auto coord = split_on_x<double>(col_names[i]);
    for(size_t j = 0; j < n; ++j)
      coords(i,j) = coord[j];
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

Matrix row_normalize(Matrix m) {
  m.each_row(do_normalize<arma::Row<double>>);
  return m;
}
