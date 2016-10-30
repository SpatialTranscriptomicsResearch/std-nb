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
                                 col_names, "")),
      experiments(counts.n_cols, 0),
      experiment_names(1, path) {}

Counts::Counts(const vector<string> &rnames, const vector<string> &cnames,
               const IMatrix &cnts, const vector<size_t> &exps,
               const vector<string> &exp_names)
    : row_names(rnames),
      col_names(cnames),
      counts(cnts),
      experiments(exps),
      experiment_names(exp_names) {
  assert(rnames.size() == cnts.n_rows);
  assert(cnames.size() == cnts.n_cols);
  assert(experiments.size() == cnts.n_cols);
}

Counts &Counts::operator=(const Counts &other) {
  row_names = other.row_names;
  col_names = other.col_names;
  experiments = other.experiments;
  experiment_names = other.experiment_names;
  counts = other.counts;
  return *this;
}

template <typename T>
unordered_map<T, size_t> generate_index_map(const vector<T> &v) {
  const size_t n = v.size();
  unordered_map<T, size_t> m;
  for (size_t i = 0; i < n; ++i)
    m[v[i]] = i;
  return m;
}

Counts combine_counts(const Counts &a, const Counts &b, bool intersect) {
  auto n1 = a.row_names;
  auto n2 = b.row_names;
  sort(begin(n1), end(n1));
  sort(begin(n2), end(n2));
  vector<string> rnames(n1.size() + n2.size());
  if (intersect) {
    auto it = set_intersection(begin(n1), end(n1), begin(n2), end(n2),
                               begin(rnames));
    rnames.resize(it - begin(rnames));
  } else {
    auto it = set_union(begin(n1), end(n1), begin(n2), end(n2), begin(rnames));
    rnames.resize(it - begin(rnames));
  }

  vector<string> cnames = a.col_names;
  for (auto &name : b.col_names)
    cnames.push_back(name);

  auto m1 = generate_index_map(a.row_names);
  auto m2 = generate_index_map(b.row_names);

  const size_t nrow = rnames.size();
  const size_t ncol = cnames.size();

  const size_t ncol1 = a.col_names.size();

  IMatrix cnt(nrow, ncol);
  size_t col_idx = 0;
  for (; col_idx < ncol1; ++col_idx) {
    size_t row_idx = 0;
    for (auto &name : rnames) {
      auto iter = m1.find(name);
      if (iter != end(m1))
        cnt(row_idx, col_idx) = a.counts(iter->second, col_idx);
      row_idx++;
    }
  }
  for (; col_idx < ncol; ++col_idx) {
    size_t row_idx = 0;
    for (auto &name : rnames) {
      auto iter = m2.find(name);
      if (iter != end(m2))
        cnt(row_idx, col_idx) = b.counts(iter->second, col_idx - ncol1);
      row_idx++;
    }
  }

  // prepare vector of spot -> experiment IDs
  vector<size_t> exps = a.experiments;
  size_t max_label = 0;
  for (auto x : exps)
    if (x > max_label)
      max_label = x;
  max_label++;
  for (auto x : b.experiments)
    exps.push_back(x + max_label);

  // prepare vector of spot -> experiment labels
  vector<string> exp_names = a.experiment_names;
  for (auto x : b.experiment_names)
    exp_names.push_back(x);

  return {rnames, cnames, cnt, exps, exp_names};
}

Counts Counts::operator*(const Counts &other) const {
  return combine_counts(*this, other, true);
}

Counts Counts::operator+(const Counts &other) const {
  return combine_counts(*this, other, false);
}

void Counts::select_top(size_t n) {
  using pair_t = pair<size_t, size_t>;
  vector<pair_t> rowsum_and_index;
  const size_t nrow = row_names.size();
  const size_t ncol = col_names.size();

  for (size_t r = 0; r < nrow; ++r) {
    size_t sum = 0;
    for (size_t c = 0; c < ncol; ++c)
      sum += counts(r, c);
    rowsum_and_index.push_back(pair_t(sum, r));
  }

  sort(begin(rowsum_and_index), end(rowsum_and_index),
       [](const pair_t &a, const pair_t &b) { return a > b; });

  n = min<size_t>(n, nrow);

  vector<string> new_row_names(n);
  for (size_t i = 0; i < n; ++i)
    new_row_names[i] = row_names[rowsum_and_index[i].second];

  IMatrix new_counts(n, ncol);

  for (size_t r = 0; r < n; ++r)
    for (size_t c = 0; c < ncol; ++c)
      new_counts(r, c) = counts(rowsum_and_index[r].second, c);

  row_names = new_row_names;
  counts = new_counts;
}

vector<Counts> Counts::split_experiments() const {
  const size_t E = experiment_names.size();
  const size_t G = row_names.size();
  const size_t S = col_names.size();
  vector<size_t> sizes(E, 0);
  vector<vector<size_t>> idxs(E);
  vector<vector<string>> spot_names(E);
  for (size_t s = 0; s < S; ++s) {
    sizes[experiments[s]]++;
    spot_names[experiments[s]].push_back(col_names[s]);
    idxs[experiments[s]].push_back(s);
  }

  vector<Counts> split_counts;
  for (size_t e = 0; e < E; ++e) {
    IMatrix cnts(G, sizes[e]);
    size_t s = 0;
    for (auto idx : idxs[e]) {
      assert(idx < S);
      assert(s < sizes[e]);
      for (size_t g = 0; g < G; ++g)
        cnts(g, s) = counts(g, idx);
      s++;
    }
    Counts part_counts(row_names, spot_names[e], cnts,
                       vector<size_t>(sizes[e], 0), {experiment_names[e]});
    split_counts.push_back(part_counts);
  }

  return split_counts;
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
  if (false) {
    if (paths.empty())
      return {};
    Counts data(paths[0]);
    for (size_t i = 1; i < paths.size(); ++i)
      if (intersect)
        data = data * Counts(paths[i]);
      else
        data = data + Counts(paths[i]);

    if (top > 0)
      data.select_top(top);

    return data.split_experiments();
  } else {
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
