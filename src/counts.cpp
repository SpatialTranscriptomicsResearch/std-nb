#include <algorithm>
#include <fstream>
#include <exception>
#include <unordered_map>
#include <boost/tokenizer.hpp>
#include "compression.hpp"
#include "counts.hpp"
#include "io.hpp"

const bool counts_debugging = false;

using namespace std;
namespace PF = PoissonFactorization;
using Int = PF::Int;
using IMatrix = PF::IMatrix;
using Matrix = PF::Matrix;
using Vector = PF::Vector;

Counts::Counts(const string &path, const string &label, const string &separator)
    : row_names(),
      col_names(),
      counts(parse_file<IMatrix>(path, read_counts, separator, row_names,
                                 col_names, label)),
      experiments(counts.n_cols, 0),
      experiment_names(1, path) {
  if (counts_debugging) {
    const size_t n = 10;
    for (size_t j = 0; j < min<size_t>(n, counts.n_cols); ++j)
      cout << "\t" << col_names[j];
    cout << endl;
    for (size_t i = 0; i < min<size_t>(n, counts.n_rows); ++i) {
      cout << row_names[i];
      for (size_t j = 0; j < min<size_t>(n, counts.n_cols); ++j)
        cout << "\t" << counts(i, j);
      cout << endl;
    }
  }
}

Counts::Counts(const vector<string> &rnames, const vector<string> &cnames,
               const IMatrix &cnts, const vector<size_t> &exps,
               const vector<string> &exp_names)
    : row_names(rnames),
      col_names(cnames),
      counts(cnts),
      experiments(exps),
      experiment_names(exp_names) {}

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
