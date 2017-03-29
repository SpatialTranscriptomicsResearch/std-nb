#include "aux.hpp"
#include <algorithm>
#include <random>
#include "entropy.hpp"

using namespace std;

string to_lower(string x) {
  transform(begin(x), end(x), begin(x), ::tolower);
  return x;
}

vector<string> form_factor_names(size_t n) {
  vector<string> factor_names;
  for (size_t i = 1; i <= n; ++i)
    factor_names.push_back("Factor " + std::to_string(i));
  return factor_names;
}

vector<size_t> random_order(size_t n) {
  vector<size_t> order(n);
  iota(begin(order), end(order), 0);
  shuffle(begin(order), end(order), EntropySource::rng);
  return order;
}

void enforce_positive_and_warn(const string &tag,
                               PoissonFactorization::Matrix &m, bool warn) {
  const double min_val = std::numeric_limits<double>::denorm_min();
  for (size_t i = 0; i < m.n_rows; ++i)
    for (size_t j = 0; j < m.n_cols; ++j)
      if (m(i, j) < min_val) {
        if (warn) {
          LOG(warning) << "Found problematic value " << m(i, j) << " in " << tag
                       << " matrix at position " << i << " / " << j;
          LOG(warning) << "Setting to " << min_val << " and continuing.";
        }
        m(i, j) = min_val;
      }
}

void enforce_positive_and_warn(const string &tag,
                               PoissonFactorization::Vector &v, bool warn) {
  const double min_val = std::numeric_limits<double>::denorm_min();
  for (size_t i = 0; i < v.n_rows; ++i)
    if (v(i) < min_val) {
      if (warn) {
        LOG(warning) << "Found problematic value " << v(i) << " in " << tag
                     << " vector at position " << i;
        LOG(warning) << "Setting to " << min_val << " and continuing.";
      }
      v(i) = min_val;
    }
}
