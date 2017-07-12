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

void enforce_positive_and_warn(const string &tag, STD::Matrix &m, bool warn) {
  const double min_val = std::numeric_limits<double>::denorm_min();
  for (STD::Index i = 0; i < m.rows(); ++i)
    for (STD::Index j = 0; j < m.cols(); ++j)
      if (m(i, j) < min_val) {
        if (warn) {
          LOG(warning) << "Found problematic value " << m(i, j) << " in " << tag
                       << " matrix at position " << i << " / " << j;
          LOG(warning) << "Setting to " << min_val << " and continuing.";
        }
        m(i, j) = min_val;
      }
}

void enforce_positive_and_warn(const string &tag, STD::Vector &v, bool warn) {
  const double min_val = std::numeric_limits<double>::denorm_min();
  for (STD::Index i = 0; i < v.rows(); ++i)
    if (v(i) < min_val) {
      if (warn) {
        LOG(warning) << "Found problematic value " << v(i) << " in " << tag
                     << " vector at position " << i;
        LOG(warning) << "Setting to " << min_val << " and continuing.";
      }
      v(i) = min_val;
    }
}

vector<string> split_at(char sep, const string &str) {
  vector<string> ret;
  istringstream ss(str);
  string line;
  while (getline(ss, line, sep))
    ret.push_back(line);
  return ret;
}

string trim(const string& str, char sym) {
  size_t trim_front = 0, trim_back = 0;
  while (str[trim_front] == sym)
    trim_front++;
  while (str[str.size() - trim_back - 1] == sym)
    trim_back++;
  return str.substr(trim_front, str.size() - trim_front - trim_back);
}
