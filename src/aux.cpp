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

vector<string> split_at(char sep, const string &str) {
  vector<string> ret;
  istringstream ss(str);
  string line;
  while (getline(ss, line, sep))
    ret.push_back(line);
  return ret;
}

string trim(const string &str, char sym) {
  size_t trim_front = 0, trim_back = 0;
  while (str[trim_front] == sym)
    trim_front++;
  while (str[str.size() - trim_back - 1] == sym)
    trim_back++;
  return str.substr(trim_front, str.size() - trim_front - trim_back);
}
