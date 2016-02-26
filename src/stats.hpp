#ifndef STATS_HPP
#define STATS_HPP

#include <iostream>
#include <iomanip>
#include <vector>

namespace Stats {

template <typename V>
double compute_mean(const V &v) {
  double sum = 0;
  for (const auto &x : v)
    sum += x;
  return (sum / v.size());
}

template <typename V, typename T = typename V::value_type>
std::vector<T> get_percentiles_(const V &v,
                                const std::vector<double> &percentiles
                                = {0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0}) {
  const size_t n = v.size();
  auto percentile_iter = begin(percentiles);
  size_t i = 0;
  std::vector<T> percentile_values;
  for (auto &x : v) {
    i++;
    double current = 1.0 * i / n;
    if (current >= *percentile_iter) {
      percentile_values.push_back(x);
      percentile_iter++;
    }
  }
  return percentile_values;
}

template <typename V, typename T = typename V::value_type>
std::vector<T> get_percentiles(const V &v,
                               std::vector<double> percentiles
                               = {0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0}) {
  auto sorted = v;
  {  // sort the data
    std::sort(sorted.begin(), sorted.end());
  }
  {  // sort and uniquify the percentiles
    std::sort(begin(percentiles), end(percentiles));
    auto it = std::unique(begin(percentiles), end(percentiles));
    percentiles.resize(std::distance(begin(percentiles), it));
  }
  return get_percentiles_<V, T>(sorted, percentiles);
}

template <typename V, typename T=typename V::value_type>
std::string summary(const V &v, std::vector<double> percentiles
                                = {0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0},
                    size_t width = 12) {
  std::stringstream ss;
  ss << std::setw(width) << std::right << "Mean";
  for (auto &x : percentiles)
    ss << std::setw(width) << std::right << x;
  ss << std::endl;
  ss << std::setw(width) << std::right << compute_mean(v);
  for (auto x : get_percentiles<V,T>(v, percentiles))
    ss << std::setw(width) << std::right << x;
  ss << std::endl;
  return ss.str();
}
};

#endif
