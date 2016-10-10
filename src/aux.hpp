#ifndef AUX_HPP
#define AUX_HPP

#include <algorithm>
#include <cmath>
#include <iterator>
#include <string>
#include <sstream>
#include <vector>

/**
 * Create a lower case copy of a string
 *
 * @param x String to convert to lower case
 * @return lower case copy of x
 */
std::string to_lower(std::string x);

template <typename T>
std::string to_string_embedded(const T &t, size_t w, char symbol='0') {
  std::stringstream ss;
  ss.fill(symbol);
  ss.width(w);
  ss << t;
  return ss.str();
}

/**
 * Divide by the sum to compute relative proportions
 *
 * @param begin Iterator pointing to the first value
 * @param end Iterator pointing past the last value
 * @return A vector with the relative proportions
 */
template <typename Iter>
void normalize(const Iter begin, const Iter end) {
  double z = 0;
  for (Iter iter = begin; iter != end; ++iter)
    z += *iter;
  for (Iter iter = begin; iter != end; ++iter)
    *iter /= z;
}

/**
 * Retrieve specified quantiles from a container
 *
 * Side effect: sorts the container
 * TODO automatically determine return type
 *
 * @param begin Iterator pointing to the first value
 * @param begin Iterator pointing past the last value
 * @param quantiles Quantiles to retrieve. Must be >= 0 and <= 1.
 * @return A vector with the quantile values
 */
template <typename Iter>
std::vector<typename std::iterator_traits<Iter>::value_type> get_quantiles(
    const Iter begin, const Iter end, const std::vector<double> &quantiles) {
  std::vector<typename std::iterator_traits<Iter>::value_type> res;
  std::sort(begin, end);
  const size_t N = std::distance(begin, end);
  for (auto quantile : quantiles)
    res.push_back(*(begin + size_t((N - 1) * quantile)));
  return res;
}

/**
 * Numerically stable way of computing the logarithm of the sum of the
 * exponentials of two numbers.
 */
template <typename T>
T logSumExp(T a, T b) {
  if (a > b)
    return log(1 + exp(b - a)) + a;
  else
    return log(1 + exp(a - b)) + b;
}

#endif
