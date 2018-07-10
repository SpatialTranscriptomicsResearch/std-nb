#ifndef AUX_HPP
#define AUX_HPP

#include <algorithm>
#include <cmath>
#include <iterator>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
#include "log.hpp"
#include "types.hpp"

/**
 * Create a lower case copy of a string
 *
 * @param x String to convert to lower case
 * @return lower case copy of x
 */
std::string to_lower(std::string x);

template <typename T>
std::string to_string_embedded(const T &t, size_t w, char symbol = '0') {
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
    return a + log1p(exp(b - a));
  else
    return b + log1p(exp(a - b));
}

std::vector<std::string> form_factor_names(size_t n);

template <typename V, typename M>
V colSums(const M &m) {
  const size_t X = m.rows();
  const size_t Y = m.cols();
  V v(Y);
  for (size_t y = 0; y < Y; ++y)
    v[y] = 0;
  for (size_t y = 0; y < Y; ++y)
    for (size_t x = 0; x < X; ++x)
      v[y] += m(x, y);
  return v;
}

template <typename V, typename M>
V rowSums(const M &m) {
  const size_t X = m.rows();
  const size_t Y = m.cols();
  V v(X);
  for (size_t x = 0; x < X; ++x)
    v[x] = 0;
  for (size_t x = 0; x < X; ++x)
    for (size_t y = 0; y < Y; ++y)
      v[x] += m(x, y);
  return v;
}

std::vector<size_t> random_order(size_t n);

template <typename V>
V gibbs(const V &y) {
  double m = *std::max_element(begin(y), end(y));
  V x = exp(y.array() - m);
  return x / x.sum();
}

std::vector<std::string> split_at(char sep, const std::string &str);

std::string trim(const std::string &str, char sym = ' ');

/**
 * Similar to std::accumulate but uses the first element in the input range as
 * the initial value.
 */
template <typename InputIt,
          typename value_type
          = typename std::iterator_traits<InputIt>::value_type>
value_type accumulate1(InputIt first, InputIt last) {
  value_type x = *first;
  while (++first != last) {
    x = x + *first;
  }
  return x;
}

/**
 * Prepends iterator elements by a given symbol.
 */
template <typename InputIt, typename OutputIt, typename T>
void prepend(InputIt first, InputIt last, OutputIt d_first, T value) {
  for (; first != last; ++first) {
    *d_first++ = value;
    *d_first++ = *first;
  }
}

/**
 * Intersperses iterator elements by a given symbol.
 */
template <typename InputIt, typename OutputIt, typename T>
void intersperse(InputIt first, InputIt last, OutputIt d_first, T value) {
  if (first == last) {
    return;
  }
  *d_first = *first;
  prepend(++first, last, ++d_first, value);
}

/**
 * Intersperses iterator by a given symbol and concatenates the result.
 */
template <typename InputIt, typename T>
T intercalate(InputIt begin, InputIt last, const T& x)
{
  std::vector<T> ret;
  intersperse(begin, last, std::back_inserter(ret), x);
  return std::accumulate(ret.begin(), ret.end(), T());
}

#endif
