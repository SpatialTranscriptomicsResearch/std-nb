#include "sampling.hpp"
#include <iostream>

#define _GNU_SOURCE
#include <fenv.h>

using namespace std;

template <class X>
void print_vec(const std::vector<X> &x) {
  cout << "[";
  for (auto y : x)
    cout << " " << y;
  cout << " "
       << "]" << endl;
}

int main(int argc, char **argv) {
  feenableexcept(FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW);
  // feenableexcept(FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW | FE_UNDERFLOW);
  vector<double> p = {0.5, 0.25, 0.25};
  EntropySource::seed();
  const size_t N = 1000000;
  for (size_t i = 0; i < N; ++i) {
    double r = std::gamma_distribution<double>(0.001, 1)(EntropySource::rng);
    cout << i << " " << r << endl;
    if (std::isnan(r)) {
      cout << "Problem!" << endl;
      return EXIT_FAILURE;
    }
  }
  return EXIT_SUCCESS;
  for (size_t i = 0; i < N; ++i) {
    auto x = sample_multinomial<size_t>(N, begin(p), end(p));
    auto y = sample_dirichlet<double>(begin(x), end(x));
    print_vec(x);
    print_vec(y);

    double sum = 0;
    for (auto &z : x)
      sum += z;
    if (sum != N) {
      cout << "Error: the sum of multinomial counts " << sum
           << " is not equal to " << N << endl;
      exit(EXIT_FAILURE);
    }

    sum = 0;
    for (auto &z : y)
      sum += z;
    if (fabs(sum - 1.0) > 1e-6) {
      cout << "Error: the sum of the dirichlet sampled distribution " << sum
           << " is not equal to " << 1.0 << endl
           << "Difference = " << (sum - 1.0) << endl;
      exit(EXIT_FAILURE);
    }
  }

  cout << endl;
  for (size_t i = 0; i < N; i++) {
    double x = sample_beta<double>(1.5, 0.5);
    cout << x << endl;
  }
  return EXIT_SUCCESS;
}
