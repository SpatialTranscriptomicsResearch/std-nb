#include "sampling.hpp"
#include <iostream>

using namespace std;

template <class X> void print_vec(const std::vector<X> &x) {
  cout << "[";
  for(auto y: x)
    cout << " " << y;
  cout << " " << "]" << endl;
}

int main(int argc, char **argv) {
  vector<double> p = {0.5, 0.25, 0.25};
  EntropySource::seed();
  const size_t N = 1000;
  for (size_t i = 0; i < N; ++i) {
    auto x = sample_multinomial<size_t>(N, p);
    auto y = sample_dirichlet<double>(x);
    print_vec(x);
    print_vec(y);

    double sum = 0;
    for (auto &z : x) sum += z;
    if (sum != N) {
      cout << "Error: the sum of multinomial counts " << sum
           << " is not equal to " << N << endl;
      exit(EXIT_FAILURE);
    }

    sum = 0;
    for (auto &z : y) sum += z;
    if (fabs(sum - 1.0) > 1e-6) {
      cout << "Error: the sum of the dirichlet sampled distribution " << sum
           << " is not equal to " << 1.0 << endl
           << "Difference = " << (sum - 1.0) << endl;
      exit(EXIT_FAILURE);
    }
  }

  cout << endl;
  for(size_t i = 0; i < N; i++) {
    double x = sample_beta<double>(1.5, 0.5);
    cout << x << endl;
  }
  return EXIT_SUCCESS;
}
