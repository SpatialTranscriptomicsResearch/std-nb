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
  for(size_t i = 0; i < N; ++i) {
    auto x = sample_multinomial(10000, p);
    auto y = sample_dirichlet(x);
    print_vec(x);
    print_vec(y);
  }

  cout << endl;
  for(size_t i = 0; i < N; i++) {
    double x = sample_beta(1.5, 0.5);
    cout << x << endl;
  }
  return EXIT_SUCCESS;
}
