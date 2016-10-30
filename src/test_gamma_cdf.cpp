#include "pdist.hpp"

#include <iostream>

using namespace std;

int main(int argc, char **argv){
  const double shape = 2;
  const double scale = 2;
  const size_t N = 100;
  for(size_t i = 0; i < N; ++i) {
    double x = i / 1.0;
    double p = gamma_cdf(x, shape, scale);
    double p_inv = inverse_gamma_cdf(p, shape, scale);
    cout << x << "\t" << p << "\t" << p_inv << endl;
  }
}
