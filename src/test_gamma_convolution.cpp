#include <iostream>
#include "pdist.hpp"

using namespace std;

int main(int argc, char **argv) {
  init_logging("foo.txt");

  size_t N = 1000;
  double max_val = 50;

  size_t K = 20;

  size_t n = 25;
  vector<double> shapes(n, 1);
  vector<double> scales(n, 1);
  verbosity=Verbosity::warning;
  for(size_t t = 0; t < N; ++t) {
  //   size_t t = 10;
    double x = 1.0 * t / N * max_val;
    double foo = convolved_gamma(x, K, shapes, scales);
    // LOG(info) << "The probability for k==" << t/10.0 << " = " << foo;
    cout << x << "\t" << foo << "\t" << exp(log_gamma(x, 1, 1)) << endl;
  }
}
