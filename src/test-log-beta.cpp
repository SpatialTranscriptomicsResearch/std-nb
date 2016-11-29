#include "pdist.hpp"

#include <iostream>

using namespace std;

int main(int argc, char** argv) {
  size_t N = 100;
  double a = 0.05;
  double b = 0.95;
  for (size_t n = 0; n <= N; ++n) {
    double p = 1.0 * n / N;
    double odds = p / (1 - p);
    double neg_odds = (1 - p) / p;
    cout << log_beta(p, a, b) << "\t"
         << log_beta_odds(odds, a, b) << "\t"
         << log_beta_neg_odds(neg_odds, a, b) << endl;
  }
  return 0;
}
