#include "PoissonFactorAnalysis.hpp"

using namespace std;

std::ostream &operator<<(std::ostream &os, const PoissonFactorAnalysis &pfa) {
  os << "Poisson Factor Analysis "
     << "N = " << pfa.N << " "
     << "G = " << pfa.G << " "
     << "K = " << pfa.K;
  return os;
}
