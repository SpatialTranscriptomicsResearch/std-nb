#include "parameters.hpp"
#include "aux.hpp"

using namespace std;

namespace STD {

std::ostream &operator<<(std::ostream &os, const Hyperparameters &hyperparams) {
  // TODO print all

  os << "phi_r_1 = " << hyperparams.phi_r_1 << endl;
  os << "phi_r_2 = " << hyperparams.phi_r_2 << endl;
  return os;
}

bool Parameters::targeted(Target target) const {
  return flagged(targets & target);
}
}
