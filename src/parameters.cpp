#include "parameters.hpp"
#include "aux.hpp"

using namespace std;

namespace STD {

std::ostream &operator<<(std::ostream &os, const Hyperparameters &hyperparams) {
  // TODO print all

  os << "gamma_1 = " << hyperparams.gamma_1 << endl;
  os << "gamma_2 = " << hyperparams.gamma_2 << endl;
  os << "rho_1 = " << hyperparams.rho_1 << endl;
  os << "rho_2 = " << hyperparams.rho_2 << endl;
  return os;
}

bool Parameters::targeted(Target target) const {
  return flagged(targets & target);
}
bool Parameters::forget(Target target) const {
  return flagged(targets & target);
}
}
