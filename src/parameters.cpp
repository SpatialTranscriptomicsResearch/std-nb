#include "parameters.hpp"
#include "aux.hpp"

using namespace std;

namespace STD {

double Hyperparameters::get_param(Coefficient::Distribution distribution,
                                  size_t idx) const {
  switch (distribution) {
    case Coefficient::Distribution::gamma:
      if (idx == 0)
        return gamma_1;
      else
        return gamma_2;
    case Coefficient::Distribution::beta_prime:
      if (idx == 0)
        return rho_1;
      else
        return rho_2;
    default:
      // TODO cov prior set for other disitributions
      throw std::runtime_error("Error: not implemented.");
      break;
  }
}

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
