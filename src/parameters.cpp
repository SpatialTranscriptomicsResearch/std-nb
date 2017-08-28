#include "parameters.hpp"
#include "aux.hpp"

using namespace std;

namespace STD {

GaussianProcessParameters::GaussianProcessParameters(bool use_, double len,
                                                     double spatial,
                                                     double indep)
    : use(use_),
      length_scale(len),
      spatial_variance(spatial),
      independent_variance(indep) {}

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
    case Coefficient::Distribution::log_normal:
      if (idx == 0)
        return normal_1;
      else
        return normal_2;
    case Coefficient::Distribution::log_gp:
      if (idx == 0)
        return normal_1;  // TODO gp mu parameter
      else
        return normal_2;  // TODO gp sigma parameter
    default:
      // TODO cov prior set for other disitributions
      throw std::runtime_error(
          "Error: no default hyper parameters defined for chosen "
          "distribution.");
      break;
  }
}

std::ostream &operator<<(std::ostream &os, const Hyperparameters &hyperparams) {
  // TODO print all

  os << "gamma_1 = " << hyperparams.gamma_1 << endl;
  os << "gamma_2 = " << hyperparams.gamma_2 << endl;
  os << "rho_1 = " << hyperparams.rho_1 << endl;
  os << "rho_2 = " << hyperparams.rho_2 << endl;
  os << "normal_1 = " << hyperparams.normal_1 << endl;
  os << "normal_2 = " << hyperparams.normal_2 << endl;
  return os;
}
}
