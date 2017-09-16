#include "parameters.hpp"
#include "aux.hpp"

using namespace std;

namespace STD {

GaussianProcessParameters::GaussianProcessParameters(double len, double indep,
                                                     size_t first_iter)
    : length_scale(len)
    , independent_variance(indep)
    , first_iteration(first_iter) {}

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
        return beta_prime_1;
      else
        return beta_prime_2;
    case Coefficient::Distribution::log_normal:
      if (idx == 0)
        return log_normal_1;
      else
        return log_normal_2;
    case Coefficient::Distribution::log_gp:
      if (idx == 0)
        return log_normal_1;  // TODO gp mu parameter
      else
        return log_normal_2;  // TODO gp sigma parameter
    case Coefficient::Distribution::log_gp_proxy:
      if (idx == 0)
        return log_normal_1;  // TODO gp mu parameter
      else
        return log_normal_2;  // TODO gp sigma parameter
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
  os << "beta_prime_1 = " << hyperparams.beta_prime_1 << endl;
  os << "beta_prime_2 = " << hyperparams.beta_prime_2 << endl;
  os << "log_normal_1 = " << hyperparams.log_normal_1 << endl;
  os << "log_normal_2 = " << hyperparams.log_normal_2 << endl;
  return os;
}
}
