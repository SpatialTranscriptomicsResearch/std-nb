#ifndef ADAM_HPP
#define ADAM_HPP

#include <cmath>

namespace STD {

struct adam_parameters {
  double alpha = 1e-3;
  double beta1 = 0.9;
  double beta2 = 0.999;
  double epsilon = 1e-8;
};

template <typename T>
void adam_update(const T &grad, T &mom1, T &mom2, T &x, size_t t,
                 const adam_parameters &params) {
  const auto mom1_corr = 1 - pow(params.beta1, t);
  const auto mom2_corr = sqrt(1 - pow(params.beta2, t));
  const auto alph_corr = params.alpha * mom2_corr / mom1_corr;
  mom1 = params.beta1 * mom1 + (1 - params.beta1) * grad;
  mom2 = params.beta1 * mom2 + (1 - params.beta2) * grad * grad;
  x += alph_corr * mom1 / (sqrt(mom2) + mom2_corr * params.epsilon);
}

}  // namespace STD

#endif  // ADAM_HPP
