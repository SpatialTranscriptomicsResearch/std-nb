#ifndef ADAM_HPP
#define ADAM_HPP

#include <cmath>

namespace STD {

struct adam_parameters {
  // alpha = 1e-3 is suggested in orginal publication;
  double alpha = 0.1;
  double beta1 = 0.9;
  double beta2 = 0.999;
  double epsilon = 1e-8;
};

template <typename T>
void adam_update(const T &grad, T &mom1, T &mom2, T &x, size_t t,
                 const adam_parameters &params) {
  mom1 = params.beta1 * mom1 + (1 - params.beta1) * grad;
  mom2 = params.beta2 * mom2 + (1 - params.beta2) * grad * grad;
  const auto mom2_corr = 1 - pow(params.beta2, t);
  const auto mom1_corr = 1 - pow(params.beta1, t);
  x += params.alpha
       / (sqrt(mom2 / mom2_corr) + params.epsilon)
       * mom1 / mom1_corr;
}

template <typename T>
void nadam_update(const T &grad, T &mom1, T &mom2, T &x, size_t t,
                  const adam_parameters &params) {
  mom1 = params.beta1 * mom1 + (1 - params.beta1) * grad;
  mom2 = params.beta2 * mom2 + (1 - params.beta2) * grad * grad;
  const auto mom2_corr = 1 - pow(params.beta2, t);
  const auto beta1t = pow(params.beta1, t);
  const auto beta1tp1 = beta1t * params.beta1;
  x += params.alpha
       / (sqrt(mom2 / mom2_corr) + params.epsilon)
       * (params.beta1 * mom1 / (1 - beta1tp1)
          + (1 - params.beta1) * grad / (1 - beta1t));
}

}  // namespace STD

#endif  // ADAM_HPP
