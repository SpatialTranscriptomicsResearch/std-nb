#ifndef ADAGRAD_HPP
#define ADAGRAD_HPP

namespace STD {

struct adagrad_parameters {
  double eta = 1e-3;
  double epsilon = 1e-8;
};

template <typename T>
void adagrad_update(const T &grad, T &scale, T &x,
                    const adagrad_parameters &params) {
  scale += grad * grad;
  x += params.eta / (sqrt(scale) + params.epsilon) * grad;
}

}  // namespace STD

#endif  // ADAGRAD_HPP
