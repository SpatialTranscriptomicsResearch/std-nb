#ifndef FACTORANALYSIS_HPP
#define FACTORANALYSIS_HPP

#include <cstdint>
#include <vector>
#include <boost/multi_array.hpp>

namespace FactorAnalysis {
using Int = uint32_t;
using Float = double;
using Vector = boost::multi_array<Float, 1>;
using Matrix = boost::multi_array<Float, 2>;
using IMatrix = boost::multi_array<Int, 2>;
using Tensor = boost::multi_array<Float, 3>;
using ITensor = boost::multi_array<Int, 3>;

struct Priors {
  Priors(Float c_ = 1.0, Float epsilon_ = 0.01, Float c0_ = 1.0,
         Float r0_ = 1.0, Float gamma_ = 1.0, Float alpha_ = 0.5)
      : c(c_),
        epsilon(epsilon_),
        c0(c0_),
        r0(r0_),
        gamma(gamma_),
        alpha(alpha_){};

  // priors for the beta distribution (22)
  Float c;
  Float epsilon;

  // priors for the gamma distribution (21)
  Float c0;
  Float r0;

  Float gamma;
  Float alpha;
};

struct Parameters {
  /** Adjustable step size for Metropolis-Hastings sampling of r[t] */
  double adj_step_size = 1.0;
  /** Temperature for Metropolis-Hastings sampling of r[t] */
  double temperature = 1.0;
};

Float digamma(Float x);
Float trigamma(Float x);

template <typename T>
T square(T x) {
  return x * x;
}
}
#endif
