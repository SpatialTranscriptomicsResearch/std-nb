#ifndef PARAMETERS_HPP
#define PARAMETERS_HPP

#include <cstdint>
#include "coefficient.hpp"
#include "compression_mode.hpp"
#include "covariate.hpp"
#include "optimization_method.hpp"
#include "rprop.hpp"
#include "sampling_method.hpp"
#include "types.hpp"

namespace STD {

const std::string default_output_string = "THIS PATH SHOULD NOT EXIST";

struct Hyperparameters {
  Hyperparameters(Float gamma_1_ = 1, Float gamma_2_ = 1,
                  Float beta_prime_1_ = 2, Float beta_prime_2_ = 2,
                  Float normal_1_ = exp(0), Float normal_2_ = 1)
      : gamma_1(gamma_1_)
      , gamma_2(gamma_2_)
      , beta_prime_1(beta_prime_1_)
      , beta_prime_2(beta_prime_2_)
      , normal_1(normal_1_)
      , normal_2(normal_2_){};

  // TODO add: hyper-hyper-parameters

  // default values for the gamma distribution
  Float gamma_1;
  Float gamma_2;

  // default values for the beta prime distribution
  Float beta_prime_1;
  Float beta_prime_2;

  // default values for the log normal distribution
  double normal_1;
  double normal_2;

  double get_param(Coefficient::Distribution distribution, size_t idx) const;
};

std::ostream &operator<<(std::ostream &os, const Hyperparameters &hyperparams);

struct GaussianProcessParameters {
  GaussianProcessParameters(double len = 5, double indep = 1, size_t first_iteration = 80);
  double length_scale;
  double independent_variance;
  size_t first_iteration;
};

struct Parameters {
  /** Minimal positive value to enforce for parameters */
  double min_value = 1e-16;
  bool warn_lower_limit = false;
  double hmc_epsilon = 1e-2;
  size_t hmc_L = 5;
  size_t hmc_N = 15;
  double dropout_gene_spot = 0;
  CompressionMode compression_mode = CompressionMode::gzip;
  Hyperparameters hyperparameters;

  rprop_parameters rprop;

  std::string output_directory = default_output_string;
  double lbfgs_epsilon = 1e-5;
  size_t lbfgs_iter = 100;
  size_t report_interval = 200;

  Optimize::Method optim_method = Optimize::Method::RPROP;
  Sampling::Method sample_method = Sampling::Method::Mean;

  size_t sample_iterations = 10;

  size_t grad_iterations = 10000;
  double grad_alpha = 1e-1;
  double grad_anneal = 0.999;

  GaussianProcessParameters gp = {};

  double temperature = 1;

  // TODO make CLI configurable
  Coefficient::Distribution default_distribution = Coefficient::Distribution::normal;
};
}
#endif
