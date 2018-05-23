#ifndef PARAMETERS_HPP
#define PARAMETERS_HPP

#include <cstdint>
#include "adagrad.hpp"
#include "adam.hpp"
#include "coefficient.hpp"
#include "compression_mode.hpp"
#include "covariate.hpp"
#include "optimization_method.hpp"
#include "rprop.hpp"
#include "types.hpp"

namespace STD {

const std::string default_output_string = "THIS PATH SHOULD NOT EXIST";

struct Hyperparameters {
  Hyperparameters(Float gamma_1_ = 1, Float gamma_2_ = 1, Float beta_1_ = 2,
                  Float beta_2_ = 2, Float beta_prime_1_ = 2,
                  Float beta_prime_2_ = 2, Float normal_1_ = 0,
                  Float normal_2_ = 1)
      : gamma_1(gamma_1_),
        gamma_2(gamma_2_),
        beta_1(beta_1_),
        beta_2(beta_2_),
        beta_prime_1(beta_prime_1_),
        beta_prime_2(beta_prime_2_),
        normal_1(normal_1_),
        normal_2(normal_2_){};

  // TODO add: hyper-hyper-parameters

  // default values for the gamma distribution
  Float gamma_1;
  Float gamma_2;

  // default values for the beta distribution
  Float beta_1;
  Float beta_2;

  // default values for the beta prime distribution
  Float beta_prime_1;
  Float beta_prime_2;

  // default values for the log normal distribution
  double normal_1;
  double normal_2;

  double get_param(Coefficient::Type distribution, size_t idx) const;
};

std::ostream &operator<<(std::ostream &os, const Hyperparameters &hyperparams);

struct GaussianProcessParameters {
  GaussianProcessParameters(size_t first_iteration = 0, bool free_mean=false);
  size_t first_iteration;
  bool free_mean;
};

struct Parameters {
  /** Minimal positive value to enforce for parameters */
  double min_value = 1e-16;
  bool warn_lower_limit = false;
  double hmc_epsilon = 1e-2;
  size_t hmc_L = 5;
  size_t hmc_N = 15;
  double dropout_gene_spot = 0;
  double downsample = 1;
  bool adjust_seq_depth = false;
  CompressionMode compression_mode = CompressionMode::gzip;
  Hyperparameters hyperparameters;

  adagrad_parameters adagrad;
  adam_parameters adam;
  bool adam_nesterov_momentum = false;
  rprop_parameters rprop;

  std::string output_directory = default_output_string;
  size_t report_interval = 200;

  Optimize::Method optim_method = Optimize::Method::RPROP;

  size_t grad_iterations = 10000;
  double grad_alpha = 1e-1;
  double grad_anneal = 0.999;

  GaussianProcessParameters gp = {0, false};

  double temperature = 1;

  // TODO make CLI configurable
  Coefficient::Type default_distribution = Coefficient::Type::normal;
};
}  // namespace STD
#endif
