#ifndef PARAMETERS_HPP
#define PARAMETERS_HPP

#include <cstdint>
#include "coefficient.hpp"
#include "compression_mode.hpp"
#include "covariate.hpp"
#include "formula.hpp"
#include "optimization_method.hpp"
#include "rprop.hpp"
#include "sampling_method.hpp"
#include "types.hpp"

namespace STD {

const std::string default_output_string = "THIS PATH SHOULD NOT EXIST";

struct Hyperparameters {
  Hyperparameters(Float gamma_1_ = 1, Float gamma_2_ = 1, Float lambda_1_ = 50,
                  Float lambda_2_ = 50, Float rho_1_ = 2, Float rho_2_ = 2,
                  Float theta_r_1_ = 1, Float theta_r_2_ = 1,
                  Float theta_p_1_ = 0.05, Float theta_p_2_ = 0.95,
                  Float spot_a_ = 10, Float spot_b_ = 10, Float bline1 = 50,
                  Float bline2 = 50, Float normal_1_ = exp(0),
                  Float normal_2_ = 1)
      : gamma_1(gamma_1_),
        gamma_2(gamma_2_),
        lambda_1(lambda_1_),
        lambda_2(lambda_2_),
        rho_1(rho_1_),
        rho_2(rho_2_),
        theta_r_1(theta_r_1_),
        theta_r_2(theta_r_2_),
        theta_p_1(theta_p_1_),
        theta_p_2(theta_p_2_),
        spot_a(spot_a_),
        spot_b(spot_b_),
        beta_1(bline1),
        beta_2(bline2),
        normal_1(normal_1_),
        normal_2(normal_2_){};

  // TODO add: hyper-hyper-parameters

  // priors for the gamma distribution of r[g][t]
  Float gamma_1;
  Float gamma_2;

  Float lambda_1;
  Float lambda_2;

  // priors for the gamma distribution of p[g][t]
  Float rho_1;
  Float rho_2;
  //
  // priors for the gamma distribution of r[t]
  Float theta_r_1;
  Float theta_r_2;

  // priors for the gamma distribution of p[t]
  Float theta_p_1;
  Float theta_p_2;

  Float spot_a;
  Float spot_b;

  Float beta_1;
  Float beta_2;

  double normal_1;
  double normal_2;

  double get_param(Coefficient::Distribution distribution, size_t idx) const;
};

std::ostream &operator<<(std::ostream &os, const Hyperparameters &hyperparams);

struct GaussianProcessParameters {
  GaussianProcessParameters(bool use = false, double len = 5,
                            double spatial = 1, double indep = 1,
                            size_t first_iteration = 80);
  bool use;
  double length_scale;
  double spatial_variance;
  double independent_variance;
  size_t first_iteration;
};

struct Parameters {
  /** Maximal number of propositions for Metropolis-Hastings sampling */
  double n_iter = 100;
  /** Minimal positive value to enforce for parameters */
  double min_value = 1e-16;
  /** Temperature for Metropolis-Hastings sampling of r[g][t] */
  double temperature = 1.0;
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

  Formula rate_formula = DefaultRateFormula();
  Formula variance_formula = DefaultVarianceFormula();

  DistributionMode distribution_mode = DistributionMode::log_normal;

  GaussianProcessParameters gp = {};
};
}
#endif
