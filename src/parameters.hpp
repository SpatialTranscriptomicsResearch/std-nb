#ifndef PARAMETERS_HPP
#define PARAMETERS_HPP

#include <cstdint>
#include "compression_mode.hpp"
#include "optimization_method.hpp"
#include "sampling_method.hpp"
#include "target.hpp"
#include "types.hpp"

namespace STD {

const std::string default_output_string = "THIS PATH SHOULD NOT EXIST";

struct Hyperparameters {
  Hyperparameters(Float phi_r_1_ = 1, Float phi_r_2_ = 1,
                  Float local_phi_r_1_ = 50, Float local_phi_r_2_ = 50,
                  Float phi_p_1_ = 2, Float phi_p_2_ = 2, Float theta_r_1_ = 1,
                  Float theta_r_2_ = 1, Float theta_p_1_ = 0.05,
                  Float theta_p_2_ = 0.95, Float spot_a_ = 10,
                  Float spot_b_ = 10, Float bline1 = 50, Float bline2 = 50)
      : phi_r_1(phi_r_1_),
        phi_r_2(phi_r_2_),
        local_phi_r_1(local_phi_r_1_),
        local_phi_r_2(local_phi_r_2_),
        phi_p_1(phi_p_1_),
        phi_p_2(phi_p_2_),
        theta_r_1(theta_r_1_),
        theta_r_2(theta_r_2_),
        theta_p_1(theta_p_1_),
        theta_p_2(theta_p_2_),
        spot_a(spot_a_),
        spot_b(spot_b_),
        baseline_1(bline1),
        baseline_2(bline2){};

  // TODO add: hyper-hyper-parameters

  // priors for the gamma distribution of r[g][t]
  Float phi_r_1;
  Float phi_r_2;

  Float local_phi_r_1;
  Float local_phi_r_2;

  // priors for the gamma distribution of p[g][t]
  Float phi_p_1;
  Float phi_p_2;
  //
  // priors for the gamma distribution of r[t]
  Float theta_r_1;
  Float theta_r_2;

  // priors for the gamma distribution of p[t]
  Float theta_p_1;
  Float theta_p_2;

  Float spot_a;
  Float spot_b;

  Float baseline_1;
  Float baseline_2;
};

std::ostream &operator<<(std::ostream &os, const Hyperparameters &hyperparams);

struct Parameters {
  /** Maximal number of propositions for Metropolis-Hastings sampling */
  double n_iter = 100;
  /** Temperature for Metropolis-Hastings sampling of r[g][t] */
  double temperature = 1.0;
  bool expected_contributions = false;
  bool normalize_spot_stats = false;
  bool p_empty_map = false;
  bool contributions_map = false;
  double hmc_epsilon = 1e-2;
  size_t hmc_L = 5;
  size_t hmc_N = 15;
  bool ignore_priors = false;
  double dropout_gene_spot = 0;
  CompressionMode compression_mode = CompressionMode::gzip;
  Hyperparameters hyperparameters;
  Target targets = DefaultTarget();
  bool targeted(Target target) const;

  std::string output_directory = default_output_string;
  double field_lambda_dirichlet = 1;
  double field_lambda_laplace = 1;
  size_t mesh_additional = 10000;
  double lbfgs_epsilon = 1e-5;
  size_t lbfgs_iter = 100;
  size_t report_interval = 200;

  Optimize::Method optim_method = Optimize::Method::RPROP;
  Sampling::Method sample_method = Sampling::Method::Mean;
  size_t grad_iterations = 10000;
  double grad_alpha = 1e-1;
  double grad_anneal = 0.999;

  double mesh_hull_enlarge = 1.03;
  double mesh_hull_distance = 2;
};
}
#endif
