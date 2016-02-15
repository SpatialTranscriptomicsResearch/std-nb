#ifndef VARIANTMODEL_HPP
#define VARIANTMODEL_HPP

#include "counts.hpp"
#include "FactorAnalysis.hpp"
#include "verbosity.hpp"

namespace FactorAnalysis {
struct VariantModel {
  /** number of genes */
  size_t G;
  /** number of samples */
  size_t S;
  /** number of factors */
  size_t T;
  /** number of experiments */
  size_t E;

  Priors priors;
  Parameters parameters;

  /** hidden contributions to the count data due to the different factors */
  ITensor contributions;

  /** factor loading matrix */
  Matrix phi;

  /** factor score matrix */
  Matrix theta;

  /** spot scaling vector */
  Vector spot_scaling;

  /** experiment scaling vector */
  Vector experiment_scaling;
  Vector experiment_scaling_long;

  /** shape parameter for the prior of the loading matrix */
  Matrix r;
  /** scale parameter for the prior of the loading matrix */
  Matrix p;

  /** shape parameter for the prior of the loading matrix */
  Vector r_theta;
  /** scale parameter for the prior of the loading matrix */
  Vector p_theta;


  Verbosity verbosity;

  VariantModel(const Counts &counts, const size_t T, const Priors &priors,
               const Parameters &parameters, Verbosity verbosity);

  VariantModel(const std::string &phi_path, const std::string &theta_path,
               const std::string &spot_scaling_path,
               const std::string &experiment_scaling_path,
               const std::string &r_path, const std::string &p_path,
               const Priors &priors, const Parameters &parameters,
               Verbosity verbosity);

  double log_likelihood(const IMatrix &counts) const;

  /** sample count decomposition */
  void sample_contributions(const IMatrix &counts);

  /** sample phi */
  void sample_phi();

  /** sample p and r */
  void sample_p_and_r();

  /** sample theta */
  void sample_theta();

  /** sample p_theta and r_theta */
  void sample_p_and_r_theta();

  /** sample r_theta */
  void sample_r_theta();

  /** sample p_theta */
  void sample_p_theta();

  /** sample spot scaling factors */
  void sample_spot_scaling();

  /** sample experiment scaling factors */
  void sample_experiment_scaling(const Counts &data);

  /** sample each of the variables from their conditional posterior */
  void gibbs_sample(const Counts &data, bool timing);

  std::vector<Int> sample_reads(size_t g, size_t s, size_t n = 1) const;

  double posterior_expectation(size_t g, size_t s) const;
  double posterior_variance(size_t g, size_t s) const;
  Matrix posterior_expectations() const;
  Matrix posterior_variances() const;

  /** check that parameter invariants are fulfilled */
  void check_model(const IMatrix &counts) const;

 private:
  void update_experiment_scaling_long(const Counts &data);
};
}

std::ostream &operator<<(std::ostream &os,
                         const FactorAnalysis::VariantModel &pfa);

#endif
