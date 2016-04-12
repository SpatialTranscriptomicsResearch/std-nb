#ifndef VARIANTMODEL_HPP
#define VARIANTMODEL_HPP

#include "counts.hpp"
#include "entropy.hpp"
#include "FactorAnalysis.hpp"
#include "verbosity.hpp"

namespace FactorAnalysis {
struct VariantModel {
  /** number of genes */
  const size_t G;
  /** number of samples */
  const size_t S;
  /** number of factors */
  const size_t T;
  /** number of experiments */
  const size_t E;

  Hyperparameters hyperparameters;
  Parameters parameters;

  /** hidden contributions to the count data due to the different factors */
  Matrix contributions_gene_type, contributions_spot_type;
  Vector contributions_spot, contributions_experiment;

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
  /* Stored as negative-odds */
  Matrix p;

  /** shape parameter for the prior of the mixing matrix */
  Vector r_theta;
  /** scale parameter for the prior of the mixing matrix */
  /* Stored as negative-odds */
  Vector p_theta;


  Verbosity verbosity;

  struct Paths {
    Paths(const std::string &prefix, const std::string &suffix = "");
    std::string phi, theta, spot, experiment, r_phi, p_phi, r_theta, p_theta;
  };

  VariantModel(const Counts &counts, const size_t T,
               const Hyperparameters &hyperparameters,
               const Parameters &parameters, Verbosity verbosity);

  VariantModel(const Counts &counts, const Paths &paths,
               const Hyperparameters &hyperparameters,
               const Parameters &parameters, Verbosity verbosity);

  void store(const Counts &counts, const std::string &prefix,
                   bool mean_and_variance = false) const;

  double log_likelihood(const IMatrix &counts) const;

  /** sample count decomposition */
  void sample_contributions(const IMatrix &counts);
  void sample_contributions_sub(const IMatrix &counts, size_t g, size_t s,
                                RNG &rng, Matrix &contrib_gene_type,
                                Matrix &contrib_spot_type);

  /** sample phi */
  void sample_phi();
  Float sample_phi_sub(size_t g, size_t t, Float theta_t, RNG &rng) const;

  /** sample p and r */
  void sample_p_and_r();

  /** sample theta */
  void sample_theta();
  Float sample_theta_sub(size_t s, size_t t, Float scale) const;

  /** sample p_theta and r_theta */
  void sample_p_and_r_theta();

  /** sample spot scaling factors */
  void sample_spot_scaling();

  /** sample experiment scaling factors */
  void sample_experiment_scaling(const Counts &data);

  /** sample each of the variables from their conditional posterior */
  void gibbs_sample(const Counts &data, bool timing);

  std::vector<Int> sample_reads(size_t g, size_t s, size_t n = 1) const;

  double posterior_expectation(size_t g, size_t s) const;
  double posterior_expectation_poisson(size_t g, size_t s) const;
  double posterior_variance(size_t g, size_t s) const;
  Matrix posterior_expectations() const;
  Matrix posterior_expectations_poisson() const;
  Matrix posterior_variances() const;

  /** check that parameter invariants are fulfilled */
  void check_model(const IMatrix &counts) const;

 private:
  void update_experiment_scaling_long(const Counts &data);
};

std::ostream &operator<<(std::ostream &os,
                         const FactorAnalysis::VariantModel &pfa);
}

#endif
