#ifndef VARIANTMODEL_HPP
#define VARIANTMODEL_HPP

#include "counts.hpp"
#include "entropy.hpp"
#include "FactorAnalysis.hpp"
#include "verbosity.hpp"

namespace FactorAnalysis {

enum class GibbsSample {
  empty = 0,
  contributions = 1 << 0,
  phi = 1 << 1,
  phi_r = 1 << 2,
  phi_p = 1 << 3,
  theta = 1 << 4,
  theta_r = 1 << 5,
  theta_p = 1 << 6,
  spot_scaling = 1 << 7,
  experiment_scaling = 1 << 8,
  merge_split = 1 << 9,
};

std::ostream &operator<<(std::ostream &os, const GibbsSample &which);
std::istream &operator>>(std::istream &is, GibbsSample &which);

inline constexpr GibbsSample operator&(GibbsSample a, GibbsSample b) {
  return static_cast<GibbsSample>(static_cast<int>(a) & static_cast<int>(b));
}

inline constexpr GibbsSample operator|(GibbsSample a, GibbsSample b) {
  return static_cast<GibbsSample>(static_cast<int>(a) | static_cast<int>(b));
}

inline constexpr GibbsSample operator^(GibbsSample a, GibbsSample b) {
  return static_cast<GibbsSample>(static_cast<int>(a) & static_cast<int>(b));
}

inline constexpr GibbsSample
operator~(GibbsSample a) {
  return static_cast<GibbsSample>((~static_cast<int>(a)) & ((1 << 10) - 1));
}

inline constexpr GibbsSample DefaultGibbs() {
  return GibbsSample::contributions | GibbsSample::phi | GibbsSample::phi_r
         | GibbsSample::phi_p | GibbsSample::theta | GibbsSample::theta_r
         | GibbsSample::theta_p | GibbsSample::spot_scaling
         | GibbsSample::experiment_scaling | GibbsSample::merge_split;
}

inline bool flagged(GibbsSample x) {
  return (GibbsSample::empty | x) != GibbsSample::empty;
}

struct Paths {
  Paths(const std::string &prefix, const std::string &suffix = "");
  std::string phi, theta, spot, experiment, r_phi, p_phi, r_theta, p_theta;
  std::string contributions_gene_type, contributions_spot_type,
      contributions_spot, contributions_experiment;
};

struct VariantModel {
  /** number of genes */
  size_t G;
  // const size_t G;
  /** number of samples */
  size_t S;
  // const size_t S;
  /** number of factors */
  size_t T;
  // const size_t T;
  /** number of experiments */
  size_t E;
  // const size_t E;

  Hyperparameters hyperparameters;
  Parameters parameters;

  /** hidden contributions to the count data due to the different factors */
  IMatrix contributions_gene_type, contributions_spot_type;
  IVector contributions_spot, contributions_experiment;

  /** Normalizing factor to translate Poisson rates \lambda_{xgst} to relative
   * frequencies \lambda_{gst} / z_{gs} for the multionomial distribution */
  Matrix lambda_gene_spot;

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
  Matrix r_phi;
  /** scale parameter for the prior of the loading matrix */
  /* Stored as negative-odds */
  Matrix p_phi;

  /** shape parameter for the prior of the mixing matrix */
  Vector r_theta;
  /** scale parameter for the prior of the mixing matrix */
  /* Stored as negative-odds */
  Vector p_theta;

  Verbosity verbosity;

  VariantModel(const Counts &counts, const size_t T,
               const Hyperparameters &hyperparameters,
               const Parameters &parameters, Verbosity verbosity);

  VariantModel(const Counts &counts, const Paths &paths,
               const Hyperparameters &hyperparameters,
               const Parameters &parameters, Verbosity verbosity);

  void initialize_r_phi();
  void initialize_p_phi();
  void initialize_phi();
  void initialize_r_theta();
  void initialize_p_theta();
  void initialize_theta();

  void store(const Counts &counts, const std::string &prefix,
             bool mean_and_variance = false) const;

  Matrix weighted_theta() const;

  double log_likelihood(const IMatrix &counts) const;
  double log_likelihood_factor(const IMatrix &counts, size_t t) const;
  double log_likelihood_poisson_counts(const IMatrix &counts) const;

  /** sample count decomposition */
  void sample_contributions(const IMatrix &counts);
  void sample_contributions_sub(const IMatrix &counts, size_t g, size_t s,
                                RNG &rng, IMatrix &contrib_gene_type,
                                IMatrix &contrib_spot_type);

  /** sample phi */
  void sample_phi();
  Float sample_phi_sub(size_t g, size_t t, Float theta_t, RNG &rng) const;

  /** sample phi_p and phi_r */
  void sample_p_and_r();

  /** sample theta */
  std::vector<Float> compute_intensities_gene_type() const;
  void sample_theta();

  /** sample p_theta and r_theta */
  void sample_p_and_r_theta();

  /** sample spot scaling factors */
  void sample_spot_scaling();

  /** sample experiment scaling factors */
  void sample_experiment_scaling(const Counts &data);

  /** sample each of the variables from their conditional posterior */
  void gibbs_sample(const Counts &data, GibbsSample which, bool timing);

  void sample_split_merge(const Counts &data, GibbsSample which);
  void sample_merge(const Counts &data, size_t t1, size_t t2,
                    GibbsSample which);
  void sample_split(const Counts &data, size_t t, GibbsSample which);
  void lift_sub_model(const VariantModel &sub_model, size_t t1, size_t t2);

  VariantModel run_submodel(size_t t, size_t n, const Counts &counts,
                            GibbsSample which, const std::string &prefix,
                            const std::vector<size_t> &init_factors
                            = std::vector<size_t>());

  size_t find_weakest_factor() const;

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
