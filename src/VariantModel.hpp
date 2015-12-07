#ifndef VARIANTMODEL_HPP
#define VARIANTMODEL_HPP

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

  Priors priors;
  Parameters parameters;

  ITensor contributions;

  /** factor loading matrix */
  Matrix phi;

  /** factor score matrix */
  Matrix theta;

  /** shape parameter for the prior of the loading matrix */
  Matrix r;
  /** scale parameter for the prior of the loading matrix */
  Matrix p;

  Verbosity verbosity;

  VariantModel(const IMatrix &counts, const size_t T, const Priors &priors,
               const Parameters &parameters, Verbosity verbosity);

  double log_likelihood(const IMatrix &counts) const;

  /** sample count decomposition */
  void sample_contributions(const IMatrix &counts);

  /** sample phi */
  void sample_phi();

  /** sample p */
  void sample_p();

  /** sample r */
  void sample_r();

  /** sample theta */
  void sample_theta();

  /** sample each of the variables from their conditional posterior */
  void gibbs_sample(const IMatrix &counts);
};
}

std::ostream &operator<<(std::ostream &os,
                         const FactorAnalysis::VariantModel &pfa);
#endif
