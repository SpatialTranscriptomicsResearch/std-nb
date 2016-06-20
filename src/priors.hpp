#ifndef PRIORS_HPP
#define PRIORS_HPP

#include <cstddef>
#include "types.hpp"
#include "parameters.hpp"

namespace PoissonFactorization {
namespace PRIOR {
namespace PHI {

struct Gamma {
  size_t G, S, T;
  /** shape parameter for the prior of the loading matrix */
  Matrix r;
  /** scale parameter for the prior of the loading matrix */
  /* Stored as negative-odds */
  Matrix p;
  Parameters parameters;

  Gamma(size_t G_, size_t S_, size_t T_, const Parameters &params);
  Gamma(const Gamma &other);
  /** sample p_phi and r_phi */
  /* This is a simple Metropolis-Hastings sampling scheme */
  void sample(const Matrix &theta, const IMatrix &contributions_gene_type,
              const Vector &spot_scaling,
              const Vector &experiment_scaling_long);

  void store(const std::string &prefix,
             const std::vector<std::string> &gene_names,
             const std::vector<std::string> &factor_names) const;

  void lift_sub_model(const Gamma &sub_model, size_t t1, size_t t2);

private:
  void initialize_r();
  void initialize_p();
};

struct Dirichlet {
  size_t G, S, T;
  Float alpha_prior;
  Matrix alpha;

  Dirichlet(size_t G_, size_t S_, size_t T_, const Parameters &parameters);
  Dirichlet(const Dirichlet &other);
  /** This routine does nothing, as this sub-model doesn't have random variables
   * but only hyper-parameters */
  void sample(const Matrix &theta, const IMatrix &contributions_gene_type,
              const Vector &spot_scaling,
              const Vector &experiment_scaling_long) const;
  void store(const std::string &prefix,
             const std::vector<std::string> &gene_names,
             const std::vector<std::string> &factor_names) const;

  void lift_sub_model(const Dirichlet &sub_model, size_t t1, size_t t2) const;
};

/** This routine doesn't print, for the same reason as sample() does nothing */
std::ostream &operator<<(std::ostream &os, const Gamma &x);
std::ostream &operator<<(std::ostream &os, const Dirichlet &x);
}

namespace THETA {

struct Gamma {
  size_t G, S, T;
  /** shape parameter for the prior of the mixing matrix */
  Vector r;
  /** scale parameter for the prior of the mixing matrix */
  /* Stored as negative-odds */
  Vector p;
  Parameters parameters;

  Gamma(size_t G_, size_t S_, size_t T_, const Parameters &params);
  Gamma(const Gamma &other);
  /** sample p_phi and r_phi */
  /* This is a simple Metropolis-Hastings sampling scheme */
  void sample(const Matrix &phi, const IMatrix &contributions_spot_type,
              const Vector &spot_scaling,
              const Vector &experiment_scaling_long);

  void store(const std::string &prefix,
             const std::vector<std::string> &spot_names,
             const std::vector<std::string> &factor_names) const;

  void lift_sub_model(const Gamma &sub_model, size_t t1, size_t t2);

private:
  void initialize_r();
  void initialize_p();
};

struct Dirichlet {
  size_t G, S, T;
  Float alpha_prior;
  std::vector<Float> alpha;

  Dirichlet(size_t G_, size_t S_, size_t T_, const Parameters &parameters);
  Dirichlet(const Dirichlet &other);
  /** This routine does nothing, as this sub-model doesn't have random variables
   * but only hyper-parameters */
  void sample(const Matrix &phi, const IMatrix &contributions_spot_type,
              const Vector &spot_scaling,
              const Vector &experiment_scaling_long) const;
  void store(const std::string &prefix,
             const std::vector<std::string> &spot_names,
             const std::vector<std::string> &factor_names) const;

  void lift_sub_model(const Dirichlet &sub_model, size_t t1, size_t t2) const;
};

/** This routine doesn't print, for the same reason as sampl() does nothing */
std::ostream &operator<<(std::ostream &os, const Gamma &x);
std::ostream &operator<<(std::ostream &os, const Dirichlet &x);
}
}
}

#endif
