#ifndef FEATURES_HPP
#define FEATURES_HPP

#include "io.hpp"
#include "log.hpp"
#include "parallel.hpp"
#include "pdist.hpp"
#include "priors.hpp"
#include "sampling.hpp"

namespace PoissonFactorization {
namespace Feature {

const Float phi_scaling = 1.0;

enum class Kind {
  Dirichlet,
  Gamma
  //, HierGamma
};

std::ostream &operator<<(std::ostream &os, Kind kind);
std::istream &operator>>(std::istream &is, Kind kind);

template <Kind kind>
struct Traits {};

template <>
struct Traits<Kind::Gamma> {
  typedef PRIOR::PHI::Gamma prior_type;
};

template <>
struct Traits<Kind::Dirichlet> {
  typedef PRIOR::PHI::Dirichlet prior_type;
};

template <Kind kind>
struct Features {
  typedef typename Traits<kind>::prior_type prior_type;
  Features(size_t G_, size_t S_, size_t T_, const Parameters &params)
      : G(G_),
        S(S_),
        T(T_),
        phi(G, T),
        parameters(params),
        prior(G, S, T, parameters) {
    initialize();
  };
  size_t G, S, T;
  Matrix phi;
  Parameters parameters;
  prior_type prior;

  void initialize_factor(size_t t);
  void initialize();
  void sample(const Matrix &theta, const IMatrix &contributions_gene_type,
              const Vector &spot_scaling,
              const Vector &experiment_scaling_long);
  void store(const std::string &prefix,
             const std::vector<std::string> &gene_names,
             const std::vector<std::string> &factor_names) const {
    write_matrix(phi, prefix + "phi.txt", gene_names, factor_names);
    prior.store(prefix, gene_names, factor_names);
  };

  double log_likelihood_factor(const IMatrix &counts, size_t t) const;
  std::vector<Float> marginalize_genes() const {
    std::vector<Float> intensities(T, 0);
#pragma omp parallel for if (DO_PARALLEL)
    for (size_t t = 0; t < T; ++t)
      for (size_t g = 0; g < G; ++g)
        intensities[t] += phi(g, t);
    return intensities;
  };

  void lift_sub_model(const Features<kind> &sub_model, size_t t1, size_t t2) {
    prior.lift_sub_model(sub_model.prior, t1, t2);
    for (size_t g = 0; g < G; ++g)
      phi(g, t1) = sub_model.phi(g, t2);
  }
};

template <>
void Features<Kind::Gamma>::initialize_factor(size_t t);

template <>
void Features<Kind::Dirichlet>::initialize_factor(size_t t);

template <>
void Features<Kind::Gamma>::initialize();

template <>
void Features<Kind::Dirichlet>::initialize();

template <>
double Features<Kind::Gamma>::log_likelihood_factor(const IMatrix &counts,
                                                    size_t t) const;

template <>
double Features<Kind::Dirichlet>::log_likelihood_factor(const IMatrix &counts,
                                                        size_t t) const;

/** sample phi */
template <>
void Features<Kind::Gamma>::sample(const Matrix &theta,
                                   const IMatrix &contributions_gene_type,
                                   const Vector &spot_scaling,
                                   const Vector &experiment_scaling_long);

template <>
void Features<Kind::Dirichlet>::sample(const Matrix &theta,
                                       const IMatrix &contributions_gene_type,
                                       const Vector &spot_scaling,
                                       const Vector &experiment_scaling_long);
}
}

#endif
