#ifndef NHDP_HPP
#define NHDP_HPP

#include <random>
#include "hierarchical_kmeans.hpp"
#include "types.hpp"

namespace PoissonFactorization {

struct HDP {
  struct Parameters {
    Float feature_alpha = 1;
    Float mix_alpha = 1;
    Float mix_alpha_zero = 1;
  };

  /** number of genes */
  size_t G;
  /** number of samples */
  size_t S;
  /** number of factors */
  size_t T;

  /** maximal number of factors */
  size_t maxT;

  Parameters parameters;

  /** marginals of hidden count contributions by the different factors */
  Matrix counts_gene_type, counts_spot_type;
  Vector counts_type, desc_counts_type;

  HDP(size_t g, size_t s, size_t t, const Parameters &params);

  size_t add_factor();

  Vector compute_prior(const Vector &v) const;
  size_t sample_type(size_t g, size_t s) const;
  void register_read(size_t g, size_t s);
  void register_read(size_t g, size_t s, size_t t, size_t n);

  HDP sample(const IMatrix &counts) const;
  Matrix sample_gene_expression() const;

  HDP &operator+=(const HDP &m);
};
}

#endif
