#ifndef NHDP_HPP
#define NHDP_HPP

#include "types.hpp"

namespace PoissonFactorization {

struct nHDP {
  static constexpr Float feature_alpha = 1;
  static constexpr Float feature_beta = 1;

  static constexpr Float mix_alpha = 1;
  static constexpr Float mix_beta = 1;

  static constexpr Float tree_alpha = 1;

  /** number of genes */
  size_t G;
  /** number of samples */
  size_t S;
  /** number of factors */
  size_t T;

  /** marginals of hidden count contributions by the different factors */
  IMatrix counts_gene_type, counts_spot_type;
  /** descendants' marginals of hidden count contributions by the different factors */
  IMatrix desc_counts_spot_type;

  IVector counts_type;

  nHDP(size_t g, size_t s, size_t t);

  size_t parent_of(size_t t) const;
  std::vector<size_t> children_of(size_t t) const;

  size_t sample_type(size_t g, size_t s) const;
  void register_read(size_t g, size_t s);
};
}

#endif
