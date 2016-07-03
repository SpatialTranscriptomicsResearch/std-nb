#ifndef NHDP_HPP
#define NHDP_HPP

#include "types.hpp"
#include "hierarchical_kmeans.hpp"

namespace PoissonFactorization {

struct nHDP {
  struct Parameters {
    Float feature_alpha = 1;

    Float mix_alpha = 1;
    Float mix_beta = 1;

    Float tree_alpha = 0.001;
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
  /** descendants' marginals of hidden count contributions by the different
   * factors */
  Matrix desc_counts_gene_type, desc_counts_spot_type;

  Vector counts_type;

  nHDP(size_t g, size_t s, size_t t, const Parameters &params);

  void add_hierarchy(size_t t, const Hierarchy &hierarchy, double concentration);

  std::vector<size_t> parent_of;
  std::vector<std::vector<size_t>> children_of;

  size_t add_node(size_t parent);

  size_t sample_type(size_t g, size_t s) const;
  void register_read(size_t g, size_t s);

  std::string to_dot() const;
};
}

#endif
