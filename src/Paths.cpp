#include "Paths.hpp"

namespace PoissonFactorization {
Paths::Paths(const std::string &prefix, const std::string &suffix)
    : phi(prefix + "phi.txt" + suffix),
      theta(prefix + "theta.txt" + suffix),
      spot(prefix + "spot_scaling.txt" + suffix),
      experiment(prefix + "experiment_scaling.txt" + suffix),
      r_phi(prefix + "r_phi.txt" + suffix),
      p_phi(prefix + "p_phi.txt" + suffix),
      r_theta(prefix + "r_theta.txt" + suffix),
      p_theta(prefix + "p_theta.txt" + suffix),
      contributions_gene_type(prefix + "contributions_gene_type.txt" + suffix),
      contributions_spot_type(prefix + "contributions_spot_type.txt" + suffix),
      contributions_gene(prefix + "contributions_gene.txt" + suffix),
      contributions_spot(prefix + "contributions_spot.txt" + suffix),
      contributions_experiment(prefix + "contributions_experiment.txt"
                               + suffix){};
}
