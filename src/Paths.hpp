#ifndef PATHS_HPP
#define PATHS_HPP

#include <string>

namespace PoissonFactorization {

struct Paths {
  Paths(const std::string &prefix, const std::string &suffix = "");
  std::string phi, theta, spot, experiment, r_phi, p_phi, r_theta, p_theta;
  std::string contributions_gene_type, contributions_spot_type,
      contributions_gene, contributions_spot, contributions_experiment;
};
}

#endif
