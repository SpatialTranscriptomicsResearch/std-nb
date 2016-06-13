#include <boost/tokenizer.hpp>
#include "Model.hpp"

using namespace std;
namespace PoissonFactorization {

// extern size_t sub_model_cnt = 0; // TODO

bool gibbs_test(Float nextG, Float G, Float temperature) {
  double dG = nextG - G;
  double r = RandomDistribution::Uniform(EntropySource::rng);
  double p = std::min<double>(1.0, MetropolisHastings::boltzdist(-dG, temperature));
  LOG(debug) << "T = " << temperature << " nextG = " << nextG << " G = " << G
      << " dG = " << dG << " p = " << p << " r = " << r;
  if (std::isnan(nextG) == 0 and (dG > 0 or r <= p)) {
    LOG(verbose) << "Accepted!";
    return true;
  } else {
    LOG(verbose) << "Rejected!";
    return false;
  }
}

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
      contributions_spot(prefix + "contributions_spot.txt" + suffix),
      contributions_experiment(prefix + "contributions_experiment.txt"
                               + suffix){};

size_t num_lines(const string &path) {
  int number_of_lines = 0;
  string line;
  ifstream ifs(path);

  while (getline(ifs, line))
    ++number_of_lines;
  return number_of_lines;
}
}
