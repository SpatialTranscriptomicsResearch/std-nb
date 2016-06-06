#include <omp.h>
#include <boost/tokenizer.hpp>
#include "VariantModel.hpp"
#include "montecarlo.hpp"

using namespace std;
namespace PoissonFactorization {

// extern size_t sub_model_cnt = 0; // TODO

bool gibbs_test(Float nextG, Float G, Verbosity verbosity, Float temperature) {
  double dG = nextG - G;
  double r = RandomDistribution::Uniform(EntropySource::rng);
  double p = std::min<double>(1.0, MCMC::boltzdist(-dG, temperature));
  if (verbosity >= Verbosity::Verbose)
    std::cerr << "T = " << temperature << " nextG = " << nextG << " G = " << G
      << " dG = " << dG << " p = " << p << " r = " << r << std::endl;
  if (std::isnan(nextG) == 0 and (dG > 0 or r <= p)) {
    if (verbosity >= Verbosity::Verbose)
      std::cerr << "Accepted!" << std::endl;
    return true;
  } else {
    if (verbosity >= Verbosity::Verbose)
      std::cerr << "Rejected!" << std::endl;
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
