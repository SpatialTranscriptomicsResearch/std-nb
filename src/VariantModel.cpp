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

double compute_conditional_theta(const pair<Float, Float> &x,
                                 const vector<Int> &count_sums,
                                 const vector<Float> &weight_sums,
                                 const Hyperparameters &hyperparameters) {
  const size_t S = count_sums.size();
  const Float current_r = x.first;
  const Float current_p = x.second;
  double r = log_beta_neg_odds(current_p, hyperparameters.theta_p_1,
                               hyperparameters.theta_p_2)
             // NOTE: gamma_distribution takes a shape and scale parameter
             + log_gamma(current_r, hyperparameters.theta_r_1,
                         1 / hyperparameters.theta_r_2)
             + S * (current_r * log(current_p) - lgamma(current_r));
#pragma omp parallel for reduction(+ : r) if (DO_PARALLEL)
  for (size_t s = 0; s < S; ++s)
    // The next line is part of the negative binomial distribution.
    // The other factors aren't needed as they don't depend on either of
    // r[t] and p[t], and thus would cancel when computing the score
    // ratio.
    r += lgamma(current_r + count_sums[s])
         - (current_r + count_sums[s]) * log(current_p + weight_sums[s]);
  return r;
}

}
