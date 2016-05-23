#include <omp.h>
#include <boost/tokenizer.hpp>
#include "VariantModel.hpp"
#include "montecarlo.hpp"

using namespace std;
namespace FactorAnalysis {
// extern size_t sub_model_cnt = 0; // TODO

std::ostream &operator<<(std::ostream &os, const GibbsSample &which) {
  if(which == GibbsSample::empty) {
    os << "empty";
    return os;
  } else {
    bool first = true;
    if(flagged(which & GibbsSample::contributions)) {
      os << "contributions";
      first = false;
    }
    if(flagged(which & GibbsSample::phi)) {
      os << (first ? "" : "," ) << "phi";
      first = false;
    }
    if(flagged(which & GibbsSample::phi_r)) {
      os << (first ? "" : "," ) << "phi_r";
      first = false;
    }
    if(flagged(which & GibbsSample::phi_p)) {
      os << (first ? "" : "," ) << "phi_p";
      first = false;
    }
    if(flagged(which & GibbsSample::theta)) {
      os << (first ? "" : "," ) << "theta";
      first = false;
    }
    if(flagged(which & GibbsSample::theta_p)) {
      os << (first ? "" : "," ) << "theta_p";
      first = false;
    }
    if(flagged(which & GibbsSample::theta_r)) {
      os << (first ? "" : "," ) << "theta_r";
      first = false;
    }
    if(flagged(which & GibbsSample::spot_scaling)) {
      os << (first ? "" : "," ) << "spot_scaling";
      first = false;
    }
    if(flagged(which & GibbsSample::experiment_scaling)) {
      os << (first ? "" : "," ) << "experiment_scaling";
      first = false;
    }
    if(flagged(which & GibbsSample::merge_split)) {
      os << (first ? "" : "," ) << "merge_split";
      first = false;
    }

  }
  return os;
}

std::istream &operator>>(std::istream &is, GibbsSample &which) {
  which = GibbsSample::empty;
  using tokenizer = boost::tokenizer<boost::char_separator<char>>;
  boost::char_separator<char> sep(",");

  string line;
  getline(is, line);
  tokenizer tok(line, sep);
  for (auto token : tok) {
    if(token == "contributions")
      which = which | GibbsSample::contributions;
    else if(token == "phi")
      which = which | GibbsSample::phi;
    else if(token == "phi_r")
      which = which | GibbsSample::phi_r;
    else if(token == "phi_p")
      which = which | GibbsSample::phi_p;
    else if(token == "theta")
      which = which | GibbsSample::theta;
    else if(token == "theta_r")
      which = which | GibbsSample::theta_r;
    else if(token == "theta_p")
      which = which | GibbsSample::theta_p;
    else if(token == "spot_scaling")
      which = which | GibbsSample::spot_scaling;
    else if(token == "experiment_scaling")
      which = which | GibbsSample::experiment_scaling;
    else if(token == "merge_split")
      which = which | GibbsSample::merge_split;
    else
      throw(std::runtime_error("Unknown sampling token: " + token));
  }
  return is;
}

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

double compute_conditional(const pair<Float, Float> &x, Int count_sum,
                           Float weight_sum,
                           const Hyperparameters &hyperparameters) {
  const Float current_r = x.first;
  const Float current_p = x.second;
  return log_beta_neg_odds(current_p, hyperparameters.phi_p_1,
                           hyperparameters.phi_p_2)
         // NOTE: gamma_distribution takes a shape and scale parameter
         + log_gamma(current_r, hyperparameters.phi_r_1,
                     1 / hyperparameters.phi_r_2)
         // The next lines are part of the negative binomial distribution.
         // The other factors aren't needed as they don't depend on either of
         // r[g][t] and p[g][t], and thus would cancel when computing the score
         // ratio.
         + current_r * log(current_p)
         - (current_r + count_sum) * log(current_p + weight_sum)
         + lgamma(current_r + count_sum) - lgamma(current_r);
}


}
