#include <omp.h>
#include <boost/tokenizer.hpp>
#include "VariantModel.hpp"
#include "montecarlo.hpp"

using namespace std;
namespace PoissonFactorization {

void print_matrix_head(ostream &os, const Matrix &m, const std::string &label,
                       size_t n) {
  if (label != "")
    os << label << std::endl;
  size_t X = m.n_rows;
  size_t Y = m.n_cols;
  for (size_t x = 0; x < std::min<size_t>(X, n); ++x) {
    for (size_t y = 0; y < Y; ++y)
      os << (y > 0 ? "\t" : "") << m(x, y);
    os << std::endl;
  }

  if (label != "")
    os << label << " ";
  os << "column sums" << std::endl;

  size_t zeros = 0;
  for (size_t y = 0; y < Y; ++y) {
    double sum = 0;
    for (size_t x = 0; x < X; ++x) {
      if (m(x, y) == 0)
        zeros++;
      sum += m(x, y);
    }
    os << (y > 0 ? "\t" : "") << sum;
  }
  os << std::endl;

  os << "There are " << zeros << " zeros";
  if (label != "")
    os << " in " << label;
  os << ". This corresponds to " << (100.0 * zeros / X / Y) << "%."
     << std::endl;
}

// extern size_t sub_model_cnt = 0; // TODO

namespace PRIOR {
namespace PHI {
Gamma::Gamma(size_t G_, size_t S_, size_t T_, const Parameters &params)
    : G(G_), S(S_), T(T_), r(G, T), p(G, T), parameters(params) {
  initialize_r();
  initialize_p();
}

Gamma::Gamma(const Gamma &other)
    : G(other.G),
      S(other.S),
      T(other.T),
      r(other.r),
      p(other.p),
      parameters(other.parameters) {}

void Gamma::initialize_r() {
  // initialize r_phi
  // if (verbosity >= Verbosity::Debug) // TODO-verbosity
  std::cout << "initializing r_phi." << std::endl;
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t g = 0; g < G; ++g) {
    const size_t thread_num = omp_get_thread_num();
    for (size_t t = 0; t < T; ++t)
      // NOTE: std::gamma_distribution takes a shape and scale parameter
      r(g, t) = std::gamma_distribution<Float>(
          parameters.hyperparameters.phi_r_1,
          1 / parameters.hyperparameters.phi_r_2)(
          EntropySource::rngs[thread_num]);
  }
}
void Gamma::initialize_p() {
  // initialize p_phi
  // if (verbosity >= Verbosity::Debug) // TODO-verbosity
  std::cout << "initializing p_phi." << std::endl;
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t g = 0; g < G; ++g) {
    const size_t thread_num = omp_get_thread_num();
    for (size_t t = 0; t < T; ++t)
      p(g, t) = prob_to_neg_odds(sample_beta<Float>(
          parameters.hyperparameters.phi_p_1,
          parameters.hyperparameters.phi_p_2, EntropySource::rngs[thread_num]));
  }
}

void Gamma::sample(const Matrix &theta, const IMatrix &contributions_gene_type,
                   const Vector &spot_scaling,
                   const Vector &experiment_scaling_long) {
  Verbosity verbosity = Verbosity::Info;
  if (verbosity >= Verbosity::Verbose)  // TODO-verbosity
    std::cout << "Sampling P and R" << std::endl;

  auto gen = [&](const std::pair<Float, Float> &x, std::mt19937 &rng) {
    std::normal_distribution<double> rnorm;
    const double f1 = exp(rnorm(rng));
    const double f2 = exp(rnorm(rng));
    return std::pair<Float, Float>(f1 * x.first, f2 * x.second);
  };

  for (size_t t = 0; t < T; ++t) {
    Float weight_sum = 0;
    for (size_t s = 0; s < S; ++s) {
      Float x = theta(s, t) * spot_scaling[s];
      if (parameters.activate_experiment_scaling)
        x *= experiment_scaling_long[s];
      weight_sum += x;
    }
    MetropolisHastings mh(parameters.temperature, parameters.prop_sd,
                          verbosity);

#pragma omp parallel for if (DO_PARALLEL)
    for (size_t g = 0; g < G; ++g) {
      const Int count_sum = contributions_gene_type(g, t);
      const size_t thread_num = omp_get_thread_num();
      auto res = mh.sample(std::pair<Float, Float>(r(g, t), p(g, t)),
                           parameters.n_iter, EntropySource::rngs[thread_num],
                           gen, compute_conditional, count_sum, weight_sum,
                           parameters.hyperparameters);
      r(g, t) = res.first;
      p(g, t) = res.second;
    }
  }
}

void Gamma::store(const std::string &prefix,
                  const std::vector<std::string> &gene_names,
                  const std::vector<std::string> &factor_names) const {
  write_matrix(r, prefix + "r_phi.txt", gene_names, factor_names);
  write_matrix(p, prefix + "p_phi.txt", gene_names, factor_names);
}

void Gamma::lift_sub_model(const Gamma &sub_model, size_t t1, size_t t2) {
  for (size_t g = 0; g < G; ++g) {
    r(g, t1) = sub_model.r(g, t2);
    p(g, t1) = sub_model.p(g, t2);
  }
}

Dirichlet::Dirichlet(size_t G_, size_t S_, size_t T_,
                     const Parameters &parameters)
    : G(G_),
      S(S_),
      T(T_),
      alpha_prior(parameters.hyperparameters.alpha),
      alpha(G, alpha_prior) {}

Dirichlet::Dirichlet(const Dirichlet &other)
    : G(other.G),
      S(other.S),
      T(other.T),
      alpha_prior(other.alpha_prior),
      alpha(other.alpha) {}

void Dirichlet::sample(const Matrix &theta,
                       const IMatrix &contributions_gene_type,
                       const Vector &spot_scaling,
                       const Vector &experiment_scaling_long) const {}

void Dirichlet::store(const std::string &prefix,
                      const std::vector<std::string> &gene_names,
                      const std::vector<std::string> &factor_names) const {}

void Dirichlet::lift_sub_model(const Dirichlet &sub_model, size_t t1, size_t t2) const {}

ostream &operator<<(ostream &os, const Gamma &x) {
  print_matrix_head(os, x.r, "R of Φ");
  print_matrix_head(os, x.p, "P of Φ");
  return os;
}

ostream &operator<<(ostream &os, const Dirichlet &x) {
  // do nothing, as Dirichlet class does not have random variables
  return os;
}
}
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
