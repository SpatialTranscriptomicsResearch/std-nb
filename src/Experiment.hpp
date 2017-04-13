#ifndef EXPERIMENT_HPP
#define EXPERIMENT_HPP

#include <random>
#include "Paths.hpp"
#include "Theta.hpp"
#include "compression.hpp"
#include "counts.hpp"
#include "entropy.hpp"
#include "hamiltonian_monte_carlo.hpp"
#include "metropolis_hastings.hpp"
#include "odds.hpp"
#include "parameters.hpp"
#include "stats.hpp"
#include "target.hpp"
#include "timer.hpp"
#include "verbosity.hpp"

namespace STD {

#ifdef NDEBUG
const bool noisy = false;
#else
const bool noisy = true;
#endif

struct Model;

struct Experiment {
  Model *model;

  /** number of genes */
  size_t G;
  /** number of samples */
  size_t S;
  /** number of factors */
  size_t T;

  Counts counts;
  Matrix coords;

  Parameters parameters;

  /** hidden contributions to the count data due to the different factors */
  Matrix contributions_gene_type, contributions_spot_type;
  Vector contributions_gene, contributions_spot;

  /** local features */
  Matrix phi_l;
  /** local feature baseline */
  Matrix phi_b;

  /** factor score matrix */
  Theta weights;
  Matrix field;

  /** spot scaling vector */
  Vector spot;

  Experiment(Model *model, const Counts &counts, size_t T,
             const Parameters &parameters);

  void enforce_positive_parameters();

  void store(const std::string &prefix, const std::vector<size_t> &order) const;
  void restore(const std::string &prefix);

  Matrix log_likelihood() const;

  inline Float &theta(size_t s, size_t t) { return weights.matrix(s, t); };
  inline Float theta(size_t s, size_t t) const { return weights.matrix(s, t); };

  Matrix field_fitness_posterior(const Matrix &candidate_field) const;
  Matrix field_fitness_posterior_gradient(const Matrix &candidate_field) const;

  /** sample count decomposition */
  Matrix sample_contributions_gene(size_t g, RNG &rng);
  Matrix sample_contributions_spot(size_t s, RNG &rng);
  Vector sample_contributions_gene_spot(size_t g, size_t s, RNG &rng) const;

  Vector marginalize_genes() const;
  Vector marginalize_spots() const;

  // computes a matrix M(g,t)
  // with M(g,t) = baseline_phi(g) global_phi(g,t) sum_s theta(s,t) sigma(s)
  Matrix explained_gene_type() const;
  // computes a matrix M(g,t)
  // with M(g,t) = baseline_phi(g) global_phi(g,t) phi(g,t) sum_s theta(s,t)
  // sigma(s)
  Matrix expected_gene_type() const;
  // computes a matrix M(s,t)
  // with M(s,t) = sigma(s) sum_g baseline_phi(g) phi(g,t) global_phi(g,t)
  Matrix explained_spot_type() const;
  // computes a matrix M(s,t)
  // with M(s,t) = theta(s,t) sigma(s) sum_g baseline_phi(g) phi(g,t)
  // global_phi(g,t)
  Matrix expected_spot_type() const;
  // computes a vector V(g)
  // with V(g) = sum_t phi(g,t) global_phi(g,t) sum_s theta(s,t) sigma(s)
  Vector explained_gene() const;

  std::vector<std::vector<size_t>> active_factors(double threshold = 1.0) const;
};

/* -------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------- */

inline double deriv_log_nb_r(double k, double r, double p, double theta) {
  const double prod = r * theta;
  const double digamma_diff
      = (prod + k == prod)
            ? 0
            // TODO if prod is low then use the sum of logs expression for the
            // difference of gammas
            : digamma(prod + k) - digamma(prod);

  return theta * (digamma_diff + log(1 - p));
}

inline double deriv_log_nb_p(double k, double r, double p, double theta) {
  return k / p - r * theta / (1 - p);
}

/* -------------------------------------------------------------------------- */

inline double curv_log_nb_r(double k, double r, double p, double theta) {
  const double prod = r * theta;
  const double trigamma_diff
      = (prod + k == prod)
            ? 0
            // TODO if prod is low then use the sum of logs expression for the
            // difference of gammas
            : trigamma(prod + k) - trigamma(prod);

  return theta * theta * trigamma_diff;
}

inline double curv_log_nb_rp(double k, double r, double p, double theta) {
  return -theta * (1 - p);
}

inline double curv_log_nb_p(double k, double r, double p, double theta) {
  return -r * theta / (1 - p) / (1 - p) - k / p / p;
}

/* -------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------- */

inline double deriv_log_nb_exp_mu(double k, double mu, double nu,
                                  double theta) {
  return mu * mu
         * ((2 * nu - mu) * theta
                * (digamma((mu * mu * theta + k * nu - k * mu) / (nu - mu))
                   - digamma((mu * mu * theta) / (nu - mu)))
            + theta * (nu * (2 * log(mu / nu) + 1) - mu * (log(mu / nu) + 1))
            - k * (nu / mu - 1))
         / (nu - mu) / (nu - mu);
}

inline double deriv_log_nb_exp_nu(double k, double mu, double nu,
                                  double theta) {
  return mu * mu
         * (nu * theta * (digamma((mu * mu * theta + k * nu - k * mu) / nu - mu)
                          - digamma(mu * mu * theta / (nu - mu)))
            + ((log(mu / nu) + 1) * nu - mu) * theta - k * nu / mu + k)
         / (nu - mu) / (nu - mu);
}

inline double deriv_log_nb_exp_mu_nu(double k, double mu, double nu,
                                     double theta) {
  return mu * mu
         * (nu * theta * (digamma((mu * mu * theta + k * nu - k * mu) / nu - mu)
                          - digamma(mu * mu * theta / (nu - mu)))
            + ((log(mu / nu) + 1) * nu - mu) * theta - k * nu / mu + k)
         / (nu - mu) / (nu - mu);
}

inline double deriv_prior_nb_mu(double mu, double nu,
                                const Hyperparameters &params) {
  const double a = params.phi_r_1;
  const double b = params.phi_r_2;
  const double alpha = params.phi_p_1;
  const double beta = params.phi_p_2;
  return ((beta + 2 * a - 3) * nu * nu
          + ((-2 * beta - alpha - 3 * a + 6) * mu - 2 * b * mu * mu) * nu
          + b * mu * mu * mu + (beta + alpha + a - 3) * mu * mu)
         / (nu - mu) / (nu - mu);
}

inline double deriv_prior_nb_nu(double mu, double nu,
                                const Hyperparameters &params) {
  const double a = params.phi_r_1;
  const double b = params.phi_r_2;
  const double alpha = params.phi_p_1;
  const double beta = params.phi_p_2;
  return ((beta + 2 * a - 3) * nu * nu
          + ((-2 * beta - alpha - a + 4) * mu - b * mu * mu) * nu
          + mu * mu * (beta + alpha + a - 3))
         / (nu - mu) / (nu - mu);
}

inline double mean_NB_rp(double r, double p) { return r * p / (1 - p); }

inline double var_NB_rp(double r, double p) {
  return r * p / (1 - p) / (1 - p);
}

inline double mean_NB_rno(double r, double no) { return r / no; }

inline double var_NB_rno(double r, double no) {
  return var_NB_rp(r, neg_odds_to_prob(no));
}

inline double lgamma_diff(double a, double b) {
  double s = a + b;
  if (s != a)
    return lgamma(s) - lgamma(a);
  else
    return 0;
}

inline double lgamma_diff_1p(double a, double b) {
  double s = a + b;
  if (s != a + 1)
    return lgamma(s) - lgamma(a + 1);
  else
    return 0;
}

inline double digamma_diff(double a, double b) {
  double s = a + b;
  if (s != a)
    return digamma(s) - digamma(a);
  else
    return 0;
}

inline double trigamma_diff(double a, double b) {
  double s = a + b;
  if (s != a)
    return trigamma(s) - trigamma(a);
  else
    return 0;
}

std::ostream &operator<<(std::ostream &os, const Experiment &experiment);

Experiment operator*(const Experiment &a, const Experiment &b);
Experiment operator+(const Experiment &a, const Experiment &b);
Experiment operator-(const Experiment &a, const Experiment &b);
Experiment operator*(const Experiment &a, double x);
Experiment operator/(const Experiment &a, double x);
Experiment operator-(const Experiment &a, double x);
}

#endif
