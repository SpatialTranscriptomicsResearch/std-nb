#ifndef EXPERIMENT_HPP
#define EXPERIMENT_HPP

#include <random>
#include "Theta.hpp"
#include "compression.hpp"
#include "counts.hpp"
#include "covariate.hpp"
#include "entropy.hpp"
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

  std::shared_ptr<Matrix> inv_covariance;

  std::vector<size_t> rate_coeff_idxs;
  std::vector<size_t> odds_coeff_idxs;
  std::vector<size_t> &coeff_idxs(Coefficient::Variable variable);

  Matrix compute_gene_type_table(const std::vector<size_t> &coeff_idxs) const;
  Matrix compute_spot_type_table(const std::vector<size_t> &coeff_idxs) const;

  Matrix field;

  /** hidden contributions to the count data due to the different factors */
  Matrix contributions_gene_type, contributions_spot_type;
  Vector contributions_gene, contributions_spot;

  Experiment(Model *model, const Counts &counts, size_t T,
             const Parameters &parameters);

  void enforce_positive_parameters();

  void store(const std::string &prefix, const std::vector<size_t> &order) const;
  void restore(const std::string &prefix);

  // TODO covariates reactivate likelihood
  // Matrix log_likelihood() const;

  Matrix field_fitness_posterior() const;
  Matrix field_fitness_posterior_gradient() const;

  /** sample count decomposition */
  Vector sample_contributions_gene_spot(size_t g, size_t s,
                                        const Matrix &rate_gt,
                                        const Matrix &rate_st,
                                        const Matrix &odds_gt,
                                        const Matrix &odds_st, RNG &rng) const;

  Vector marginalize_genes() const;

  Matrix expectation() const;
  Matrix variance() const;

  // computes a matrix M(g,t) =
  //   beta(g) gamma(g,t) lambda(g,t) sum_s theta(s,t) sigma(s)
  Matrix expected_gene_type() const;
  // computes a matrix M(s,t) =
  //   theta(s,t) sigma(s) sum_g beta(g) lambda(g,t) gamma(g,t)
  Matrix expected_spot_type() const;

  size_t size() const;
  void setZero();
  Vector vectorize() const;
  template <typename Iter>
  void from_vector(Iter &iter) {
    if (parameters.targeted(Target::field)) {
      LOG(debug) << "Getting local field from vector";
      for (auto &x : field)
        x = *iter++;
    }
  }
};

std::ostream &operator<<(std::ostream &os, const Experiment &experiment);

Experiment operator+(const Experiment &a, const Experiment &b);
}

#endif
