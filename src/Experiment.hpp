#ifndef EXPERIMENT_HPP
#define EXPERIMENT_HPP

#include <random>
#include "Theta.hpp"
#include "compression.hpp"
#include "counts.hpp"
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

  /** local features */
  Matrix phi_l;
  /** local feature baseline */
  Vector phi_b;

  /** factor score matrix */
  Matrix theta;
  Matrix field;

  /** spot scaling vector */
  Vector spot;

  /** hidden contributions to the count data due to the different factors */
  Matrix contributions_gene_type, contributions_spot_type;
  Vector contributions_gene, contributions_spot;

  Experiment(Model *model, const Counts &counts, size_t T,
             const Parameters &parameters);

  void enforce_positive_parameters();

  void store(const std::string &prefix, const std::vector<size_t> &order) const;
  void restore(const std::string &prefix);

  Matrix log_likelihood() const;

  Matrix field_fitness_posterior() const;
  Matrix field_fitness_posterior_gradient() const;

  /** sample count decomposition */
  Matrix sample_contributions_gene(size_t g, RNG &rng);
  Matrix sample_contributions_spot(size_t s, RNG &rng);
  Vector sample_contributions_gene_spot(size_t g, size_t s, RNG &rng) const;

  Vector marginalize_genes() const;
  Vector marginalize_spots() const;

  // computes a matrix M(g,t)
  // with M(g,t) = baseline_phi(g) global_phi(g,t) phi(g,t) sum_s theta(s,t)
  // sigma(s)
  Matrix expected_gene_type() const;
  // computes a matrix M(s,t)
  // with M(s,t) = theta(s,t) sigma(s) sum_g baseline_phi(g) phi(g,t)
  // global_phi(g,t)
  Matrix expected_spot_type() const;

  size_t size() const;
  void set_zero();
  Vector vectorize() const;
  template <typename Iter>
  void from_log_vector(Iter &iter) {
    if (parameters.targeted(Target::local)) {
      LOG(debug) << "Getting local R from vector";
      for (auto &x : phi_l)
        x = exp(*iter++);
    }
    if (parameters.targeted(Target::baseline)) {
      LOG(debug) << "Getting baseline R from vector";
      for (auto &x : phi_b)
        x = exp(*iter++);
    }
    if (parameters.targeted(Target::theta)) {
      LOG(debug) << "Getting theta from vector";
      for (auto &x : theta)
        x = exp(*iter++);
    }
    if (parameters.targeted(Target::field)) {
      LOG(debug) << "Getting global field from vector";
      for (auto &x : field)
        x = exp(*iter++);
    }
    if (parameters.targeted(Target::spot)) {
      LOG(debug) << "Getting spots from vector";
      for (auto &x : spot)
        x = exp(*iter++);
    }
  }
  std::vector<std::vector<size_t>> active_factors(double threshold = 1.0) const;
};

std::ostream &operator<<(std::ostream &os, const Experiment &experiment);

Experiment operator*(const Experiment &a, const Experiment &b);
Experiment operator+(const Experiment &a, const Experiment &b);
Experiment operator-(const Experiment &a, const Experiment &b);
Experiment operator*(const Experiment &a, double x);
Experiment operator/(const Experiment &a, double x);
Experiment operator-(const Experiment &a, double x);
}

#endif
