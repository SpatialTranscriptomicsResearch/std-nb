#ifndef EXPERIMENT_HPP
#define EXPERIMENT_HPP

#include <random>
#include "compression.hpp"
#include "counts.hpp"
#include "covariate.hpp"
#include "entropy.hpp"
#include "gp.hpp"
#include "odds.hpp"
#include "parameters.hpp"
#include "stats.hpp"
#include "timer.hpp"
#include "verbosity.hpp"

#include "spec_parser/Expression.hpp"

namespace STD {

using Expression = spec_parser::ExpressionPtr<CoefficientPtr>;

#ifdef NDEBUG
const bool noisy = false;
#else
const bool noisy = true;
#endif

const std::string FILENAME_ENDING = ".tsv";

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

  double scale_ratio;

  Parameters parameters;

  std::vector<size_t> rate_coeff_idxs;
  std::vector<size_t> odds_coeff_idxs;

  void ensure_dimensions() const;

  /** hidden contributions to the count data due to the different factors */
  Matrix contributions_gene_type, contributions_spot_type;
  Vector contributions_gene, contributions_spot;

  Experiment(Model *model, const Counts &counts, size_t T,
             const Parameters &parameters);

  void store(const std::string &prefix, const std::vector<size_t> &order) const;
  void restore(const std::string &prefix);

  /** sample count decomposition */
  Vector sample_contributions_gene_spot(size_t g, size_t s, const Vector &rate,
                                        const Vector &odds, RNG &rng) const;
};

std::ostream &operator<<(std::ostream &os, const Experiment &experiment);

Experiment operator+(const Experiment &a, const Experiment &b);
}

#endif
