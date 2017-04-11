#ifndef PRIORS_HPP
#define PRIORS_HPP

#include <cstddef>
#include "entropy.hpp"
#include "log.hpp"
#include "metropolis_hastings.hpp"
#include "odds.hpp"
#include "parallel.hpp"
#include "parameters.hpp"
#include "pdist.hpp"
#include "sampling.hpp"
#include "types.hpp"

namespace STD {

const std::string FILENAME_ENDING = ".tsv";

namespace PRIOR {

template <typename T>
std::pair<T, T> gen_log_normal_pair(const std::pair<T, T> &x,
                                    std::mt19937 &rng) {
  std::normal_distribution<double> rnorm;
  const double f1 = exp(rnorm(rng));
  const double f2 = exp(rnorm(rng));
  return {f1 * x.first, f2 * x.second};
};

namespace THETA {

struct Gamma {
  size_t dim1, dim2;
  /** shape parameter for the prior of the mixing matrix */
  Vector r;
  /** scale parameter for the prior of the mixing matrix */
  /* Stored as negative-odds */
  Vector p;
  Parameters parameters;

  Gamma(size_t dim1_, size_t dim2_, const Parameters &params);
  Gamma(const Gamma &other);
  /** sample p_phi and r_phi */
  /* This is a simple Metropolis-Hastings sampling scheme */
  void sample(const Matrix &observed, const Matrix &field);

  void store(const std::string &prefix,
             const std::vector<std::string> &spot_names,
             const std::vector<std::string> &factor_names,
             const std::vector<size_t> &order) const;
  void restore(const std::string &prefix);

  void enforce_positive_parameters(const std::string &tag);

private:
  void initialize_r();
  void initialize_p();
};

/** This routine doesn't print, for the same reason as sample() does nothing */
std::ostream &operator<<(std::ostream &os, const Gamma &x);
}
}
}

#endif
