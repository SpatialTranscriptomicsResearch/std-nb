#ifndef POISSONFACTORANALYSIS_HPP
#define POISSONFACTORANALYSIS_HPP

#include <cstdint>
#include <vector>
#include <boost/multi_array.hpp>
#include "montecarlo.hpp"
#include "sampling.hpp"
#include "pdist.hpp"

struct PoissonFactorAnalysis {
  using Int = uint32_t;
  using Float = double;
  using Vector = boost::multi_array<Float, 1>;
  using CVector = boost::multi_array<Float, 1>;
  using Matrix = boost::multi_array<Float, 2>;
  using IMatrix = boost::multi_array<Int, 2>;
  using Tensor = boost::multi_array<Float, 3>;
  using ITensor = boost::multi_array<Int, 3>;

  struct Priors {
    Priors(Float c_ = 1.0, Float epsilon_ = 0.01, Float c0_ = 1.0,
           Float r0_ = 1.0, Float gamma_ = 1.0, Float alpha_ = 0.5)
        : c(c_),
          epsilon(epsilon_),
          c0(c0_),
          r0(r0_),
          gamma(gamma_),
          alpha(alpha_){};

    // priors for the beta distribution (22)
    // const Float c;
    Float c;
    // const Float epsilon;
    Float epsilon;

    // priors for the gamma distribution (21)
    // const Float c0;
    Float c0;
    // const Float r0;
    Float r0;

    // const Float gamma;
    Float gamma;
    // const Float alpha;
    Float alpha;
  };

  /** number of genes */
  // const size_t G;
  size_t G;
  /** number of samples */
  // const size_t N;
  size_t N;
  /** number of factors */
  // const size_t K;
  size_t K;

  // const Priors priors;
  Priors priors;

  // const Matrix counts;
  ITensor contributions;

  /** factor loading matrix */
  Matrix phi;

  /** factor score matrix */
  Matrix theta;

  /** shape parameter for the prior of the factor scores */
  Vector r;
  /** scale parameter for the prior of the factor scores */
  Vector p;

  Verbosity verbosity;

  PoissonFactorAnalysis(const IMatrix &counts, const size_t K,
                        const Priors &priors, Verbosity verbosity);
  double log_likelihood(const IMatrix &counts) const;

  /** sample count decomposition */
  void sample_contributions(const IMatrix &counts);

  /** sample phi */
  void sample_phi();

  /** sample p */
  void sample_p();

  /** sample r */
  void sample_r();

  /** sample theta */
  void sample_theta();
};

namespace MCMC {
template <>
class Evaluator<PoissonFactorAnalysis> {
  using PFA = PoissonFactorAnalysis;
  PFA::IMatrix counts;

 public:
  Evaluator<PFA>(const PFA::IMatrix &counts_) : counts(counts_){};

  double evaluate(PFA &pfa) const { return pfa.log_likelihood(counts); };
};

template <>
class Generator<PoissonFactorAnalysis> {
  using PFA = PoissonFactorAnalysis;
  PFA::IMatrix counts;

 public:
  Generator<PFA>(const PFA::IMatrix &counts_) : counts(counts_) {}

  PFA generate(const PFA &current) const {
    PFA next(current);

    std::uniform_int_distribution<size_t> r_unif(0, 4);
    size_t i = r_unif(EntropySource::rng);
    while (i == 3 or i == 2) i = r_unif(EntropySource::rng);
    switch (i) {
      case 0:
        next.sample_contributions(counts);
        break;
      case 1:
        // sample phi
        next.sample_phi();
        break;
      case 2:
        next.sample_p();
        break;
      case 3:
        next.sample_r();
        break;
      case 4:
        next.sample_theta();
        break;
    }
    return next;
  };
};
}

std::ostream &operator<<(std::ostream &os, const PoissonFactorAnalysis &pfa);

#endif
