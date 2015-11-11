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

  PoissonFactorAnalysis(const IMatrix &counts, const size_t K_,
                        const Priors &priors_)
      : G(counts.shape()[0]),
        N(counts.shape()[1]),
        K(K_),
        priors(priors_),
        contributions(boost::extents[G][N][K]),
        phi(boost::extents[G][K]),
        theta(boost::extents[N][K]),
        r(boost::extents[K]),
        p(boost::extents[K]) {
    // randomly initialize P
    for (size_t k = 0; k < K; ++k)
      p[k] = sample_beta<Float>(priors.c * priors.epsilon,
                                priors.c * (1 - priors.epsilon));

    // randomly initialize P
    for (size_t k = 0; k < K; ++k)
      r[k] = std::gamma_distribution<Float>(
          priors.c0 * priors.r0, 1.0 / priors.c0)(EntropySource::rng);
    // randomly initialize Theta
    for (size_t n = 0; n < N; ++n)
      for (size_t k = 0; k < K; ++k) {
        theta[n][k] = std::gamma_distribution<Float>(
            r[k], p[k] / (1 - p[k]))(EntropySource::rng);
      }

    // randomly initialize Phi
    for (size_t k = 0; k < K; ++k) {
      auto phi_ = sample_dirichlet<Float>(std::vector<Float>(G, priors.alpha));
      for (size_t g = 0; g < G; ++g) phi[g][k] = phi_[k];
    }

    // randomly initialize the contributions
    for (size_t g = 0; g < G; ++g)
      for (size_t n = 0; n < N; ++n) {
        std::vector<double> p(K);
        double z = 0;
        for (size_t k = 0; k < K; ++k) z += p[k] = phi[g][k] * theta[n][k];
        for (size_t k = 0; k < K; ++k) p[k] /= z;
        auto v = sample_multinomial<Int>(counts[g][n], p);
        for (size_t k = 0; k < K; ++k) contributions[g][n][k] = v[k];
      }
  }

  double log_likelihood(const IMatrix &counts) const {
    double l = 0;
    std::vector<double> alpha(G, priors.alpha);
    for (size_t k = 0; k < K; ++k) {
      std::vector<double> phik(G, 0);
      for (size_t g = 0; g < G; ++g) phik[g] = phi[g][k];
      l += K * log_dirichlet(alpha, phik);
      l += log_gamma(r[k], priors.c0 * priors.r0, 1.0 / priors.c0);
      l += log_beta(p[k], priors.c * priors.epsilon,
                    priors.c * (1 - priors.epsilon));
      for (size_t n = 0; n < N; ++n)
        l += log_gamma(theta[n][k], r[k], p[k] / (1 - p[k]));
      for (size_t g = 0; g < G; ++g)
        for (size_t n = 0; n < N; ++n)
          l += log_poisson(contributions[g][n][k], phi[g][k] * theta[n][k]);
    }
    /*
    for (size_t g = 0; g < G; ++g)
      for (size_t n = 0; n < N; ++n) {
        double rate = 0;
        for (size_t k = 0; k < K; ++k) rate += phi[g][k] * theta[n][k];
        l += log_poisson(counts[g][n], rate);
      }
      */
    return l;
  }
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
  Verbosity verbosity;

 public:
  Generator<PFA>(const PFA::IMatrix &counts_, Verbosity verbosity_)
      : counts(counts_), verbosity(verbosity_){};

  PFA generate(const PFA &current) const {
    PFA next(current);

    std::uniform_int_distribution<size_t> r_unif(0, 4);
    size_t i = r_unif(EntropySource::rng);
    while (i == 3 or i == 2) i = r_unif(EntropySource::rng);
    switch (i) {
      case 0:
        // sample count decomposition
        if (verbosity >= Verbosity::Verbose)
          std::cout << "Sampling contributions" << std::endl;
        for (size_t g = 0; g < current.G; ++g)
          for (size_t n = 0; n < current.N; ++n) {
            std::vector<double> p(current.K);
            double z = 0;
            for (size_t k = 0; k < current.K; ++k)
              z += p[k] = next.phi[g][k] * next.theta[n][k];
            for (size_t k = 0; k < current.K; ++k) p[k] /= z;
            auto v = sample_multinomial<PFA::Int>(counts[g][n], p);
            for (size_t k = 0; k < current.K; ++k)
              next.contributions[g][n][k] = v[k];
          }
        break;
      case 1:
        // sample phi
        if (verbosity >= Verbosity::Verbose)
          std::cout << "Sampling Phi" << std::endl;
        for (size_t k = 0; k < current.K; ++k) {
          std::vector<double> a(current.G, current.priors.alpha);
          for (size_t g = 0; g < current.G; ++g)
            for (size_t n = 0; n < current.N; ++n)
              a[g] += current.contributions[g][n][k];
          // if (verbosity >= Verbosity::Verbose)
          //   std::cout << "Sampling Phi k = " << k << std::endl;
          auto phi = sample_dirichlet<PFA::Float>(a);
          for (size_t g = 0; g < current.G; ++g) next.phi[g][k] = phi[k];
        }
        break;
      case 2:
        // sample p
        if (verbosity >= Verbosity::Verbose)
          std::cout << "Sampling P" << std::endl;
        for (size_t k = 0; k < current.K; ++k) {
          PFA::Int sum = 0;
          for (size_t g = 0; g < current.G; ++g)
            for (size_t n = 0; n < current.N; ++n)
              sum += current.contributions[g][n][k];
          next.p[k] = sample_beta<PFA::Float>(
              current.priors.c * current.priors.epsilon + sum,
              current.priors.c * (1 - current.priors.epsilon) +
                  current.N * current.r[k]);
        }
        break;
      case 3:
        // sample r
        if (verbosity >= Verbosity::Verbose)
          std::cout << "Sampling R" << std::endl;
        // TODO
        break;
      case 4:
        // sample theta
        if (verbosity >= Verbosity::Verbose)
          std::cout << "Sampling Theta" << std::endl;
        for (size_t k = 0; k < current.K; ++k)
          for (size_t n = 0; n < current.N; ++n) {
            PFA::Int sum = 0;
            for (size_t g = 0; g < current.G; ++g)
              sum += current.contributions[g][n][k];
            next.theta[n][k] = std::gamma_distribution<PFA::Float>(
                current.r[k] + sum, current.p[k])(EntropySource::rng);
          }
        break;
    }
    return next;
  };
};
}

std::ostream &operator<<(std::ostream &os, const PoissonFactorAnalysis &pfa);

#endif
