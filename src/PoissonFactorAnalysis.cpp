#include <omp.h>
#include "PoissonFactorAnalysis.hpp"

#define DO_PARALLEL 1

using namespace std;
using PFA = PoissonFactorAnalysis;

std::ostream &operator<<(std::ostream &os, const PoissonFactorAnalysis &pfa) {
  os << "Poisson Factor Analysis "
     << "N = " << pfa.N << " "
     << "G = " << pfa.G << " "
     << "T = " << pfa.T << endl;

  if (pfa.verbosity >= Verbosity::Verbose) {
    os << "Phi" << endl;
    for (size_t g = 0; g < min<size_t>(pfa.G, 10); ++g) {
      for (size_t t = 0; t < pfa.T; ++t)
        os << (t > 0 ? "\t" : "") << pfa.phi[g][t];
      os << endl;
    }

    os << "Phi factor sums" << endl;
    for (size_t t = 0; t < pfa.T; ++t) {
      double sum = 0;
      for (size_t g = 0; g < pfa.G; ++g) sum += pfa.phi[g][t];
      os << (t > 0 ? "\t" : "") << sum;
    }
    os << endl;

    os << "Theta" << endl;
    for (size_t n = 0; n < min<size_t>(pfa.N, 10); ++n) {
      for (size_t t = 0; t < pfa.T; ++t)
        os << (t > 0 ? "\t" : "") << pfa.theta[n][t];
      os << endl;
    }

    os << "Theta factor sums" << endl;
    for (size_t t = 0; t < pfa.T; ++t) {
      double sum = 0;
      for (size_t n = 0; n < pfa.N; ++n) sum += pfa.theta[n][t];
      os << (t > 0 ? "\t" : "") << sum;
    }
    os << endl;

    os << "P" << endl;
    for (size_t t = 0; t < pfa.T; ++t) os << (t > 0 ? "\t" : "") << pfa.p[t];
    os << endl;

    os << "R" << endl;
    for (size_t t = 0; t < pfa.T; ++t) os << (t > 0 ? "\t" : "") << pfa.r[t];
    os << endl;
  }

  return os;
}

PFA::PoissonFactorAnalysis(const IMatrix &counts, const size_t T_,
                           const Priors &priors_, Verbosity verbosity_)
    : G(counts.shape()[0]),
      N(counts.shape()[1]),
      T(T_),
      priors(priors_),
      contributions(boost::extents[G][N][T]),
      phi(boost::extents[G][T]),
      theta(boost::extents[N][T]),
      r(boost::extents[T]),
      p(boost::extents[T]),
      verbosity(verbosity_) {
  if (false) {
    // randomly initialize P
    for (size_t t = 0; t < T; ++t)
      p[t] = sample_beta<Float>(priors.c * priors.epsilon,
                                priors.c * (1 - priors.epsilon));

    // randomly initialize R
    for (size_t t = 0; t < T; ++t)
      r[t] = std::gamma_distribution<Float>(
          priors.c0 * priors.r0, 1.0 / priors.c0)(EntropySource::rng);

    // randomly initialize Theta
    for (size_t n = 0; n < N; ++n)
      for (size_t t = 0; t < T; ++t) {
        theta[n][t] = std::gamma_distribution<Float>(
            r[t], p[t] / (1 - p[t]))(EntropySource::rng);
      }

    // randomly initialize Phi
    for (size_t t = 0; t < T; ++t) {
      auto phi_ = sample_dirichlet<Float>(std::vector<Float>(G, priors.alpha));
      for (size_t g = 0; g < G; ++g) phi[g][t] = phi_[t];
    }
  } else {
    // randomly initialize Phi
    // Phi = rand(P,T);
    // Phi = bsxfun(@rdivide,Phi,sum(Phi,1));
    for (size_t t = 0; t < T; ++t) {
      double sum = 0;
      for (size_t g = 0; g < G; ++g)
        sum += phi[g][t] = RandomDistribution::Uniform(EntropySource::rng);
      for (size_t g = 0; g < G; ++g) phi[g][t] /= sum;
    }

    // initialize Theta
    // Theta = zeros(T,N)+1/T;
    for (size_t n = 0; n < N; ++n)
      for (size_t t = 0; t < T; ++t) theta[n][t] = 1.0 / T;

    // randomly initialize P
    // p_k=ones(T,1)*0.5;
    for (size_t t = 0; t < T; ++t) p[t] = 0.5 * T;

    // initialize R
    // r_k= 50/T*ones(T,1)
    for (size_t t = 0; t < T; ++t) r[t] = 50.0 / T;
  }

  // randomly initialize the contributions
  for (size_t g = 0; g < G; ++g)
    for (size_t n = 0; n < N; ++n) {
      std::vector<double> prob(T);
      double z = 0;
      for (size_t t = 0; t < T; ++t) z += prob[t] = phi[g][t] * theta[n][t];
      for (size_t t = 0; t < T; ++t) prob[t] /= z;
      auto v = sample_multinomial<Int>(counts[g][n], prob);
      for (size_t t = 0; t < T; ++t) contributions[g][n][t] = v[t];
    }
}

double PFA::log_likelihood(const IMatrix &counts) const {
  double l = 0;
  std::vector<double> alpha(G, priors.alpha);
  for (size_t t = 0; t < T; ++t) {
    std::vector<double> phik(G, 0);
    for (size_t g = 0; g < G; ++g) phik[g] = phi[g][t];
    l += log_dirichlet(alpha, phik);
    l += log_gamma(r[t], priors.c0 * priors.r0, 1.0 / priors.c0);
    l += log_beta(p[t], priors.c * priors.epsilon,
                  priors.c * (1 - priors.epsilon));
    for (size_t n = 0; n < N; ++n)
      l += log_gamma(theta[n][t], r[t], p[t] / (1 - p[t]));
    for (size_t g = 0; g < G; ++g)
      for (size_t n = 0; n < N; ++n)
        l += log_poisson(contributions[g][n][t], phi[g][t] * theta[n][t]);
  }
  /*
  for (size_t g = 0; g < G; ++g)
    for (size_t n = 0; n < N; ++n) {
      double rate = 0;
      for (size_t t = 0; t < T; ++t) rate += phi[g][t] * theta[n][t];
      l += log_poisson(counts[g][n], rate);
    }
    */
  return l;
}

/** sample count decomposition */
void PFA::sample_contributions(const IMatrix &counts) {
  if (verbosity >= Verbosity::Verbose)
    std::cout << "Sampling contributions" << std::endl;
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t g = 0; g < G; ++g)
    for (size_t n = 0; n < N; ++n) {
      std::vector<double> rel_rate(T);
      double z = 0;
      for (size_t t = 0; t < T; ++t) z += rel_rate[t] = phi[g][t] * theta[n][t];
      for (size_t t = 0; t < T; ++t) rel_rate[t] /= z;
      auto v = sample_multinomial<Int>(counts[g][n], rel_rate);
      for (size_t t = 0; t < T; ++t) contributions[g][n][t] = v[t];
    }
}

/** sample phi */
void PFA::sample_phi() {
  if (verbosity >= Verbosity::Verbose) std::cout << "Sampling Phi" << std::endl;
  for (size_t t = 0; t < T; ++t) {
    std::vector<double> a(G, priors.alpha);
#pragma omp parallel for if (DO_PARALLEL)
    for (size_t g = 0; g < G; ++g)
      for (size_t n = 0; n < N; ++n) a[g] += contributions[g][n][t];
    auto phi_k = sample_dirichlet<Float>(a);
    for (size_t g = 0; g < G; ++g) phi[g][t] = phi_k[g];
  }
}

/** sample p */
void PFA::sample_p() {
  if (verbosity >= Verbosity::Verbose) std::cout << "Sampling P" << std::endl;
  for (size_t t = 0; t < T; ++t) {
    Int sum = 0;
#pragma omp parallel for reduction (+ : sum) if (DO_PARALLEL)
    for (size_t g = 0; g < G; ++g)
      for (size_t n = 0; n < N; ++n) sum += contributions[g][n][t];
    p[t] = sample_beta<Float>(priors.c * priors.epsilon + sum,
                              priors.c * (1 - priors.epsilon) + N * r[t]);
  }
}

/** sample r */
void PFA::sample_r() {
  if (verbosity >= Verbosity::Verbose) std::cout << "Sampling R" << std::endl;
  // TODO
}

/** sample theta */
void PFA::sample_theta() {
  if (verbosity >= Verbosity::Verbose)
    std::cout << "Sampling Theta" << std::endl;
  for (size_t t = 0; t < T; ++t)
// #pragma omp parallel for if (DO_PARALLEL)
    for (size_t n = 0; n < N; ++n) {
      Int sum = 0;
#pragma omp parallel for reduction (+ : sum) if (DO_PARALLEL)
      for (size_t g = 0; g < G; ++g) sum += contributions[g][n][t];
      theta[n][t] =
          std::gamma_distribution<Float>(r[t] + sum, p[t])(EntropySource::rng);
    }
}
