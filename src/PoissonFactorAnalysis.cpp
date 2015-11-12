#include "PoissonFactorAnalysis.hpp"

using namespace std;
using PFA = PoissonFactorAnalysis;

std::ostream &operator<<(std::ostream &os, const PoissonFactorAnalysis &pfa) {
  os << "Poisson Factor Analysis "
     << "N = " << pfa.N << " "
     << "G = " << pfa.G << " "
     << "K = " << pfa.K;
  return os;
}

PFA::PoissonFactorAnalysis(const IMatrix &counts, const size_t K_,
                           const Priors &priors_, Verbosity verbosity_)
    : G(counts.shape()[0]),
      N(counts.shape()[1]),
      K(K_),
      priors(priors_),
      contributions(boost::extents[G][N][K]),
      phi(boost::extents[G][K]),
      theta(boost::extents[N][K]),
      r(boost::extents[K]),
      p(boost::extents[K]),
      verbosity(verbosity_) {
  // randomly initialize P
  for (size_t k = 0; k < K; ++k)
    p[k] = sample_beta<Float>(priors.c * priors.epsilon,
                              priors.c * (1 - priors.epsilon));

  // randomly initialize P
  for (size_t k = 0; k < K; ++k)
    r[k] = std::gamma_distribution<Float>(priors.c0 * priors.r0,
                                          1.0 / priors.c0)(EntropySource::rng);
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

double PFA::log_likelihood(const IMatrix &counts) const {
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

/** sample count decomposition */
void PFA::sample_contributions(const IMatrix &counts) {
  if (verbosity >= Verbosity::Verbose)
    std::cout << "Sampling contributions" << std::endl;
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

/** sample phi */
void PFA::sample_phi() {
  if (verbosity >= Verbosity::Verbose) std::cout << "Sampling Phi" << std::endl;
  for (size_t k = 0; k < K; ++k) {
    std::vector<double> a(G, priors.alpha);
    for (size_t g = 0; g < G; ++g)
      for (size_t n = 0; n < N; ++n) a[g] += contributions[g][n][k];
    auto phi_k = sample_dirichlet<Float>(a);
    for (size_t g = 0; g < G; ++g) phi[g][k] = phi_k[k];
  }
}

/** sample p */
void PFA::sample_p() {
  if (verbosity >= Verbosity::Verbose) std::cout << "Sampling P" << std::endl;
  for (size_t k = 0; k < K; ++k) {
    Int sum = 0;
    for (size_t g = 0; g < G; ++g)
      for (size_t n = 0; n < N; ++n) sum += contributions[g][n][k];
    p[k] = sample_beta<Float>(priors.c * priors.epsilon + sum,
                              priors.c * (1 - priors.epsilon) + N * r[k]);
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
  for (size_t k = 0; k < K; ++k)
    for (size_t n = 0; n < N; ++n) {
      Int sum = 0;
      for (size_t g = 0; g < G; ++g) sum += contributions[g][n][k];
      theta[n][k] =
          std::gamma_distribution<Float>(r[k] + sum, p[k])(EntropySource::rng);
    }
}
