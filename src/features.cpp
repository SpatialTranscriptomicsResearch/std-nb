#include "features.hpp"
#include "aux.hpp"
#include "odds.hpp"

using namespace std;

namespace PoissonFactorization {
namespace Feature {

string to_string(Kind kind) {
  switch (kind) {
    case Kind::Dirichlet:
      return "Dirichlet";
      break;
    case Kind::Gamma:
      return "Gamma";
      break;
  }
}

ostream &operator<<(ostream &os, Kind kind) {
  os << to_string(kind);
  return os;
}

istream &operator>>(istream &is, Kind &kind) {
  string token;
  is >> token;
  token = to_lower(token);
  if (token == "dirichlet")
    kind = Kind::Dirichlet;
  else if (token == "gamma")
    kind = Kind::Gamma;
  else
    throw(runtime_error("Cannot parse mixing distribution type '" + token
                             + "'."));
  return is;
}

template <>
void Features<Kind::Gamma>::initialize_factor(size_t t) {
  // initialize p of Φ
  LOG(debug) << "Initializing P of Φ";
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t g = 0; g < G; ++g)
    prior.p(g, t) = prob_to_neg_odds(sample_beta<Float>(
        parameters.hyperparameters.phi_p_1, parameters.hyperparameters.phi_p_2,
        EntropySource::rngs[omp_get_thread_num()]));

  // initialize r of Φ
  LOG(debug) << "Initializing R of Φ";
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t g = 0; g < G; ++g)
    // NOTE: std::gamma_distribution takes a shape and scale parameter
    prior.r(g, t) = std::gamma_distribution<Float>(
        parameters.hyperparameters.phi_r_1,
        1 / parameters.hyperparameters.phi_r_2)(
        EntropySource::rngs[omp_get_thread_num()]);

  // initialize Φ
  LOG(debug) << "Initializing Φ";
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t g = 0; g < G; ++g)
    // NOTE: std::gamma_distribution takes a shape and scale parameter
    phi(g, t)
        = std::gamma_distribution<Float>(prior.r(g, t), 1 / prior.p(g, t))(
            EntropySource::rngs[omp_get_thread_num()]);
}

template <>
void Features<Kind::Dirichlet>::initialize_factor(size_t t) {
  std::vector<double> a(G);
  for (size_t g = 0; g < G; ++g)
    a[g] = prior.alpha[g];
  auto x
      = sample_dirichlet<Float>(a, EntropySource::rngs[omp_get_thread_num()]);
  for (size_t g = 0; g < G; ++g)
    phi(g, t) = x[g];
}

template <>
void Features<Kind::Gamma>::initialize() {
  LOG(debug) << "Initializing Φ from Gamma distribution";
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t g = 0; g < G; ++g) {
    const size_t thread_num = omp_get_thread_num();
    for (size_t t = 0; t < T; ++t)
      // NOTE: gamma_distribution takes a shape and scale parameter
      phi(g, t) = std::gamma_distribution<Float>(
          prior.r(g, t), 1 / prior.p(g, t))(EntropySource::rngs[thread_num]);
  }
}

template <>
void Features<Kind::Dirichlet>::initialize() {
  LOG(debug) << "Initializing Φ from Dirichlet distribution";
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t t = 0; t < T; ++t)
    initialize_factor(t);
}

template <>
// TODO ensure no NaNs or infinities are generated
double Features<Kind::Gamma>::log_likelihood_factor(const IMatrix &counts,
    size_t t) const {
  double l = 0;

#pragma omp parallel for reduction(+ : l) if (DO_PARALLEL)
  for (size_t g = 0; g < G; ++g)
    // NOTE: log_gamma takes a shape and scale parameter
    l += log_gamma(phi(g, t), prior.r(g, t), 1.0 / prior.p(g, t));

#pragma omp parallel for reduction(+ : l) if (DO_PARALLEL)
  for (size_t g = 0; g < G; ++g)
    // NOTE: log_gamma takes a shape and scale parameter
    l += log_gamma(prior.r(g, t), parameters.hyperparameters.phi_r_1,
                   1.0 / parameters.hyperparameters.phi_r_2);

#pragma omp parallel for reduction(+ : l) if (DO_PARALLEL)
  for (size_t g = 0; g < G; ++g)
    l += log_beta_neg_odds(prior.p(g, t), parameters.hyperparameters.phi_p_1,
                           parameters.hyperparameters.phi_p_2);

  return l;
}

template <>
// TODO ensure no NaNs or infinities are generated
double Features<Kind::Dirichlet>::log_likelihood_factor(const IMatrix &counts,
                                                        size_t t) const {
  std::vector<Float> p(G);
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t g = 0; g < G; ++g)
    p[g] = phi(g, t);

  return log_dirichlet(p, prior.alpha);
}

/** sample phi */
template <>
void Features<Kind::Gamma>::sample(const Matrix &theta,
                                   const IMatrix &contributions_gene_type,
                                   const Vector &spot_scaling,
                                   const Vector &experiment_scaling_long) {
  LOG(info) << "Sampling Φ from Gamma distribution";

  // pre-computation
  Vector theta_t(T, arma::fill::zeros);
  for (size_t s = 0; s < S; ++s) {
    Float prod = spot_scaling[s];
    if (parameters.activate_experiment_scaling)
      prod *= experiment_scaling_long[s];
    for (size_t t = 0; t < T; ++t)
      theta_t[t] += theta(s, t) * prod;
  }

// main step: sampling
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t g = 0; g < G; ++g) {
    const size_t thread_num = omp_get_thread_num();
    for (size_t t = 0; t < T; ++t)
      // NOTE: gamma_distribution takes a shape and scale parameter
      phi(g, t) = std::gamma_distribution<Float>(
          prior.r(g, t) + contributions_gene_type(g, t),
          1.0 / (prior.p(g, t) + theta_t[t]))(EntropySource::rngs[thread_num]);
  }

  // enforce means if necessary
  if ((parameters.enforce_mean & ForceMean::Phi) != ForceMean::None)
    for (size_t t = 0; t < T; ++t) {
      double z = 0;
#pragma omp parallel for reduction(+ : z) if (DO_PARALLEL)
      for (size_t g = 0; g < G; ++g)
        z += phi(g, t);
#pragma omp parallel for if (DO_PARALLEL)
      for (size_t g = 0; g < G; ++g)
        phi(g, t) = phi(g, t) / z * phi_scaling;
    }
}

template <>
void Features<Kind::Dirichlet>::sample(const Matrix &theta,
                                       const IMatrix &contributions_gene_type,
                                       const Vector &spot_scaling,
                                       const Vector &experiment_scaling_long) {
  LOG(info) << "Sampling Φ from Dirichlet distribution";
  for (size_t t = 0; t < T; ++t) {
    std::vector<Float> a(G, parameters.hyperparameters.alpha);
#pragma omp parallel for if (DO_PARALLEL)
    for (size_t g = 0; g < G; ++g)
      a[g] += contributions_gene_type(g, t);
    auto phi_k = sample_dirichlet<Float>(a);
    for (size_t g = 0; g < G; ++g)
      phi(g, t) = phi_k[g];
  }
}
}
}
