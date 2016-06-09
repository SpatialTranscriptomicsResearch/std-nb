#include "mix.hpp"
#include "odds.hpp"

using namespace std;

namespace PoissonFactorization {
namespace Mix {

std::ostream &operator<<(std::ostream &os, Kind kind) {
  switch (kind) {
    case Kind::Dirichlet:
      os << "Dirichlet";
      break;
    case Kind::Gamma:
      os << "Gamma";
      break;
  }
  return os;
}

std::istream &operator>>(std::istream &is, Kind &kind) {
  string token;
  is >> token;
  if (token == "dirichlet")
    kind = Kind::Dirichlet;
  else if (token == "gamma")
    kind = Kind::Gamma;
  else
    throw(std::runtime_error("Cannot parse mixing distribution type '" + token
                             + "'."));
  return is;
}

template <>
void Weights<Kind::Gamma>::initialize_factor(size_t t) {
  // randomly initialize p of Θ
  LOG(debug) << "Initializing P of Θ";
  if (true)  // TODO make this CLI-switchable
    prior.p[t] = prob_to_neg_odds(
        sample_beta<Float>(parameters.hyperparameters.theta_p_1,
                           parameters.hyperparameters.theta_p_2));
  else
    prior.p[t] = 1;

  // initialize r of Θ
  LOG(debug) << "Initializing R of Θ";
  // NOTE: std::gamma_distribution takes a shape and scale parameter
  prior.r[t] = std::gamma_distribution<Float>(
      parameters.hyperparameters.theta_r_1,
      1 / parameters.hyperparameters.theta_r_2)(EntropySource::rng);

  // initialize Θ
  LOG(debug) << "Initializing Θ";
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t s = 0; s < S; ++s)
    // NOTE: std::gamma_distribution takes a shape and scale parameter
    theta(s, t) = std::gamma_distribution<Float>(
        prior.r(t), 1 / prior.p(t))(EntropySource::rng);
}

template <>
void Weights<Kind::Dirichlet>::initialize_factor(size_t t) {
  // TODO
  std::vector<double> a(S);
  for (size_t s = 0; s < S; ++s)
    a[s] = prior.alpha[s];
  auto x
      = sample_dirichlet<Float>(a, EntropySource::rngs[omp_get_thread_num()]);
  // for (size_t s = 0; s < S; ++s)
  //   phi(g, t) = x[g];
}

template <>
void Weights<Kind::Gamma>::initialize() {
  // initialize Θ
  LOG(debug) << "Initializing Θ from Gamma distribution";
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t s = 0; s < S; ++s) {
    const size_t thread_num = omp_get_thread_num();
    for (size_t t = 0; t < T; ++t)
      // NOTE: std::gamma_distribution takes a shape and scale parameter
      theta(s, t) = std::gamma_distribution<Float>(
          prior.r(t), 1 / prior.p(t))(EntropySource::rngs[thread_num]);
  }
}

template <>
void Weights<Kind::Dirichlet>::initialize() {
  LOG(debug) << "Initializing Θ from Dirichlet distribution" << std::endl;
  std::vector<double> a(T);
  for (size_t t = 0; t < T; ++t)
    a[t] = prior.alpha[t];
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t s = 0; s < S; ++s) {
    const size_t thread_num = omp_get_thread_num();
    auto x = sample_dirichlet<Float>(a, EntropySource::rngs[thread_num]);
    for (size_t t = 0; t < T; ++t)
      theta(s, t) = x[t];
  }
}

template <>
// TODO ensure no NaNs or infinities are generated
double Weights<Kind::Gamma>::log_likelihood_factor(const IMatrix &counts,
                                                   size_t t) const {
  double l = 0;

#pragma omp parallel for reduction(+ : l) if (DO_PARALLEL)
  for (size_t s = 0; s < S; ++s) {
    // NOTE: log_gamma takes a shape and scale parameter
    auto cur = log_gamma(theta(s, t), prior.r(t), 1.0 / prior.p(t));
    if (false and cur > 0)
      LOG(debug) << "ll_cur > 0 for (s,t) = (" + std::to_string(s) + ", " + std::to_string(t) + "): " + std::to_string(cur)
        + " theta = " + std::to_string(theta(s,t))
        + " r = " + std::to_string(prior.r(t))
        + " p = " + std::to_string(prior.p(t))
        + " (r - 1) * log(theta) = " + std::to_string((prior.r(t)- 1) * log(theta(s,t)))
        + " - theta / 1/p = " + std::to_string(- theta(s,t) / 1/prior.p(t))
        + " - lgamma(r) = " + std::to_string(- lgamma(prior.r(t)))
        + " - r * log(1/p) = " + std::to_string(- prior.r(t) * log(1/prior.p(t)));
    l += cur;
  }

  // NOTE: log_gamma takes a shape and scale parameter
  l += log_gamma(prior.r(t), parameters.hyperparameters.theta_r_1,
                 1.0 / parameters.hyperparameters.theta_r_2);

  l += log_beta_neg_odds(prior.p(t), parameters.hyperparameters.theta_p_1,
                         parameters.hyperparameters.theta_p_2);

  return l;
}

template <>
// TODO ensure no NaNs or infinities are generated
double Weights<Kind::Dirichlet>::log_likelihood_factor(const IMatrix &counts,
                                                       size_t t) const {
  // TODO
  assert(false);
  std::vector<Float> p(S);
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t s = 0; s < S; ++s)
    p[s] = theta(s, t);

  return log_dirichlet(p, prior.alpha);
}
}
}
