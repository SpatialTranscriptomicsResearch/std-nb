#include "PartialModel.hpp"
#include "aux.hpp"
#include "odds.hpp"

using namespace std;

namespace PoissonFactorization {
namespace Partial {

template <>
void Model<Variable::Feature, Kind::Gamma>::initialize_factor(size_t t) {
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
    matrix(g, t)
        = std::gamma_distribution<Float>(prior.r(g, t), 1 / prior.p(g, t))(
            EntropySource::rngs[omp_get_thread_num()]);
}

template <>
void Model<Variable::Feature, Kind::Dirichlet>::initialize_factor(size_t t) {
  auto x = sample_dirichlet<Float>(prior.alpha.begin_col(t),
                                   prior.alpha.end_col(t),
                                   EntropySource::rngs[omp_get_thread_num()]);
  for (size_t g = 0; g < G; ++g)
    matrix(g, t) = x[g];
}

template <>
void Model<Variable::Feature, Kind::Gamma>::initialize() {
  matrix = Matrix(G, T);
  LOG(debug) << "Initializing Φ from Gamma distribution";
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t g = 0; g < G; ++g) {
    const size_t thread_num = omp_get_thread_num();
    for (size_t t = 0; t < T; ++t)
      // NOTE: gamma_distribution takes a shape and scale parameter
      matrix(g, t) = std::gamma_distribution<Float>(
          prior.r(g, t), 1 / prior.p(g, t))(EntropySource::rngs[thread_num]);
  }
}

template <>
void Model<Variable::Feature, Kind::Dirichlet>::initialize() {
  matrix = Matrix(G, T);
  LOG(debug) << "Initializing Φ from Dirichlet distribution";
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t t = 0; t < T; ++t)
    initialize_factor(t);
}

template <>
// TODO ensure no NaNs or infinities are generated
double Model<Variable::Feature, Kind::Gamma>::log_likelihood(
    const IMatrix &counts) const {
  double l = 0;
  for (size_t t = 0; t < T; ++t)
    l += log_likelihood_factor(counts, t);
  return l;
}
template <>
// TODO ensure no NaNs or infinities are generated
double Model<Variable::Feature, Kind::Gamma>::log_likelihood_factor(
    const IMatrix &counts, size_t t) const {
  double l = 0;

#pragma omp parallel for reduction(+ : l) if (DO_PARALLEL)
  for (size_t g = 0; g < G; ++g)
    // NOTE: log_gamma takes a shape and scale parameter
    l += log_gamma(matrix(g, t), prior.r(g, t), 1.0 / prior.p(g, t));

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
// TODO check whether using OMP is actually faster here!
double Model<Variable::Feature, Kind::Dirichlet>::log_likelihood(
    const IMatrix &counts) const {
  double l = 0;
  for(size_t t = 0; t < T; ++t)
    l += log_likelihood_factor(counts, t);
  return l;
}

template <>
// TODO ensure no NaNs or infinities are generated
// TODO check whether using OMP is actually faster here!
double Model<Variable::Feature, Kind::Dirichlet>::log_likelihood_factor(
    const IMatrix &counts, size_t t) const {
  std::vector<Float> p(G);
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t g = 0; g < G; ++g)
    p[g] = matrix(g, t);

  std::vector<Float> alpha(G);
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t g = 0; g < G; ++g)
    alpha[g] = prior.alpha(g,t) + prior.alpha_prior;

  return log_dirichlet(p, alpha);
}
}
}
