#include <omp.h>
#include "VariantModel.hpp"
#include "montecarlo.hpp"
#include "pdist.hpp"

#define DO_PARALLEL 1

using namespace std;
namespace FactorAnalysis {
VariantModel::VariantModel(const IMatrix &counts, const size_t T_,
                           const Priors &priors_, const Parameters &parameters_,
                           Verbosity verbosity_)
    : G(counts.shape()[0]),
      S(counts.shape()[1]),
      T(T_),
      priors(priors_),
      parameters(parameters_),
      contributions(boost::extents[G][S][T]),
      phi(boost::extents[G][T]),
      theta(boost::extents[S][T]),
      r(boost::extents[G][T]),
      p(boost::extents[G][T]),
      verbosity(verbosity_) {
  if (false) {
    /*
    // randomly initialize P
    for (size_t g = 0; g < G; ++g)
      for (size_t t = 0; t < T; ++t)
        p[g][t] = sample_beta<Float>(priors.c * priors.epsilon,
                                     priors.c * (1 - priors.epsilon));

    // randomly initialize R
    for (size_t g = 0; g < G; ++g)
      for (size_t t = 0; t < T; ++t)
        // NOTE: gamma_distribution takes a shape and scale parameter
        r[g][t] = gamma_distribution<Float>(
            priors.c0 * priors.r0, 1.0 / priors.c0)(EntropySource::rng);

    // TODO adapt
    // randomly initialize Theta
    for (size_t s = 0; s < S; ++s)
      for (size_t t = 0; t < T; ++t) {
        // NOTE: gamma_distribution takes a shape and scale parameter
        theta[s][t] = gamma_distribution<Float>(
            r[t], p[t] / (1 - p[t]))(EntropySource::rng);
      }

    // TODO adapt
    // randomly initialize Phi
    for (size_t t = 0; t < T; ++t) {
      auto phi_ = sample_dirichlet<Float>(vector<Float>(G, priors.alpha));
      for (size_t g = 0; g < G; ++g) phi[g][t] = phi_[t];
    }
    */
  } else {
    // randomly initialize Phi
    // Phi = zeros(T,S)+1/T;
    for (size_t g = 0; g < G; ++g)
      for (size_t t = 0; t < T; ++t) phi[g][t] = 1.0 / T / G;

    // initialize Theta
    // Theta = rand(P,T);
    // Theta = bsxfun(@rdivide,Phi,sum(Phi,1));
    for (size_t s = 0; s < S; ++s) {
      double sum = 0;
      for (size_t t = 0; t < T; ++t)
        sum += theta[s][t] = RandomDistribution::Uniform(EntropySource::rng);
      for (size_t t = 0; t < T; ++t) theta[s][t] /= sum;
    }

    // randomly initialize P
    // p_k=ones(T,1)*0.5;
    for (size_t g = 0; g < G; ++g)
      for (size_t t = 0; t < T; ++t) p[g][t] = 0.5 * G * T;

    // initialize R
    // r_k= 50/T*ones(T,1)
    for (size_t g = 0; g < G; ++g)
      for (size_t t = 0; t < T; ++t) r[g][t] = 50.0 / G / T;
  }

  // randomly initialize the contributions
  for (size_t g = 0; g < G; ++g)
    for (size_t s = 0; s < S; ++s) {
      vector<double> prob(T);
      double z = 0;
      for (size_t t = 0; t < T; ++t) z += prob[t] = phi[g][t] * theta[s][t];
      for (size_t t = 0; t < T; ++t) prob[t] /= z;
      auto v = sample_multinomial<Int>(counts[g][s], prob);
      for (size_t t = 0; t < T; ++t) contributions[g][s][t] = v[t];
    }
}

double VariantModel::log_likelihood(const IMatrix &counts) const {
  double l = 0;
  vector<double> alpha(G, priors.alpha);
  for (size_t t = 0; t < T; ++t) {
#pragma omp parallel for reduction(+ : l) if (DO_PARALLEL)
    for (size_t g = 0; g < G; ++g)
      // NOTE: log_gamma takes a shape and scale parameter
      l += log_gamma(r[g][t], priors.c0 * priors.r0, 1.0 / priors.c0);
#pragma omp parallel for reduction(+ : l) if (DO_PARALLEL)
    for (size_t g = 0; g < G; ++g)
      l += log_beta(p[g][t], priors.c * priors.epsilon,
                    priors.c * (1 - priors.epsilon));
#pragma omp parallel for reduction(+ : l) if (DO_PARALLEL)
    for (size_t g = 0; g < G; ++g)
      // NOTE: log_gamma takes a shape and scale parameter
      l += log_gamma(phi[g][t], r[g][t], p[g][t] / (1 - p[g][t]));
#pragma omp parallel for reduction(+ : l) if (DO_PARALLEL)
    for (size_t g = 0; g < G; ++g)
      for (size_t s = 0; s < S; ++s)
        l += log_poisson(contributions[g][s][t], phi[g][t] * theta[s][t]);
  }

  for (size_t g = 0; g < G; ++g) {
    vector<double> thetak(G, 0);
    for (size_t t = 0; t < T; ++t) thetak[g] = theta[g][t];
    l += log_dirichlet(thetak, alpha);
  }

  /*
  for (size_t g = 0; g < G; ++g)
    for (size_t s = 0; s < S; ++s) {
      double rate = 0;
      for (size_t t = 0; t < T; ++t) rate += phi[g][t] * theta[s][t];
      l += log_poisson(counts[g][s], rate);
    }
    */
  return l;
}

/** sample count decomposition */
// NOTE this is proven to be correct
void VariantModel::sample_contributions(const IMatrix &counts) {
  if (verbosity >= Verbosity::Verbose) cout << "Sampling contributions" << endl;
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t g = 0; g < G; ++g)
    for (size_t s = 0; s < S; ++s) {
      vector<double> rel_rate(T);
      double z = 0;
      for (size_t t = 0; t < T; ++t) z += rel_rate[t] = phi[g][t] * theta[s][t];
      for (size_t t = 0; t < T; ++t) rel_rate[t] /= z;
      auto v = sample_multinomial<Int>(counts[g][s], rel_rate);
      for (size_t t = 0; t < T; ++t) contributions[g][s][t] = v[t];
    }
}

/** sample theta */
// NOTE this is proven to be correct
void VariantModel::sample_theta() {
  if (verbosity >= Verbosity::Verbose) cout << "Sampling Θ" << endl;
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t s = 0; s < S; ++s) {
    vector<double> a(T, priors.alpha);
    for (size_t g = 0; g < G; ++g)
      for (size_t t = 0; t < T; ++t) a[t] += contributions[g][s][t];
    auto theta_k = sample_dirichlet<Float>(a);
    for (size_t t = 0; t < T; ++t) theta[s][t] = theta_k[t];
  }
}

/** sample p */
void VariantModel::sample_p() {
  if (verbosity >= Verbosity::Verbose) cout << "Sampling P" << endl;
  for (size_t t = 0; t < T; ++t)
#pragma omp parallel for if (DO_PARALLEL)
    for (size_t g = 0; g < G; ++g) {
      Int sum = 0;
      for (size_t s = 0; s < S; ++s) sum += contributions[g][s][t];
      p[g][t] =
          sample_beta<Float>(priors.c * priors.epsilon + sum,
                             priors.c * (1 - priors.epsilon) + S * r[g][t]);
    }
}

/** sample r */
/* This is a simple Metropolis-Hastings sampling scheme */
void VariantModel::sample_r() {
  if (verbosity >= Verbosity::Verbose) cout << "Sampling R" << endl;
  auto compute_conditional = [&](Float x, size_t g, size_t t) {
    double l = log_gamma(x, priors.c0 * priors.r0, 1.0 / priors.c0);
    l += log_gamma(phi[g][t], x, p[g][t] / (1 - p[g][t]));
    return l;
  };

  normal_distribution<double> rnorm(0, 0.5);
  for (size_t t = 0; t < T; ++t) {
    for (size_t g = 0; g < G; ++g) {
      const double current_r = r[g][t];
      const double current_ll = compute_conditional(current_r, g, t);
      const size_t n_iter_initial = 100;
      size_t n_iter = n_iter_initial;
      bool accept = false;
      while (n_iter--) {
        const double f = exp(rnorm(EntropySource::rng));
        const double new_r = current_r * f;
        const double new_ll = compute_conditional(new_r, g, t);

        if (new_ll > current_ll) {
          if (verbosity >= Verbosity::Debug) cout << "Improved!" << endl;
          accept = true;
        } else {
          const Float dG = new_ll - current_ll;
          double rnd = RandomDistribution::Uniform(EntropySource::rng);
          double prob =
              min<double>(1.0, MCMC::boltzdist(-dG, parameters.temperature));
          if (std::isnan(new_ll) == 0 and (dG > 0 or rnd <= prob)) {
            accept = true;
            if (verbosity >= Verbosity::Debug) cout << "Accepted!" << endl;
          } else {
            if (verbosity >= Verbosity::Debug) cout << "Rejected!" << endl;
          }
        }
        if (accept) {
          r[g][t] = new_r;
          break;
        }
      }
      if (verbosity >= Verbosity::Debug)
        cout << "Left MCMC " << (accept ? "" : "un") << "successfully for r["
             << g << "][" << t << "] after " << (n_iter_initial - n_iter)
             << " iterations." << endl;
    }
  }
}

/** sample phi */
// NOTE this is proven to be correct
void VariantModel::sample_phi() {
  if (verbosity >= Verbosity::Verbose) cout << "Sampling Φ" << endl;
  Vector theta_t(boost::extents[T]);
  for (size_t t = 0; t < T; ++t)
    for (size_t s = 0; s < S; ++s) theta_t[t] += theta[s][t];
  for (size_t t = 0; t < T; ++t)
    for (size_t g = 0; g < G; ++g) {
      Int sum = 0;
#pragma omp parallel for reduction(+ : sum) if (DO_PARALLEL)
      for (size_t s = 0; s < S; ++s) sum += contributions[g][s][t];
      // NOTE: gamma_distribution takes a shape and scale parameter
      phi[g][t] = gamma_distribution<Float>(
          r[g][t] + sum,
          1.0 / ((1 - p[g][t]) / p[g][t] + theta_t[t]))(EntropySource::rng);
    }
}

void VariantModel::gibbs_sample(const IMatrix &counts) {
  sample_contributions(counts);
  if (verbosity >= Verbosity::Everything)
    cout << "Log-likelihood = " << log_likelihood(counts) << endl;
  sample_phi();
  if (verbosity >= Verbosity::Everything)
    cout << "Log-likelihood = " << log_likelihood(counts) << endl;
  sample_p();
  if (verbosity >= Verbosity::Everything)
    cout << "Log-likelihood = " << log_likelihood(counts) << endl;
  sample_r();
  if (verbosity >= Verbosity::Everything)
    cout << "Log-likelihood = " << log_likelihood(counts) << endl;
  sample_theta();
  if (verbosity >= Verbosity::Everything)
    cout << "Log-likelihood = " << log_likelihood(counts) << endl;
}
}

ostream &operator<<(ostream &os, const FactorAnalysis::VariantModel &pfa) {
  os << "Variant Poisson Factor Analysis "
     << "S = " << pfa.S << " "
     << "G = " << pfa.G << " "
     << "T = " << pfa.T << endl;

  if (pfa.verbosity >= Verbosity::Verbose) {
    os << "Φ" << endl;
    for (size_t g = 0; g < min<size_t>(pfa.G, 10); ++g) {
      for (size_t t = 0; t < pfa.T; ++t)
        os << (t > 0 ? "\t" : "") << pfa.phi[g][t];
      os << endl;
    }

    os << "Φ factor sums" << endl;
    for (size_t t = 0; t < pfa.T; ++t) {
      double sum = 0;
      for (size_t g = 0; g < pfa.G; ++g) sum += pfa.phi[g][t];
      os << (t > 0 ? "\t" : "") << sum;
    }
    os << endl;

    os << "Θ" << endl;
    for (size_t s = 0; s < min<size_t>(pfa.S, 10); ++s) {
      for (size_t t = 0; t < pfa.T; ++t)
        os << (t > 0 ? "\t" : "") << pfa.theta[s][t];
      os << endl;
    }

    os << "Θ factor sums" << endl;
    for (size_t t = 0; t < pfa.T; ++t) {
      double sum = 0;
      for (size_t s = 0; s < pfa.S; ++s) sum += pfa.theta[s][t];
      os << (t > 0 ? "\t" : "") << sum;
    }
    os << endl;

    os << "P" << endl;
    for (size_t g = 0; g < min<size_t>(pfa.G, 10); ++g) {
      for (size_t t = 0; t < pfa.T; ++t)
        os << (t > 0 ? "\t" : "") << pfa.p[g][t];
      os << endl;
    }

    os << "R" << endl;
    for (size_t g = 0; g < min<size_t>(pfa.G, 10); ++g) {
      for (size_t t = 0; t < pfa.T; ++t)
        os << (t > 0 ? "\t" : "") << pfa.r[g][t];
      os << endl;
    }
  }

  return os;
}
