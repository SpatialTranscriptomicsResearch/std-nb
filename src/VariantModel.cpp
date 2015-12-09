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
// NOTE this is likely correct
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
void VariantModel::sample_r() {
  // TODO implement
  /*
  if (verbosity >= Verbosity::Verbose) cout << "Sampling R" << endl;
  for (size_t t = 0; t < T; ++t) {
    vector<Int> count_spot_type(S, 0);
    Int sum = 0;
    for (size_t s = 0; s < S; ++s) {
      Int sum_spot = 0;
#pragma omp parallel for reduction(+ : sum_spot) if (DO_PARALLEL)
      for (size_t g = 0; g < G; ++g) sum_spot += contributions[g][s][t];
      sum += sum_spot;
      count_spot_type[s] = sum_spot;
    }
    if (sum == 0) {
      // NOTE: gamma_distribution takes a shape and scale parameter
      r[t] = gamma_distribution<Float>(
          priors.c0 * priors.r0,
          1 / (priors.c0 - S * log(1 - p[t])))(EntropySource::rng);
    } else {
      if (verbosity >= Verbosity::Debug) cout << "Sum counts = " << sum << endl;
      // TODO: check sampling of R when sum != 0
      const Float alpha = priors.c0 * priors.r0;
      // TODO: determine which of the following two is the right one to use
      // const Float beta = 1 / priors.c0;
      // NOTE: likely it is the latter definition here that is correct
      const Float beta = priors.c0;
      const Float rt = r[t];
      const Float pt = p[t];
      const Float rt2 = square(rt);
      const Float log_1_minus_p = log(1 - pt);
      const Float digamma_r = digamma(rt);
      const Float trigamma_r = trigamma(rt);

      Float digamma_sum = 0;
      for (auto &x : count_spot_type) digamma_sum += digamma(x + rt);

      Float trigamma_term = 0;
      for (auto &x : count_spot_type)
        trigamma_term +=
            trigamma(x + rt) + (log_1_minus_p - digamma_r) * digamma(rt + x);

      const Float numerator =
          rt2 * (S * (digamma_r - log_1_minus_p) - digamma_sum + beta) +
          (1 - alpha) * rt;
      const Float denominator =
          rt2 * (trigamma_r * S - trigamma_term +
                 (log_1_minus_p - digamma_r) * digamma_sum) +
          alpha - 1;

      const Float ratio = numerator / denominator;

      Float r_prime = rt - ratio;

      if (verbosity >= Verbosity::Debug)
        cout << "numerator = " << numerator << " denominator = " << denominator
             << " ratio = " << ratio << " R' = " << r_prime << endl;
             */

      /** compute conditional posterior of r (or rather: a value proportional to
       * it) */
    /*
      auto compute_cond_posterior = [&](Float x) {
        double log_posterior =
            log_gamma(x, priors.c0 * priors.r0, 1 / priors.c0);
        for (auto &y : count_spot_type)
          log_posterior += log_negative_binomial(y, x, pt);
        return log_posterior;
      };

      // NOTE: log_gamma takes a shape and scale parameter
      double log_posterior_current = compute_cond_posterior(rt);

      if (verbosity >= Verbosity::Debug) {
        // NOTE: log_gamma takes a shape and scale parameter
        double log_posterior_prime = compute_cond_posterior(r_prime);

        cout << "R = " << rt << " R' = " << r_prime << endl
             << "f(R) = " << log_posterior_current
             << " f(R') = " << log_posterior_prime << endl;
      }

      if (r_prime < 0) {
        // TODO improve this hack! e.g. by using an exp-transform
        if (verbosity >= Verbosity::Debug)
          cout << "Warning R' < 0! Setting to " << rt / 2 << endl;
        r_prime = rt / 2;
      }

      while (true) {
        const Float r_new = normal_distribution<Float>(
            r_prime,
            parameters.adj_step_size * sqrt(r_prime))(EntropySource::rng);

        // NOTE: log_gamma takes a shape and scale parameter
        double log_posterior_new = compute_cond_posterior(r_new);

        if (log_posterior_new > log_posterior_current) {
          r[t] = r_new;
          if (verbosity >= Verbosity::Debug)
            cout << "T = " << parameters.temperature << " current = " << rt
                 << " next = " << r_new << endl
                 << "nextG = " << log_posterior_new
                 << " G = " << log_posterior_current
                 << " dG = " << (log_posterior_new - log_posterior_current)
                 << endl << "Improved!" << endl;
          break;
        } else {
          const Float dG = log_posterior_new - log_posterior_current;
          double rnd = RandomDistribution::Uniform(EntropySource::rng);
          double prob =
              min<double>(1.0, MCMC::boltzdist(-dG, parameters.temperature));
          if (verbosity >= Verbosity::Debug)
            cout << "T = " << parameters.temperature << " current = " << rt
                 << " next = " << r_new << endl
                 << "nextG = " << log_posterior_new
                 << " G = " << log_posterior_current << " dG = " << dG
                 << " prob = " << prob << " rnd = " << rnd << endl;
          if (std::isnan(log_posterior_new) == 0 and (dG > 0 or rnd <= prob)) {
            if (verbosity >= Verbosity::Debug) cout << "Accepted!" << endl;
            r[t] = r_new;
            break;
          } else {
            if (verbosity >= Verbosity::Debug) cout << "Rejected!" << endl;
          }
        }

        if (r_prime < 0) exit(EXIT_FAILURE);
      }
    }
  }
  */
}

/** sample phi */
void VariantModel::sample_phi() {
  if (verbosity >= Verbosity::Verbose) cout << "Sampling Φ" << endl;
  for (size_t t = 0; t < T; ++t)
    for (size_t g = 0; g < G; ++g) {
      Int sum = 0;
#pragma omp parallel for reduction(+ : sum) if (DO_PARALLEL)
      for (size_t s = 0; s < S; ++s) sum += contributions[g][s][t];
      // NOTE: gamma_distribution takes a shape and scale parameter
      phi[g][t] =
          gamma_distribution<Float>(r[g][t] + sum, p[g][t])(EntropySource::rng);
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
