#include <omp.h>
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/trigamma.hpp>
#include "PoissonFactorAnalysis.hpp"
#include "montecarlo.hpp"

#define DO_PARALLEL 1

using namespace std;
using PFA = PoissonFactorAnalysis;

PFA::Float digamma(PFA::Float x) { return boost::math::digamma(x); }

PFA::Float trigamma(PFA::Float x) { return boost::math::trigamma(x); }

template <typename T>
T square(T x) {
  return x * x;
}

ostream &operator<<(ostream &os, const PoissonFactorAnalysis &pfa) {
  os << "Poisson Factor Analysis "
     << "S = " << pfa.S << " "
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
    for (size_t s = 0; s < min<size_t>(pfa.S, 10); ++s) {
      for (size_t t = 0; t < pfa.T; ++t)
        os << (t > 0 ? "\t" : "") << pfa.theta[s][t];
      os << endl;
    }

    os << "Theta factor sums" << endl;
    for (size_t t = 0; t < pfa.T; ++t) {
      double sum = 0;
      for (size_t s = 0; s < pfa.S; ++s) sum += pfa.theta[s][t];
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
      // NOTE: gamma_distribution takes a shape and scale parameter
      r[t] = gamma_distribution<Float>(priors.c0 * priors.r0,
                                       1.0 / priors.c0)(EntropySource::rng);

    // randomly initialize Theta
    for (size_t s = 0; s < S; ++s)
      for (size_t t = 0; t < T; ++t) {
      // NOTE: gamma_distribution takes a shape and scale parameter
        theta[s][t] = gamma_distribution<Float>(
            r[t], p[t] / (1 - p[t]))(EntropySource::rng);
      }

    // randomly initialize Phi
    for (size_t t = 0; t < T; ++t) {
      auto phi_ = sample_dirichlet<Float>(vector<Float>(G, priors.alpha));
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
    // Theta = zeros(T,S)+1/T;
    for (size_t s = 0; s < S; ++s)
      for (size_t t = 0; t < T; ++t) theta[s][t] = 1.0 / T;

    // randomly initialize P
    // p_k=ones(T,1)*0.5;
    for (size_t t = 0; t < T; ++t) p[t] = 0.5 * T;

    // initialize R
    // r_k= 50/T*ones(T,1)
    for (size_t t = 0; t < T; ++t) r[t] = 50.0 / T;
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

double PFA::log_likelihood(const IMatrix &counts) const {
  double l = 0;
  vector<double> alpha(G, priors.alpha);
  for (size_t t = 0; t < T; ++t) {
    vector<double> phik(G, 0);
    for (size_t g = 0; g < G; ++g) phik[g] = phi[g][t];
    l += log_dirichlet(phik, alpha);
    // NOTE: log_gamma takes a shape and scale parameter
    l += log_gamma(r[t], priors.c0 * priors.r0, 1.0 / priors.c0);
    l += log_beta(p[t], priors.c * priors.epsilon,
                  priors.c * (1 - priors.epsilon));
    for (size_t s = 0; s < S; ++s)
      // NOTE: log_gamma takes a shape and scale parameter
      l += log_gamma(theta[s][t], r[t], p[t] / (1 - p[t]));
    for (size_t g = 0; g < G; ++g)
      for (size_t s = 0; s < S; ++s)
        l += log_poisson(contributions[g][s][t], phi[g][t] * theta[s][t]);
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
void PFA::sample_contributions(const IMatrix &counts) {
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

/** sample phi */
void PFA::sample_phi() {
  if (verbosity >= Verbosity::Verbose) cout << "Sampling Phi" << endl;
  for (size_t t = 0; t < T; ++t) {
    vector<double> a(G, priors.alpha);
#pragma omp parallel for if (DO_PARALLEL)
    for (size_t g = 0; g < G; ++g)
      for (size_t s = 0; s < S; ++s) a[g] += contributions[g][s][t];
    auto phi_k = sample_dirichlet<Float>(a);
    for (size_t g = 0; g < G; ++g) phi[g][t] = phi_k[g];
  }
}

/** sample p */
void PFA::sample_p() {
  if (verbosity >= Verbosity::Verbose) cout << "Sampling P" << endl;
  for (size_t t = 0; t < T; ++t) {
    Int sum = 0;
#pragma omp parallel for reduction(+ : sum) if (DO_PARALLEL)
    for (size_t g = 0; g < G; ++g)
      for (size_t s = 0; s < S; ++s) sum += contributions[g][s][t];
    p[t] = sample_beta<Float>(priors.c * priors.epsilon + sum,
                              priors.c * (1 - priors.epsilon) + S * r[t]);
  }
}

/** sample r */
void PFA::sample_r() {
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
      const Float beta = 1 / priors.c0;
      const Float rt = r[t];
      const Float pt = p[t];
      const Float rt2 = square(rt);
      const Float S2 = square(S);
      // for(auto &x: count_spot_type)
      //   x += rt;
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
          rt2 * S2 * square(digamma_r - log_1_minus_p) +
          S * (2 * rt2 * (log_1_minus_p - digamma_r) * digamma_sum -
               rt2 * trigamma_r +
               2 * (beta * rt2 + (1 - alpha) * rt) * digamma_r -
               2 * beta * log_1_minus_p * rt2 +
               2 * (alpha - 1) * log_1_minus_p * rt) +
          rt2 * trigamma_term + rt2 * square(digamma_sum) +
          (rt2 * digamma_r - (log_1_minus_p + 2 * beta) * rt2 +
           2 * (alpha - 1) * rt) *
              digamma_sum +
          square(beta) * rt2 + 2 * (1 - alpha) * beta * rt + square(alpha) -
          3 * alpha + 2;
      // const Float x = -numerator / denominator;
      const Float r_prime = rt - (-numerator / denominator);

      if (verbosity >= Verbosity::Debug) cout << "R' = " << r_prime << endl;

      /** compute conditional posterior of r (or rather: a value proportional to it) */
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
}

/** sample theta */
void PFA::sample_theta() {
  if (verbosity >= Verbosity::Verbose) cout << "Sampling Theta" << endl;
  for (size_t t = 0; t < T; ++t)
    for (size_t s = 0; s < S; ++s) {
      Int sum = 0;
#pragma omp parallel for reduction(+ : sum) if (DO_PARALLEL)
      for (size_t g = 0; g < G; ++g) sum += contributions[g][s][t];
      // NOTE: gamma_distribution takes a shape and scale parameter
      theta[s][t] =
          gamma_distribution<Float>(r[t] + sum, p[t])(EntropySource::rng);
    }
}

void PFA::gibbs_sample(const IMatrix &counts) {
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
