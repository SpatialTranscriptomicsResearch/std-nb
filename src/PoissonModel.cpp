#include <omp.h>
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/trigamma.hpp>
#include "PoissonModel.hpp"
#include "montecarlo.hpp"
#include "pdist.hpp"
#include "timer.hpp"

#define DO_PARALLEL 1

using namespace std;
namespace FactorAnalysis {
PoissonModel::PoissonModel(const IMatrix &counts, const size_t T_,
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
      p[t] = sample_beta<Float>(priors.phi_p_1 * priors.phi_p_2,
                                priors.phi_p_1 * (1 - priors.phi_p_2));

    // randomly initialize R
    for (size_t t = 0; t < T; ++t)
      // NOTE: gamma_distribution takes a shape and scale parameter
      r[t] = gamma_distribution<Float>(priors.phi_r_1 * priors.phi_r_2,
                                       1.0 / priors.phi_r_1)(EntropySource::rng);

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

double PoissonModel::log_likelihood(const IMatrix &counts) const {
  double l = 0;
  vector<double> alpha(G, priors.alpha);
  for (size_t t = 0; t < T; ++t) {
    vector<double> phik(G, 0);
    for (size_t g = 0; g < G; ++g) phik[g] = phi[g][t];
    l += log_dirichlet(phik, alpha);
    // NOTE: log_gamma takes a shape and scale parameter
    l += log_gamma(r[t], priors.phi_r_1 * priors.phi_r_2, 1.0 / priors.phi_r_1);
    l += log_beta(p[t], priors.phi_p_1 * priors.phi_p_2,
                  priors.phi_p_1 * (1 - priors.phi_p_2));
#pragma omp parallel for reduction(+ : l) if (DO_PARALLEL)
    for (size_t s = 0; s < S; ++s)
      // NOTE: log_gamma takes a shape and scale parameter
      l += log_gamma(theta[s][t], r[t], p[t] / (1 - p[t]));
#pragma omp parallel for reduction(+ : l) if (DO_PARALLEL)
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
void PoissonModel::sample_contributions(const IMatrix &counts) {
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
void PoissonModel::sample_phi() {
  if (verbosity >= Verbosity::Verbose) cout << "Sampling Φ" << endl;
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
void PoissonModel::sample_p() {
  if (verbosity >= Verbosity::Verbose) cout << "Sampling P" << endl;
  for (size_t t = 0; t < T; ++t) {
    Int sum = 0;
#pragma omp parallel for reduction(+ : sum) if (DO_PARALLEL)
    for (size_t g = 0; g < G; ++g)
      for (size_t s = 0; s < S; ++s) sum += contributions[g][s][t];
    p[t] = sample_beta<Float>(priors.phi_p_1 * priors.phi_p_2 + sum,
                              priors.phi_p_1 * (1 - priors.phi_p_2) + S * r[t]);
  }
}

/** sample r */
void PoissonModel::sample_r() {
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
          priors.phi_r_1 * priors.phi_r_2,
          1 / (priors.phi_r_1 - S * log(1 - p[t])))(EntropySource::rng);
    } else {
      if (verbosity >= Verbosity::Debug) cout << "Sum counts = " << sum << endl;
      // TODO: check sampling of R when sum != 0
      const Float alpha = priors.phi_r_1 * priors.phi_r_2;
      // TODO: determine which of the following two is the right one to use
      // const Float beta = 1 / priors.phi_r_1;
      // NOTE: likely it is the latter definition here that is correct
      const Float beta = priors.phi_r_1;
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

      /** compute conditional posterior of r (or rather: a value proportional to
       * it) */
      auto compute_cond_posterior = [&](Float x) {
        // NOTE: log_gamma takes a shape and scale parameter
        double log_posterior =
            log_gamma(x, priors.phi_r_1 * priors.phi_r_2, 1 / priors.phi_r_1);
        for (auto &y : count_spot_type)
          log_posterior += log_negative_binomial(y, x, pt);
        return log_posterior;
      };

      double log_posterior_current = compute_cond_posterior(rt);

      if (verbosity >= Verbosity::Debug) {
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
            sqrt(r_prime))(EntropySource::rng);

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
void PoissonModel::sample_theta() {
  if (verbosity >= Verbosity::Verbose) cout << "Sampling Θ" << endl;
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

void PoissonModel::gibbs_sample(const IMatrix &counts, bool timing) {
  Timer timer;
  sample_contributions(counts);
  if(timing and verbosity >= Verbosity::Info)
    cout << "This took " << timer.tock() << "μs." << endl;
  if (verbosity >= Verbosity::Everything)
    cout << "Log-likelihood = " << log_likelihood(counts) << endl;
  timer.tick();
  sample_phi();
  if(timing and verbosity >= Verbosity::Info)
    cout << "This took " << timer.tock() << "μs." << endl;
  if (verbosity >= Verbosity::Everything)
    cout << "Log-likelihood = " << log_likelihood(counts) << endl;
  timer.tick();
  sample_p();
  if(timing and verbosity >= Verbosity::Info)
    cout << "This took " << timer.tock() << "μs." << endl;
  if (verbosity >= Verbosity::Everything)
    cout << "Log-likelihood = " << log_likelihood(counts) << endl;
  timer.tick();
  sample_r();
  if(timing and verbosity >= Verbosity::Info)
    cout << "This took " << timer.tock() << "μs." << endl;
  if (verbosity >= Verbosity::Everything)
    cout << "Log-likelihood = " << log_likelihood(counts) << endl;
  timer.tick();
  sample_theta();
  if(timing and verbosity >= Verbosity::Info)
    cout << "This took " << timer.tock() << "μs." << endl;
  if (verbosity >= Verbosity::Everything)
    cout << "Log-likelihood = " << log_likelihood(counts) << endl;
}

void PoissonModel::check_model(const IMatrix &counts) const {
  // check that the contributions add up to the observations
  for (size_t g = 0; g < G; ++g)
    for (size_t s = 0; s < S; ++s) {
      Int z = 0;
      for (size_t t = 0; t < T; ++t) z += contributions[g][s][t];
      if (z != counts[g][s])
        throw(runtime_error(
            "Contributions do not add up to observations for gene " +
            to_string(g) + " in spot " + to_string(s) + "."));
    }

  // check that phi is positive
  for (size_t g = 0; g < G; ++g)
    for (size_t t = 0; t < T; ++t) {
      if (phi[g][t] == 0)
        throw(runtime_error("Phi is zero for gene " + to_string(g) +
                            " in factor " + to_string(t) + "."));
      if (phi[g][t] < 0)
        throw(runtime_error("Phi is negative for gene " + to_string(g) +
                            " in factor " + to_string(t) + "."));
    }

  // check that theta is positive
  for (size_t s = 0; s < S; ++s)
    for (size_t t = 0; t < T; ++t) {
      if (theta[s][t] == 0)
        throw(runtime_error("Theta is zero for spot " + to_string(s) +
                            " in factor " + to_string(t) + "."));
      if (theta[s][t] < 0)
        throw(runtime_error("Theta is negative for spot " + to_string(s) +
                            " in factor " + to_string(t) + "."));
    }

  // check that r and p are positive, and that p is < 1
  for (size_t t = 0; t < T; ++t) {
    if (p[t] <= 0 or p[t] >= 1)
      throw(runtime_error("P[" + to_string(t) + "] is smaller zero or larger 1: p=" +
                          to_string(p[t]) + "."));
    if ((1 - p[t]) / p[t] == 0)
      throw(runtime_error("(1-P)/P is zero in factor " + to_string(t) + "."));

    if (r[t] < 0)
      throw(runtime_error("R[" + to_string(t) + "] is smaller zero: r=" + to_string(r[t]) + "."));
    if (r[t] == 0)
      throw(runtime_error("R is zero in factor " + to_string(t) + "."));
  }

  // check priors
  if (priors.phi_r_1 == 0) throw(runtime_error("The prior phi_r_1 is zero."));
  if (priors.phi_r_2 == 0) throw(runtime_error("The prior phi_r_2 is zero."));
  if (priors.phi_p_1 == 0) throw(runtime_error("The prior phi_p_1 is zero."));
  if (priors.phi_p_2 == 0) throw(runtime_error("The prior phi_p_2 is zero."));
  if (priors.phi_p_2 == 1) throw(runtime_error("The prior phi_p_2 is unit."));
  if (priors.alpha == 0) throw(runtime_error("The prior alpha is zero."));
}
}

ostream &operator<<(ostream &os, const FactorAnalysis::PoissonModel &pfa) {
  os << "Poisson Factor Analysis "
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
    for (size_t t = 0; t < pfa.T; ++t) os << (t > 0 ? "\t" : "") << pfa.p[t];
    os << endl;

    os << "R" << endl;
    for (size_t t = 0; t < pfa.T; ++t) os << (t > 0 ? "\t" : "") << pfa.r[t];
    os << endl;
  }

  return os;
}
