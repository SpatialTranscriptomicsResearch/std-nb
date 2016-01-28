#include <omp.h>
#include "VariantModel.hpp"
#include "metropolis_hastings.hpp"
#include "pdist.hpp"
#include "stats.hpp"
#include "timer.hpp"

#define DO_PARALLEL 1
#define PHI_ZERO_WARNING false

using namespace std;
namespace FactorAnalysis {
const Float spot_scaling_prior_a = 1;
const Float spot_scaling_prior_b = 1;
const Float experiment_scaling_prior_a = 1;
const Float experiment_scaling_prior_b = 1;

template <typename T>
T odds_to_prob(T x) {
  return x / (x + 1);
}

template <typename T>
T neg_odds_to_prob(T x) {
  return 1 / (x + 1);
}

template <typename T>
T prob_to_odds(T x) {
  return x / (1 - x);
}

template <typename T>
T prob_to_neg_odds(T x) {
  return (1 - x) / x;
}

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
      spot_scaling(boost::extents[S]),
      experiment_scaling(boost::extents[S]),
      r(boost::extents[G][T]),
      p(boost::extents[G][T]),
      verbosity(verbosity_) {
  if (false) {
    /*
    // randomly initialize P
    for (size_t g = 0; g < G; ++g)
      for (size_t t = 0; t < T; ++t)
        p[g][t] = sample_beta<Float>(priors.e * priors.f,
                                     priors.e * (1 - priors.f));

    // randomly initialize R
    for (size_t g = 0; g < G; ++g)
      for (size_t t = 0; t < T; ++t)
        // NOTE: gamma_distribution takes a shape and scale parameter
        r[g][t] = gamma_distribution<Float>(
            priors.c * priors.d, 1.0 / priors.c)(EntropySource::rng);

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

    // initialize spot scaling factors
    for (size_t s = 0; s < S; ++s)
      // NOTE: gamma_distribution takes a shape and scale parameter
      spot_scaling[s] = gamma_distribution<Float>(
          spot_scaling_prior_a, 1 / spot_scaling_prior_b)(EntropySource::rng);

    // initialize experiment scaling factors
    for (size_t s = 0; s < S; ++s)
      experiment_scaling[s] = 1;

    // randomly initialize P
    // p_k=ones(T,1)*0.5;
    for (size_t g = 0; g < G; ++g)
      for (size_t t = 0; t < T; ++t)
        if (false)
          p[g][t] = 0.5 * G * T;
        else
          // NOTE: gamma_distribution takes a shape and scale parameter
          p[g][t] = gamma_distribution<Float>(
              priors.e, 1 / priors.f)(EntropySource::rng);

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
      l += log_gamma(r[g][t], priors.c, 1.0 / priors.d);
    if (std::isnan(l))
      cout << "Likelihood is NAN after adding the contribution due to "
              "Gamma-distributed r[g][" << t << "]." << endl;
#pragma omp parallel for reduction(+ : l) if (DO_PARALLEL)
    for (size_t g = 0; g < G; ++g)
      // NOTE: log_gamma takes a shape and scale parameter
      l += log_gamma(p[g][t], priors.e, 1 / priors.f);
    if (std::isnan(l))
      cout << "Likelihood is NAN after adding the contribution due to "
              "Beta-distributed p[g][" << t << "]." << endl;
#pragma omp parallel for reduction(+ : l) if (DO_PARALLEL)
    for (size_t g = 0; g < G; ++g) {
      // NOTE: log_gamma takes a shape and scale parameter
      l += log_gamma(phi[g][t], r[g][t], p[g][t]);
      if (std::isnan(l))
        cout << "Likelihood is NAN after adding the contribution due to "
                "Gamma-distributed phi[" << g << "][" << t
             << "]; phi=" << phi[g][t] << " r=" << r[g][t] << " p=" << p[g][t]
             << endl;
    }
#pragma omp parallel for reduction(+ : l) if (DO_PARALLEL)
    for (size_t g = 0; g < G; ++g)
      for (size_t s = 0; s < S; ++s)
        l += log_poisson(contributions[g][s][t], phi[g][t] * theta[s][t]);
    if (std::isnan(l))
      cout << "Likelihood is NAN after adding the contribution due to "
              "Poisson-distributed contributions[g][s][" << t << "]." << endl;
  }

  for (size_t g = 0; g < G; ++g) {
    vector<double> thetak(G, 0);
    for (size_t t = 0; t < T; ++t) thetak[g] = theta[g][t];
    l += log_dirichlet(thetak, alpha);
  }
  if (std::isnan(l))
    cout << "Likelihood is NAN after adding the contribution due to "
            "Dirichlet-distributed theta." << endl;

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
  for (size_t g = 0; g < G; ++g) {
    size_t thread_num = omp_get_thread_num();
    for (size_t s = 0; s < S; ++s) {
      vector<double> rel_rate(T);
      double z = 0;
      // NOTE: in principle, lambda has a factor of q[s].
      // However, this would cancel. Thus, we do not multiply it in here.
      for (size_t t = 0; t < T; ++t) z += rel_rate[t] = phi[g][t] * theta[s][t];
      for (size_t t = 0; t < T; ++t) rel_rate[t] /= z;
      auto v = sample_multinomial<Int>(counts[g][s], rel_rate,
                                       EntropySource::rngs[thread_num]);
      for (size_t t = 0; t < T; ++t) contributions[g][s][t] = v[t];
    }
  }
}

/** sample theta */
// NOTE this is proven to be correct
void VariantModel::sample_theta() {
  if (verbosity >= Verbosity::Verbose) cout << "Sampling Θ" << endl;
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t s = 0; s < S; ++s) {
    size_t thread_num = omp_get_thread_num();
    vector<double> a(T, priors.alpha);
    for (size_t g = 0; g < G; ++g)
      for (size_t t = 0; t < T; ++t) a[t] += contributions[g][s][t];
    auto theta_ = sample_dirichlet<Float>(a, EntropySource::rngs[thread_num]);
    for (size_t t = 0; t < T; ++t) theta[s][t] = theta_[t];
  }
}

double compute_conditional(const pair<Float, Float> &x, size_t g, size_t t,
                           Int count_sum, Float weight_sum,
                           const Priors &priors) {
  const Float current_r = x.first;
  const Float current_p = x.second;
  // NOTE: gamma_distribution takes a shape and scale parameter
  return log_gamma(current_p, priors.e, 1 / priors.f) +
         log_gamma(current_r, priors.c, 1 / priors.d) +
         // The next line is part of the negative binomial distribution.
         // The other factors aren't needed as they don't depend on either of
         // r[g][t] and p[g][t], and thus would cancel when computing the score
         // ratio.
         +current_r * log(current_p) -
         (current_r + count_sum) * log(current_p + weight_sum) +
         lgamma(current_r + count_sum) - lgamma(current_r);
}

/** sample p and r */
/* This is a simple Metropolis-Hastings sampling scheme */
void VariantModel::sample_p_and_r() {
  if (verbosity >= Verbosity::Verbose) cout << "Sampling P and R" << endl;

  auto gen = [&](const pair<Float, Float> &x, mt19937 &rng) {
    normal_distribution<double> rnorm;
    const double f1 = exp(rnorm(rng));
    const double f2 = exp(rnorm(rng));
    return pair<Float, Float>(f1 * x.first, f2 * x.second);
  };

  for (size_t t = 0; t < T; ++t) {
    Float weight_sum = 0;
    for (size_t s = 0; s < S; ++s)
      weight_sum += theta[s][t] * spot_scaling[s] * experiment_scaling[s];
    MetropolisHastings mh(parameters.temperature, parameters.prop_sd,
                          verbosity);

#pragma omp parallel for if (DO_PARALLEL)
    for (size_t g = 0; g < G; ++g) {
      Int count_sum = 0;
      for (size_t s = 0; s < S; ++s) count_sum += contributions[g][s][t];
      size_t thread_num = omp_get_thread_num();
      auto res =
          mh.sample(pair<Float, Float>(r[g][t], p[g][t]), parameters.n_iter,
                    EntropySource::rngs[thread_num], gen, compute_conditional,
                    g, t, count_sum, weight_sum, priors);
      r[g][t] = res.first;
      p[g][t] = res.second;
    }
  }
}

/** sample phi */
// NOTE this is proven to be correct
void VariantModel::sample_phi() {
  if (verbosity >= Verbosity::Verbose) cout << "Sampling Φ" << endl;
  Vector theta_t(boost::extents[T]);
  for (size_t t = 0; t < T; ++t)
    for (size_t s = 0; s < S; ++s)
      theta_t[t] += theta[s][t] * spot_scaling[s] * experiment_scaling[s];
  for (size_t t = 0; t < T; ++t)
#pragma omp parallel for if (DO_PARALLEL)
    for (size_t g = 0; g < G; ++g) {
      size_t thread_num = omp_get_thread_num();
      Int summed_contribution = 0;
      for (size_t s = 0; s < S; ++s)
        summed_contribution += contributions[g][s][t];
      // NOTE: gamma_distribution takes a shape and scale parameter
      phi[g][t] = gamma_distribution<Float>(
          r[g][t] + summed_contribution,
          1.0 / (p[g][t] + theta_t[t]))(EntropySource::rngs[thread_num]);
      if (PHI_ZERO_WARNING and phi[g][t] == 0) {
        cout << "Warning: phi[" << g << "][" << t << "] = 0!" << endl << "r["
             << g << "][" << t << "] = " << r[g][t] << endl << "p[" << g << "]["
             << t << "] = " << p[g][t] << endl << "theta_t[" << t
             << "] = " << theta_t[t] << endl
             << "r[g][t] + sum = " << r[g][t] + summed_contribution << endl
             << "1.0 / (p[g][t] + theta_t[t]) = "
             << 1.0 / (p[g][t] + theta_t[t]) << endl
             << "sum = " << summed_contribution << endl;
        if (verbosity >= Verbosity::Debug) {
          Int sum2 = 0;
          for (size_t tt = 0; tt < T; ++tt)
            for (size_t s = 0; s < S; ++s) sum2 += contributions[g][s][tt];
          cout << "sum2 = " << sum2 << endl;
        }
        // exit(EXIT_FAILURE);
      }
    }
}

/** sample spot scaling factors */
void VariantModel::sample_spot_scaling() {
  if (verbosity >= Verbosity::Verbose)
    cout << "Sampling spot scaling factors" << endl;
  for (size_t s = 0; s < S; ++s) {
    Int summed_contribution = 0;
#pragma omp parallel for reduction(+ : summed_contribution) if (DO_PARALLEL)
    for (size_t g = 0; g < G; ++g)
      for (size_t t = 0; t < T; ++t)
        summed_contribution += contributions[g][s][t];

    Float intensity_sum = 0;
    for (size_t t = 0; t < T; ++t) {
      Float x = 0;
#pragma omp parallel for reduction(+ : x) if (DO_PARALLEL)
      for (size_t g = 0; g < G; ++g)
        x += phi[g][t];
      intensity_sum += x * theta[s][t];
    }
    intensity_sum *= experiment_scaling[s];

    if (verbosity >= Verbosity::Debug)
      cout << "summed_contribution=" << summed_contribution
           << " intensity_sum=" << intensity_sum
           << " prev spot_scaling[" << s << "]=" << spot_scaling[s];

    // NOTE: gamma_distribution takes a shape and scale parameter
    spot_scaling[s] = gamma_distribution<Float>(
        spot_scaling_prior_a + summed_contribution,
        1.0 / (spot_scaling_prior_b + intensity_sum))(EntropySource::rng);
    if (verbosity >= Verbosity::Debug)
      cout << "new spot_scaling[" << s << "]=" << spot_scaling[s] << endl;
  }
}

/** sample experiment scaling factors */
void VariantModel::sample_experiment_scaling() {
  if (verbosity >= Verbosity::Verbose)
    cout << "Sampling experiment scaling factors" << endl;
  for (size_t s = 0; s < S; ++s) {
    Int summed_contribution = 0;
#pragma omp parallel for reduction(+ : summed_contribution) if (DO_PARALLEL)
    for (size_t g = 0; g < G; ++g)
      for (size_t t = 0; t < T; ++t)
        summed_contribution += contributions[g][s][t];

    Float intensity_sum = 0;
    for (size_t t = 0; t < T; ++t) {
      Float x = 0;
#pragma omp parallel for reduction(+ : x) if (DO_PARALLEL)
      for (size_t g = 0; g < G; ++g)
        x += phi[g][t];
      intensity_sum += x * theta[s][t];
    }
    intensity_sum *= spot_scaling[s];

    if (verbosity >= Verbosity::Debug)
      cout << "summed_contribution=" << summed_contribution
           << " intensity_sum=" << intensity_sum << " prev experiment_scaling["
           << s << "]=" << spot_scaling[s];

    // NOTE: gamma_distribution takes a shape and scale parameter
    experiment_scaling[s] = gamma_distribution<Float>(
        experiment_scaling_prior_a + summed_contribution,
        1.0 / (experiment_scaling_prior_b + intensity_sum))(EntropySource::rng);
    if (verbosity >= Verbosity::Debug)
      cout << "new experiment_scaling[" << s << "]=" << spot_scaling[s] << endl;
  }
}

void VariantModel::gibbs_sample(const IMatrix &counts, bool timing) {
  check_model(counts);

  Timer timer;
  sample_contributions(counts);
  if (timing and verbosity >= Verbosity::Info)
    cout << "This took " << timer.tock() << "μs." << endl;
  if (verbosity >= Verbosity::Everything)
    cout << "Log-likelihood = " << log_likelihood(counts) << endl;
  check_model(counts);

  timer.tick();
  sample_phi();
  if (timing and verbosity >= Verbosity::Info)
    cout << "This took " << timer.tock() << "μs." << endl;
  if (verbosity >= Verbosity::Everything)
    cout << "Log-likelihood = " << log_likelihood(counts) << endl;
  check_model(counts);

  timer.tick();
  sample_p_and_r();
  if (timing and verbosity >= Verbosity::Info)
    cout << "This took " << timer.tock() << "μs." << endl;
  if (verbosity >= Verbosity::Everything)
    cout << "Log-likelihood = " << log_likelihood(counts) << endl;
  check_model(counts);

  timer.tick();
  sample_theta();
  if (timing and verbosity >= Verbosity::Info)
    cout << "This took " << timer.tock() << "μs." << endl;
  if (verbosity >= Verbosity::Everything)
    cout << "Log-likelihood = " << log_likelihood(counts) << endl;
  check_model(counts);

  timer.tick();
  sample_spot_scaling();
  if (timing and verbosity >= Verbosity::Info)
    cout << "This took " << timer.tock() << "μs." << endl;
  if (verbosity >= Verbosity::Everything)
    cout << "Log-likelihood = " << log_likelihood(counts) << endl;
  check_model(counts);
}

vector<Int> VariantModel::sample_reads(size_t g, size_t s, size_t n) const {
  vector<Float> prods(T);
  for (size_t t = 0; t < T; ++t) prods[t] = theta[s][t] * spot_scaling[s];

  vector<Int> v(n, 0);
  for (size_t i = 0; i < n; ++i)
    for (size_t t = 0; t < T; ++t)
      v[i] += sample_negative_binomial(
          r[g][t], prods[t] / (prods[t] + (1 - p[g][t]) / p[g][t]),
          EntropySource::rng);
  return v;
}

void VariantModel::check_model(const IMatrix &counts) const {
  return;
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
      /*
      if (phi[g][t] == 0)
        throw(runtime_error("Phi is zero for gene " + to_string(g) +
                            " in factor " + to_string(t) + "."));
                            */
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
  for (size_t g = 0; g < G; ++g)
    for (size_t t = 0; t < T; ++t) {
      if (p[g][t] < 0)
        throw(runtime_error("P[" + to_string(g) + "][" + to_string(t) +
                            "] is smaller zero: p=" + to_string(p[g][t]) +
                            "."));
      if (p[g][t] == 0)
        throw(runtime_error("P is zero for gene " + to_string(g) +
                            " in factor " + to_string(t) + "."));

      if (r[g][t] < 0)
        throw(runtime_error("R[" + to_string(g) + "][" + to_string(t) +
                            "] is smaller zero: r=" + to_string(r[g][t]) +
                            "."));
      if (r[g][t] == 0)
        throw(runtime_error("R is zero for gene " + to_string(g) +
                            " in factor " + to_string(t) + "."));
    }

  // check priors
  if (priors.c == 0) throw(runtime_error("The prior c is zero."));
  if (priors.d == 0) throw(runtime_error("The prior d is zero."));
  if (priors.e == 0) throw(runtime_error("The prior c is zero."));
  if (priors.f == 0) throw(runtime_error("The prior f is zero."));
  if (priors.alpha == 0) throw(runtime_error("The prior alpha is zero."));
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

    size_t phi_zeros = 0;
    os << "Φ factor sums" << endl;
    for (size_t t = 0; t < pfa.T; ++t) {
      double sum = 0;
      for (size_t g = 0; g < pfa.G; ++g) {
        if (pfa.phi[g][t] == 0) phi_zeros++;
        sum += pfa.phi[g][t];
      }
      os << (t > 0 ? "\t" : "") << sum;
    }
    os << endl;
    os << "There are " << phi_zeros << " zeros in Φ. This corresponds to "
       << (100.0 * phi_zeros / pfa.T / pfa.G) << "%." << endl;

    os << "Θ" << endl;
    for (size_t s = 0; s < min<size_t>(pfa.S, 10); ++s) {
      for (size_t t = 0; t < pfa.T; ++t)
        os << (t > 0 ? "\t" : "") << pfa.theta[s][t];
      os << endl;
    }

    size_t theta_zeros = 0;
    os << "Θ factor sums" << endl;
    for (size_t t = 0; t < pfa.T; ++t) {
      double sum = 0;
      for (size_t s = 0; s < pfa.S; ++s) {
        if (pfa.theta[s][t] == 0) theta_zeros++;
        sum += pfa.theta[s][t];
      }
      os << (t > 0 ? "\t" : "") << sum;
    }
    os << endl;
    os << "There are " << theta_zeros << " zeros in Θ." << endl;

    os << "R" << endl;
    for (size_t g = 0; g < min<size_t>(pfa.G, 10); ++g) {
      for (size_t t = 0; t < pfa.T; ++t)
        os << (t > 0 ? "\t" : "") << pfa.r[g][t];
      os << endl;
    }

    size_t r_zeros = 0;
    for (size_t g = 0; g < pfa.G; ++g)
      for (size_t t = 0; t < pfa.T; ++t)
        if (pfa.r[g][t] == 0) r_zeros++;
    os << "There are " << r_zeros << " zeros in r. This corresponds to "
       << (100.0 * r_zeros / pfa.G / pfa.T) << "%." << endl;

    os << "P" << endl;
    for (size_t g = 0; g < min<size_t>(pfa.G, 10); ++g) {
      for (size_t t = 0; t < pfa.T; ++t)
        os << (t > 0 ? "\t" : "") << pfa.p[g][t];
      os << endl;
    }

    size_t p_zeros = 0;
    for (size_t g = 0; g < pfa.G; ++g)
      for (size_t t = 0; t < pfa.T; ++t)
        if (pfa.p[g][t] == 0) p_zeros++;
    os << "There are " << p_zeros << " zeros in p. This corresponds to "
       << (100.0 * p_zeros / pfa.G / pfa.T) << "%." << endl;

    os << "Scaling factors" << endl;
    for (size_t s = 0; s < pfa.S; ++s)
      os << (s > 0 ? "\t" : "") << pfa.spot_scaling[s];
    os << endl;
    size_t spot_scaling_zeros = 0;
    for (size_t s = 0; s < pfa.S; ++s)
      if (pfa.spot_scaling[s] == 0) spot_scaling_zeros++;
    os << "There are " << spot_scaling_zeros << " zeros in spot_scaling." << endl;
    os << Stats::summary(pfa.spot_scaling) << endl;
  }

  return os;
}
