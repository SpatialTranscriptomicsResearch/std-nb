#include <omp.h>
#include "VariantModel.hpp"
#include "montecarlo.hpp"
#include "pdist.hpp"

#define DO_PARALLEL 1
#define PHI_ZERO_WARNING true

using namespace std;
namespace FactorAnalysis {
const Float scaling_prior_a = 1;
const Float scaling_prior_b = 1;

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
      scaling(boost::extents[S]),
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

    // initialize scaling factors
    for (size_t s = 0; s < S; ++s)
      // NOTE: gamma_distribution takes a shape and scale parameter
      scaling[s] = gamma_distribution<Float>(
          scaling_prior_a, scaling_prior_b)(EntropySource::rng);

    // randomly initialize P
    // p_k=ones(T,1)*0.5;
    for (size_t g = 0; g < G; ++g)
      for (size_t t = 0; t < T; ++t)
        if (false)
          p[g][t] = 0.5 * G * T;
        else
          p[g][t] = sample_beta<Float>(1, 1);
        /*
          p[g][t] = sample_beta<Float>(priors.c * priors.epsilon,
                                       priors.c * (1 - priors.epsilon));
        */

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
    if(std::isnan(l))
      cout << "Likelihood is NAN after adding the contribution due to Gamma-distributed r[g][" << t << "]." << endl;
#pragma omp parallel for reduction(+ : l) if (DO_PARALLEL)
    for (size_t g = 0; g < G; ++g)
      l += log_beta(p[g][t], priors.c * priors.epsilon,
                    priors.c * (1 - priors.epsilon));
    if(std::isnan(l))
      cout << "Likelihood is NAN after adding the contribution due to Beta-distributed p[g][" << t << "]." << endl;
#pragma omp parallel for reduction(+ : l) if (DO_PARALLEL)
    for (size_t g = 0; g < G; ++g) {
      // NOTE: log_gamma takes a shape and scale parameter
      l += log_gamma(phi[g][t], r[g][t], p[g][t] / (1 - p[g][t]));
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
    if(std::isnan(l))
      cout << "Likelihood is NAN after adding the contribution due to Poisson-distributed contributions[g][s][" << t << "]." << endl;
  }

  for (size_t g = 0; g < G; ++g) {
    vector<double> thetak(G, 0);
    for (size_t t = 0; t < T; ++t) thetak[g] = theta[g][t];
    l += log_dirichlet(thetak, alpha);
  }
    if(std::isnan(l))
      cout << "Likelihood is NAN after adding the contribution due to Dirichlet-distributed theta." << endl;

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
    for (size_t s = 0; s < S; ++s) theta_t[t] += theta[s][t] * scaling[s];
  for (size_t t = 0; t < T; ++t)
    for (size_t g = 0; g < G; ++g) {
      Int sum = 0;
#pragma omp parallel for reduction(+ : sum) if (DO_PARALLEL)
      for (size_t s = 0; s < S; ++s) sum += contributions[g][s][t];
      // NOTE: gamma_distribution takes a shape and scale parameter
      phi[g][t] = gamma_distribution<Float>(
          r[g][t] + sum,
          1.0 / ((1 - p[g][t]) / p[g][t] + theta_t[t]))(EntropySource::rng);
      if(PHI_ZERO_WARNING and phi[g][t] == 0) {
        cout << "Warning: phi[" << g << "][" << t << "] = 0!" << endl
          << "r[" << g << "][" << t << "] = " << r[g][t] << endl
          << "p[" << g << "][" << t << "] = " << p[g][t] << endl
          << "theta_t[" << t << "] = " << theta_t[t] << endl
          << "r[g][t] + sum = " << r[g][t] + sum << endl
          << "1.0 / ((1 - p[g][t]) / p[g][t] + theta_t[t]) = " <<  1.0 / ((1 - p[g][t]) / p[g][t] + theta_t[t]) << endl
          << "sum = " << sum << endl;
        Int sum2 = 0;
        for (size_t tt = 0; tt < T; ++tt)
          for (size_t s = 0; s < S; ++s) sum2 += contributions[g][s][tt];
        cout << "sum2 = " << sum2 << endl;
        exit(EXIT_FAILURE);
      }
    }
}

/** sample scaling factors */
void VariantModel::sample_scaling() {
  for (size_t s = 0; s < S; ++s) {
    Int count_sum = 0;
#pragma omp parallel for reduction(+ : count_sum) if (DO_PARALLEL)
    for (size_t g = 0; g < G; ++g)
      for (size_t t = 0; t < T; ++t) count_sum += contributions[g][s][t];

    Float intensity_sum = 0;
    for (size_t t = 0; t < T; ++t) {
      Float x = theta[s][t];
#pragma omp parallel for reduction(+ : x) if (DO_PARALLEL)
      for (size_t g = 0; g < G; ++g) x *= phi[g][t];
      intensity_sum += x;
    }

    if (verbosity >= Verbosity::Info)
      cout << "count_sum=" << count_sum << " intensity_sum=" << intensity_sum
           << " prev scaling[" << s << "]=" << scaling[s];

    // NOTE: gamma_distribution takes a shape and scale parameter
    scaling[s] = gamma_distribution<Float>(
        scaling_prior_a + count_sum,
        scaling_prior_b + intensity_sum)(EntropySource::rng);
    if (verbosity >= Verbosity::Info)
      cout << "new scaling[" << s << "]=" << scaling[s] << endl;
  }
}

void VariantModel::gibbs_sample(const IMatrix &counts) {
  check_model(counts);
  sample_scaling();
  if (verbosity >= Verbosity::Everything)
    cout << "Log-likelihood = " << log_likelihood(counts) << endl;
  check_model(counts);
  sample_contributions(counts);
  if (verbosity >= Verbosity::Everything)
    cout << "Log-likelihood = " << log_likelihood(counts) << endl;
  check_model(counts);
  sample_phi();
  if (verbosity >= Verbosity::Everything)
    cout << "Log-likelihood = " << log_likelihood(counts) << endl;
  check_model(counts);
  sample_p();
  if (verbosity >= Verbosity::Everything)
    cout << "Log-likelihood = " << log_likelihood(counts) << endl;
  check_model(counts);
  sample_r();
  if (verbosity >= Verbosity::Everything)
    cout << "Log-likelihood = " << log_likelihood(counts) << endl;
  check_model(counts);
  sample_theta();
  if (verbosity >= Verbosity::Everything)
    cout << "Log-likelihood = " << log_likelihood(counts) << endl;
  check_model(counts);
}

void VariantModel::check_model(const IMatrix &counts) const {
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
  for (size_t g = 0; g < G; ++g)
    for (size_t t = 0; t < T; ++t) {
      if (p[g][t] <= 0 or p[g][t] >= 1)
        throw(runtime_error("P[" + to_string(g) + "][" + to_string(t) +
                            "] is smaller zero or larger 1: p=" +
                            to_string(p[g][t]) + "."));
      if ((1 - p[g][t]) / p[g][t] == 0)
        throw(runtime_error("(1-P)/P is zero for gene " + to_string(g) +
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
  if (priors.c0 == 0) throw(runtime_error("The prior c0 is zero."));
  if (priors.r0 == 0) throw(runtime_error("The prior r0 is zero."));
  if (priors.alpha == 0) throw(runtime_error("The prior r0 is zero."));
  if (priors.c == 0) throw(runtime_error("The prior c is zero."));
  if (priors.epsilon == 0) throw(runtime_error("The prior epsilon is zero."));
  if (priors.epsilon == 1) throw(runtime_error("The prior epsilon is unit."));
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

    os << "Scaling factors" << endl;
    for (size_t s = 0; s < pfa.S; ++s)
      os << (s > 0 ? "\t" : "") << pfa.scaling[s];
    os << endl;
  }

  return os;
}
