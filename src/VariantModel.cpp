#include <omp.h>
#include "VariantModel.hpp"
#include "compression.hpp"
#include "io.hpp"
#include "metropolis_hastings.hpp"
#include "montecarlo.hpp"
#include "pdist.hpp"
#include "stats.hpp"
#include "timer.hpp"

#define DO_PARALLEL 1

#define DEFAULT_SEPARATOR "\t"
#define DEFAULT_LABEL ""

using namespace std;
namespace FactorAnalysis {
const Float phi_scaling = 1.0;

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

VariantModel::Paths::Paths(const std::string &prefix, const std::string &suffix)
    : phi(prefix + "phi.txt" + suffix),
      theta(prefix + "theta.txt" + suffix),
      spot(prefix + "spot_scaling.txt" + suffix),
      experiment(prefix + "experiment_scaling.txt" + suffix),
      r_phi(prefix + "r.txt" + suffix),
      p_phi(prefix + "p.txt" + suffix),
      r_theta(prefix + "r_theta.txt" + suffix),
      p_theta(prefix + "p_theta.txt" + suffix){};

VariantModel::VariantModel(const Counts &c, const size_t T_,
                           const Hyperparameters &hyperparameters_,
                           const Parameters &parameters_, Verbosity verbosity_)
    : G(c.counts.n_rows),
      S(c.counts.n_cols),
      T(T_),
      E(c.experiment_names.size()),
      hyperparameters(hyperparameters_),
      parameters(parameters_),
      contributions_gene_type(G, T, arma::fill::zeros),
      contributions_spot_type(S, T, arma::fill::zeros),
      contributions_spot(S, arma::fill::zeros),
      contributions_experiment(E, arma::fill::zeros),
      phi(G, T),
      theta(S, T),
      spot_scaling(S, arma::fill::ones),
      experiment_scaling(E, arma::fill::ones),
      experiment_scaling_long(S, arma::fill::ones),
      r(G, T),
      p(G, T),
      r_theta(T),
      p_theta(T),
      verbosity(verbosity_) {
  // randomly initialize Phi
  if(verbosity >= Verbosity::Debug)
    cout << "initializing phi." << endl;
  // Phi = zeros(T,S)+1/T;
  phi.fill(1.0 / T / G);

  // initialize Theta
  // Theta = rand(P,T);
  // Theta = bsxfun(@rdivide,Phi,sum(Phi,1));
  if(verbosity >= Verbosity::Debug)
    cout << "initializing theta." << endl;
  for (size_t s = 0; s < S; ++s) {
    double sum = 0;
    for (size_t t = 0; t < T; ++t)
      sum += theta(s, t) = RandomDistribution::Uniform(EntropySource::rng);
    for (size_t t = 0; t < T; ++t) theta(s, t) /= sum;
  }

  // randomly initialize P
  // p_k=ones(T,1)*0.5;
  if(verbosity >= Verbosity::Debug)
    cout << "initializing p of phi." << endl;
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t g = 0; g < G; ++g) {
    const size_t thread_num = omp_get_thread_num();
    for (size_t t = 0; t < T; ++t)
      if (false)
        p(g, t) = 0.5 * G * T;
      else
        // NOTE: gamma_distribution takes a shape and scale parameter
        p(g, t) = prob_to_neg_odds(gamma_distribution<Float>(
            hyperparameters.phi_p_1,
            1 / hyperparameters.phi_p_2)(EntropySource::rngs[thread_num]));
  }

  // initialize R
  if(verbosity >= Verbosity::Debug)
    cout << "initializing r of phi." << endl;
  // r_k= 50/T*ones(T,1)
  r.fill(50.0 / G / T);

  // randomly initialize the contributions
  if(verbosity >= Verbosity::Debug)
    cout << "initializing contributions." << endl;
  for (size_t g = 0; g < G; ++g)
    for (size_t s = 0; s < S; ++s) {
      vector<double> prob(T);
      double z = 0;
      for (size_t t = 0; t < T; ++t)
        z += prob[t] = phi(g, t) * theta(s, t);
      for (size_t t = 0; t < T; ++t)
        prob[t] /= z;
      auto v = sample_multinomial<Int>(c.counts(g, s), prob);
      for (size_t t = 0; t < T; ++t) {
        contributions_gene_type(g, t) += v[t];
        contributions_spot_type(s, t) += v[t];
      }
      contributions_spot(s) += c.counts(g, s);
      contributions_experiment(c.experiments[s]) += c.counts(g, s);
    }

  if (parameters.activate_experiment_scaling) {
    // initialize experiment scaling factors
    if (parameters.activate_experiment_scaling) {
      if (verbosity >= Verbosity::Debug)
        cout << "initializing experiment scaling." << endl;
      for (size_t s = 0; s < S; ++s)
        experiment_scaling(c.experiments[s]) += contributions_spot(s);
      Float z = 0;
      for (size_t e = 0; e < E; ++e) z += experiment_scaling(e);
      z /= E;
      for (size_t e = 0; e < E; ++e) experiment_scaling(e) /= z;
      // copy the experiment scaling parameters into the spot-indexed vector
      update_experiment_scaling_long(c);
    }
  }

  // initialize spot scaling factors
  {
  if(verbosity >= Verbosity::Debug)
    cout << "initializing spot scaling." << endl;
    Float z = 0;
    for (size_t s = 0; s < S; ++s) {
      if (verbosity >= Verbosity::Debug)
        cout << "z = " << z << " spot_scaling(s) = " << spot_scaling(s)
             << " contributions_spot(s) = " << contributions_spot(s)
             << " experiment_scaling_long(s) = " << experiment_scaling_long(s);
      z += spot_scaling(s) = contributions_spot(s) / experiment_scaling_long(s);
      if (verbosity >= Verbosity::Debug)
        cout << " spot_scaling(s) = " << spot_scaling(s) << endl;
    }
    if (verbosity >= Verbosity::Debug)
      cout << "z = " << z << endl;
    z /= S;
    if (verbosity >= Verbosity::Debug)
      cout << "z = " << z << endl;
    for (size_t s = 0; s < S; ++s)
      spot_scaling(s) /= z;
  }

  // randomly initialize P
  if(verbosity >= Verbosity::Debug)
    cout << "initializing p of theta." << endl;
  for (size_t t = 0; t < T; ++t)
    p_theta[t] = prob_to_neg_odds(sample_beta<Float>(
        hyperparameters.theta_p_1, hyperparameters.theta_p_2));

  // randomly initialize R
  if(verbosity >= Verbosity::Debug)
    cout << "initializing r of theta." << endl;
  for (size_t t = 0; t < T; ++t)
    // NOTE: gamma_distribution takes a shape and scale parameter
    r_theta[t] = gamma_distribution<Float>(
        hyperparameters.theta_r_1,
        1 / hyperparameters.theta_r_2)(EntropySource::rng);
}

size_t num_lines(const string &path) {
  int number_of_lines = 0;
  string line;
  ifstream ifs(path);

  while (getline(ifs, line))
    ++number_of_lines;
  return number_of_lines;
}

VariantModel::VariantModel(const Counts &c, const Paths &paths,
                           const Hyperparameters &hyperparameters_,
                           const Parameters &parameters_, Verbosity verbosity_)
    : G(c.counts.n_rows),
      S(c.counts.n_cols),
      T(num_lines(paths.r_theta)),
      E(c.experiment_names.size()),
      hyperparameters(hyperparameters_),
      parameters(parameters_),
      contributions_gene_type(G, T, arma::fill::zeros),
      contributions_spot_type(S, T, arma::fill::zeros),
      contributions_spot(S, arma::fill::zeros),
      contributions_experiment(E, arma::fill::zeros),
      phi(parse_file<Matrix>(paths.phi, read_matrix, DEFAULT_SEPARATOR, DEFAULT_LABEL)),
      theta(parse_file<Matrix>(paths.theta, read_matrix, DEFAULT_SEPARATOR, DEFAULT_LABEL)),
      spot_scaling(parse_file<Vector>(paths.spot, read_vector, DEFAULT_SEPARATOR)),
      experiment_scaling(parse_file<Vector>(paths.experiment, read_vector, DEFAULT_SEPARATOR)),
      experiment_scaling_long(S),
      r(parse_file<Matrix>(paths.r_phi, read_matrix, DEFAULT_SEPARATOR, DEFAULT_LABEL)),
      p(parse_file<Matrix>(paths.p_phi, read_matrix, DEFAULT_SEPARATOR, DEFAULT_LABEL)),
      r_theta(parse_file<Vector>(paths.r_theta, read_vector, DEFAULT_SEPARATOR)),
      p_theta(parse_file<Vector>(paths.p_theta, read_vector, DEFAULT_SEPARATOR)),
      verbosity(verbosity_) {
  // set contributions to 0, as we do not have data at this point
  // NOTE: when data is available, before sampling any of the other parameters,
  // it is necessary to first sample the contributions!

  update_experiment_scaling_long(c);

  if (verbosity >= Verbosity::Debug)
    cout << *this << endl;
}

double VariantModel::log_likelihood(const IMatrix &counts) const {
  double l = 0;
  vector<double> alpha(G, hyperparameters.alpha);
  for (size_t t = 0; t < T; ++t) {
#pragma omp parallel for reduction(+ : l) if (DO_PARALLEL)
    for (size_t g = 0; g < G; ++g)
      // NOTE: log_gamma takes a shape and scale parameter
      l += log_gamma(r(g, t), hyperparameters.phi_r_1,
                     1.0 / hyperparameters.phi_r_2);
    if (std::isnan(l))
      cout << "Likelihood is NAN after adding the contribution due to "
              "Gamma-distributed r[g]["
           << t << "]." << endl;
#pragma omp parallel for reduction(+ : l) if (DO_PARALLEL)
    for (size_t g = 0; g < G; ++g)
      // NOTE: log_gamma takes a shape and scale parameter
      l += log_gamma(p(g, t), hyperparameters.phi_p_1,
                     1 / hyperparameters.phi_p_2);
    if (std::isnan(l))
      cout << "Likelihood is NAN after adding the contribution due to "
              "Beta-distributed p[g]["
           << t << "]." << endl;
#pragma omp parallel for reduction(+ : l) if (DO_PARALLEL)
    for (size_t g = 0; g < G; ++g) {
      // NOTE: log_gamma takes a shape and scale parameter
      l += log_gamma(phi(g, t), r(g, t), p(g, t));
      if (std::isnan(l))
        cout << "Likelihood is NAN after adding the contribution due to "
                "Gamma-distributed phi["
             << g << "][" << t << "]; phi=" << phi(g, t) << " r=" << r(g, t)
             << " p=" << p(g, t) << endl;
    }
#pragma omp parallel for reduction(+ : l) if (DO_PARALLEL)
    for (size_t g = 0; g < G; ++g)
      for (size_t s = 0; s < S; ++s)
        // l += log_poisson(contributions(g, s, t), phi(g, t) * theta(s, t));
        // TODO fix
        exit(-1);
    if (std::isnan(l))
      cout << "Likelihood is NAN after adding the contribution due to "
              "Poisson-distributed contributions[g][s]["
           << t << "]." << endl;
  }

  for (size_t g = 0; g < G; ++g) {
    vector<double> thetak(G, 0);
    for (size_t t = 0; t < T; ++t)
      thetak[g] = theta(g, t);
    l += log_dirichlet(thetak, alpha);
  }
  if (std::isnan(l))
    cout << "Likelihood is NAN after adding the contribution due to "
            "Dirichlet-distributed theta."
         << endl;

  /*
  for (size_t g = 0; g < G; ++g)
    for (size_t s = 0; s < S; ++s) {
      double rate = 0;
      for (size_t t = 0; t < T; ++t) rate += phi(g, t) * theta(s, t);
      l += log_poisson(counts(g, s), rate);
    }
    */
  return l;
}

void VariantModel::store(const Counts &counts, const string &prefix,
                   bool mean_and_variance) const {
  vector<string> factor_names;
  for (size_t t = 1; t <= T; ++t)
    factor_names.push_back("Factor " + to_string(t));
  auto &gene_names = counts.row_names;
  auto &spot_names = counts.col_names;
  write_matrix(phi, prefix + "phi.txt", gene_names, factor_names);
  write_matrix(r, prefix + "r.txt", gene_names, factor_names);
  write_matrix(p, prefix + "p.txt", gene_names, factor_names);
  write_matrix(theta, prefix + "theta.txt", spot_names, factor_names);
  write_vector(r_theta, prefix + "r_theta.txt", factor_names);
  write_vector(p_theta, prefix + "p_theta.txt", factor_names);
  write_vector(spot_scaling, prefix + "spot_scaling.txt", spot_names);
  write_vector(experiment_scaling, prefix + "experiment_scaling.txt", counts.experiment_names);
  write_matrix(contributions_gene_type, prefix + "contributions_gene_type.txt", gene_names, factor_names);
  write_matrix(contributions_spot_type, prefix + "contributions_spot_type.txt", spot_names, factor_names);
  // TODO: should we also write out contributions_spot and contributions_experiment?
  if (mean_and_variance) {
    write_matrix(posterior_expectations(), prefix + "means.txt",
                 gene_names, spot_names);
    write_matrix(posterior_expectations_poisson(), prefix + "means_poisson.txt",
                 gene_names, spot_names);
    write_matrix(posterior_variances(), prefix + "variances.txt",
                 gene_names, spot_names);
  }
}

void VariantModel::sample_contributions_sub(const IMatrix &counts, size_t g,
                                        size_t s, RNG &rng,
                                        Matrix &contrib_gene_type,
                                        Matrix &contrib_spot_type) {
  vector<double> rel_rate(T);
  double z = 0;
  // NOTE: in principle, lambda[g][s][t] is proportional to both
  // spot_scaling[s] and experiment_scaling[s]. However, these terms would
  // cancel. Thus, we do not multiply them in here.
  for (size_t t = 0; t < T; ++t) z += rel_rate[t] = phi(g, t) * theta(s, t);
  for (size_t t = 0; t < T; ++t) rel_rate[t] /= z;
  auto v = sample_multinomial<Int>(counts(g, s), rel_rate, rng);
  for (size_t t = 0; t < T; ++t) {
    contrib_gene_type(g, t) += v[t];
    contrib_spot_type(s, t) += v[t];
  }
}

/** sample count decomposition */
void VariantModel::sample_contributions(const IMatrix &counts) {
  if (verbosity >= Verbosity::Verbose)
    cout << "Sampling contributions" << endl;
  contributions_gene_type = Matrix(G, T, arma::fill::zeros);
  contributions_spot_type = Matrix(S, T, arma::fill::zeros);
#pragma omp parallel if (DO_PARALLEL)
  {
    Matrix contrib_gene_type(G, T, arma::fill::zeros);
    Matrix contrib_spot_type(S, T, arma::fill::zeros);
    const size_t thread_num = omp_get_thread_num();
#pragma omp for
    for (size_t g = 0; g < G; ++g)
      for (size_t s = 0; s < S; ++s)
        sample_contributions_sub(counts, g, s, EntropySource::rngs[thread_num], contrib_gene_type, contrib_spot_type);
#pragma omp critical
    {
      contributions_gene_type += contrib_gene_type;
      contributions_spot_type += contrib_spot_type;
    }
  }
}

vector<Float> VariantModel::compute_intensities_gene_type() const {
  vector<Float> intensities(T, 0);
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t t = 0; t < T; ++t)
    for (size_t g = 0; g < G; ++g)
      intensities[t] += phi(g, t);
  return intensities;
}


/** sample theta */
void VariantModel::sample_theta() {
  if (verbosity >= Verbosity::Verbose)
    cout << "Sampling Θ" << endl;
  const vector<Float> intensities = compute_intensities_gene_type();

#pragma omp parallel for if (DO_PARALLEL)
  for (size_t s = 0; s < S; ++s) {
    const Float scale = spot_scaling[s] * experiment_scaling_long[s];
    for (size_t t = 0; t < T; ++t)
      // NOTE: gamma_distribution takes a shape and scale parameter
      theta(s, t) = gamma_distribution<Float>(
          r_theta[t] + contributions_spot_type(s, t),
          1.0 / (p_theta[t] + intensities[t] * scale))(
          EntropySource::rng);
  }
  if ((parameters.enforce_mean & ForceMean::Theta) != ForceMean::None)
#pragma omp parallel for if (DO_PARALLEL)
    for (size_t s = 0; s < S; ++s) {
      double z = 0;
      for (size_t t = 0; t < T; ++t)
        z += theta(s, t);
      for (size_t t = 0; t < T; ++t)
        theta(s, t) /= z;
    }
}

double compute_conditional_theta(const pair<Float, Float> &x,
                                 const vector<Int> &count_sums,
                                 const vector<Float> &weight_sums,
                                 const Hyperparameters &hyperparameters) {
  const size_t S = count_sums.size();
  const Float current_r = x.first;
  const Float current_p = x.second;
  double r = log_beta_odds(current_p, hyperparameters.theta_p_1,
                           hyperparameters.theta_p_2)
             // NOTE: gamma_distribution takes a shape and scale parameter
             + log_gamma(current_r, hyperparameters.theta_r_1,
                         1 / hyperparameters.theta_r_2)
             + S * (current_r * log(current_p) - lgamma(current_r));
#pragma omp parallel for reduction(+ : r) if (DO_PARALLEL)
  for (size_t s = 0; s < S; ++s)
    // The next line is part of the negative binomial distribution.
    // The other factors aren't needed as they don't depend on either of
    // r[t] and p[t], and thus would cancel when computing the score
    // ratio.
    r += lgamma(current_r + count_sums[s])
         - (current_r + count_sums[s]) * log(current_p + weight_sums[s]);
  return r;
}

/** sample p_theta and r_theta */
/* This is a simple Metropolis-Hastings sampling scheme */
void VariantModel::sample_p_and_r_theta() {
  if (verbosity >= Verbosity::Verbose)
    cout << "Sampling P_theta and R_theta" << endl;

  auto gen = [&](const pair<Float, Float> &x, mt19937 &rng) {
    normal_distribution<double> rnorm;
    const double f1 = exp(rnorm(rng));
    const double f2 = exp(rnorm(rng));
    return pair<Float, Float>(f1 * x.first, f2 * x.second);
  };

  for (size_t t = 0; t < T; ++t) {
    Float weight_sum = 0;
#pragma omp parallel for reduction(+ : weight_sum) if (DO_PARALLEL)
    for (size_t g = 0; g < G; ++g) weight_sum += phi(g, t);
    MetropolisHastings mh(parameters.temperature, parameters.prop_sd,
                          verbosity);

    vector<Int> count_sums(S, 0);
    vector<Float> weight_sums(S, 0);
#pragma omp parallel for if (DO_PARALLEL)
    for (size_t s = 0; s < S; ++s) {
      count_sums[s] = contributions_spot_type(s, t);
      weight_sums[s] = weight_sum * spot_scaling[s] * experiment_scaling_long[s];
    }
    auto res = mh.sample(pair<Float, Float>(r_theta[t], p_theta[t]),
                         parameters.n_iter, EntropySource::rng, gen,
                         compute_conditional_theta, count_sums, weight_sums,
                         hyperparameters);
    r_theta[t] = res.first;
    p_theta[t] = res.second;
  }
}

double compute_conditional(const pair<Float, Float> &x, Int count_sum,
                           Float weight_sum,
                           const Hyperparameters &hyperparameters) {
  const Float current_r = x.first;
  const Float current_p = x.second;
  return log_beta_odds(current_p, hyperparameters.phi_p_1,
                       hyperparameters.phi_p_2)
         // NOTE: gamma_distribution takes a shape and scale parameter
         + log_gamma(current_r, hyperparameters.phi_r_1,
                     1 / hyperparameters.phi_r_2)
         // The next lines are part of the negative binomial distribution.
         // The other factors aren't needed as they don't depend on either of
         // r[g][t] and p[g][t], and thus would cancel when computing the score
         // ratio.
         + current_r * log(current_p)
         - (current_r + count_sum) * log(current_p + weight_sum)
         + lgamma(current_r + count_sum) - lgamma(current_r);
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
      weight_sum += theta(s, t) * spot_scaling[s] * experiment_scaling_long[s];
    MetropolisHastings mh(parameters.temperature, parameters.prop_sd,
                          verbosity);

#pragma omp parallel for if (DO_PARALLEL)
    for (size_t g = 0; g < G; ++g) {
      const Int count_sum = contributions_gene_type(g, t);
      const size_t thread_num = omp_get_thread_num();
      auto res
          = mh.sample(pair<Float, Float>(r(g, t), p(g, t)), parameters.n_iter,
                      EntropySource::rngs[thread_num], gen, compute_conditional,
                      count_sum, weight_sum, hyperparameters);
      r(g, t) = res.first;
      p(g, t) = res.second;
    }
  }
}

Float VariantModel::sample_phi_sub(size_t g, size_t t, Float theta_t,
                                   RNG &rng) const {
  // NOTE: gamma_distribution takes a shape and scale parameter
  return gamma_distribution<Float>(r(g, t) + contributions_gene_type(g, t),
                                   1.0 / (p(g, t) + theta_t))(rng);
}

/** sample phi */
void VariantModel::sample_phi() {
  if (verbosity >= Verbosity::Verbose) cout << "Sampling Φ" << endl;
  Vector theta_t(T, arma::fill::zeros);
  for (size_t s = 0; s < S; ++s) {
    const Float prod = spot_scaling[s] * experiment_scaling_long[s];
    for (size_t t = 0; t < T; ++t)
      theta_t[t] += theta(s, t) * prod;
  }

#pragma omp parallel for if (DO_PARALLEL)
  for (size_t g = 0; g < G; ++g) {
    const size_t thread_num = omp_get_thread_num();
    for (size_t t = 0; t < T; ++t)
      phi(g, t) = sample_phi_sub(g, t, theta_t[t], EntropySource::rngs[thread_num]);
  }
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

/** sample spot scaling factors */
void VariantModel::sample_spot_scaling() {
  if (verbosity >= Verbosity::Verbose)
    cout << "Sampling spot scaling factors" << endl;
  Vector phi_marginal(T, arma::fill::zeros);
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t t = 0; t < T; ++t)
    for (size_t g = 0; g < G; ++g)
      phi_marginal(t) += phi(g, t);
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t s = 0; s < S; ++s) {
    const Int summed_contribution = contributions_spot(s);

    Float intensity_sum = 0;
    for (size_t t = 0; t < T; ++t)
      intensity_sum += phi_marginal(t) * theta(s, t);
    intensity_sum *= experiment_scaling_long[s];

    /*
    if (verbosity >= Verbosity::Debug)
      cout << "summed_contribution=" << summed_contribution
           << " intensity_sum=" << intensity_sum
           << " prev spot_scaling[" << s << "]=" << spot_scaling[s];
    */

    // NOTE: gamma_distribution takes a shape and scale parameter
    spot_scaling[s] = gamma_distribution<Float>(
        hyperparameters.spot_a + summed_contribution,
        1.0 / (hyperparameters.spot_b + intensity_sum))(EntropySource::rng);
    /*
    if (verbosity >= Verbosity::Debug)
      cout << "new spot_scaling[" << s << "]=" << spot_scaling[s] << endl;
    */
  }

  if ((parameters.enforce_mean & ForceMean::Spot) != ForceMean::None) {
    double z = 0;
#pragma omp parallel for reduction(+ : z) if (DO_PARALLEL)
    for(size_t s = 0; s < S; ++s)
      z += spot_scaling[s];
    z /= S;
#pragma omp parallel for if (DO_PARALLEL)
    for(size_t s = 0; s < S; ++s)
      spot_scaling[s] /= z;
  }
}

/** sample experiment scaling factors */
void VariantModel::sample_experiment_scaling(const Counts &data) {
  if (verbosity >= Verbosity::Verbose)
    cout << "Sampling experiment scaling factors" << endl;

  Vector phi_marginal(T, arma::fill::zeros);
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t t = 0; t < T; ++t)
    for (size_t g = 0; g < G; ++g)
      phi_marginal(t) += phi(g, t);
  vector<Float> intensity_sums(E, 0);
  // TODO: improve parallelism
  for (size_t s = 0; s < S; ++s) {
    double x = 0;
#pragma omp parallel for reduction(+ : x) if (DO_PARALLEL)
    for (size_t t = 0; t < T; ++t)
      x += phi_marginal(t) * theta(s, t);
    x *= spot_scaling[s];
    intensity_sums[data.experiments[s]] += x;
  }

  if (verbosity >= Verbosity::Debug)
    for (size_t e = 0; e < E; ++e)
      cout << "contributions_experiment[" << e << "]=" << contributions_experiment[e] << endl
           << "intensity_sum=" << intensity_sums[e] << endl
           << "prev experiment_scaling[" << e << "]=" << experiment_scaling[e]
           << endl;

  for (size_t e = 0; e < E; ++e) {
    // NOTE: gamma_distribution takes a shape and scale parameter
    experiment_scaling[e] = gamma_distribution<Float>(
        hyperparameters.experiment_a + contributions_experiment(e),
        1.0 / (hyperparameters.experiment_b + intensity_sums[e]))(
        EntropySource::rng);
    if (verbosity >= Verbosity::Debug)
      cout << "new experiment_scaling[" << e << "]=" << experiment_scaling[e]
           << endl;
  }

  // copy the experiment scaling parameters into the spot-indexed vector
  update_experiment_scaling_long(data);

  if ((parameters.enforce_mean & ForceMean::Experiment) != ForceMean::None) {
    double z = 0;
#pragma omp parallel for reduction(+ : z) if (DO_PARALLEL)
    for(size_t s = 0; s < S; ++s)
      z += experiment_scaling_long[s];
    z /= S;
#pragma omp parallel for if (DO_PARALLEL)
    for(size_t s = 0; s < S; ++s)
      experiment_scaling_long[s] /= z;

    for(size_t e = 0; e < E; ++e)
      experiment_scaling[e] /= z;
  }
}

/** copy the experiment scaling parameters into the spot-indexed vector */
void VariantModel::update_experiment_scaling_long(const Counts &data) {
  for (size_t s = 0; s < S; ++s)
    experiment_scaling_long[s] = experiment_scaling[data.experiments[s]];
}

void VariantModel::gibbs_sample(const Counts &data, bool timing) {
  check_model(data.counts);

  Timer timer;
  sample_contributions(data.counts);
  if (timing and verbosity >= Verbosity::Info)
    cout << "This took " << timer.tock() << "μs." << endl;
  if (verbosity >= Verbosity::Everything)
    cout << "Log-likelihood = " << log_likelihood(data.counts) << endl;
  check_model(data.counts);

  timer.tick();
  sample_spot_scaling();
  if (timing and verbosity >= Verbosity::Info)
    cout << "This took " << timer.tock() << "μs." << endl;
  if (verbosity >= Verbosity::Everything)
    cout << "Log-likelihood = " << log_likelihood(data.counts) << endl;
  check_model(data.counts);

  if (E > 1 and parameters.activate_experiment_scaling) {
    timer.tick();
    sample_experiment_scaling(data);
    if (timing and verbosity >= Verbosity::Info)
      cout << "This took " << timer.tock() << "μs." << endl;
    if (verbosity >= Verbosity::Everything)
      cout << "Log-likelihood = " << log_likelihood(data.counts) << endl;
    check_model(data.counts);
  }

  timer.tick();
  sample_p_and_r();
  if (timing and verbosity >= Verbosity::Info)
    cout << "This took " << timer.tock() << "μs." << endl;
  if (verbosity >= Verbosity::Everything)
    cout << "Log-likelihood = " << log_likelihood(data.counts) << endl;
  check_model(data.counts);

  timer.tick();
  sample_p_and_r_theta();
  if (timing and verbosity >= Verbosity::Info)
    cout << "This took " << timer.tock() << "μs." << endl;
  if (verbosity >= Verbosity::Everything)
    cout << "Log-likelihood = " << log_likelihood(data.counts) << endl;
  check_model(data.counts);

  timer.tick();
  sample_phi();
  if (timing and verbosity >= Verbosity::Info)
    cout << "This took " << timer.tock() << "μs." << endl;
  if (verbosity >= Verbosity::Everything)
    cout << "Log-likelihood = " << log_likelihood(data.counts) << endl;
  check_model(data.counts);

  timer.tick();
  sample_theta();
  if (timing and verbosity >= Verbosity::Info)
    cout << "This took " << timer.tock() << "μs." << endl;
  if (verbosity >= Verbosity::Everything)
    cout << "Log-likelihood = " << log_likelihood(data.counts) << endl;
  check_model(data.counts);
}

vector<Int> VariantModel::sample_reads(size_t g, size_t s, size_t n) const {
  vector<Float> prods(T);
  for (size_t t = 0; t < T; ++t)
    prods[t] = theta(s, t) * spot_scaling[s] * experiment_scaling_long[s];

  vector<Int> v(n, 0);
// TODO parallelize
// #pragma omp parallel for if (DO_PARALLEL)
  for (size_t i = 0; i < n; ++i)
    for (size_t t = 0; t < T; ++t)
      v[i] += sample_negative_binomial(
          r(g, t), prods[t] / (prods[t] + p(g, t)),
          EntropySource::rng);
  return v;
}

double VariantModel::posterior_expectation(size_t g, size_t s) const {
  double x = 0;
  for(size_t t = 0; t < T; ++t)
    x += r(g, t) / p(g, t) * theta(s, t);
  x *= spot_scaling[s] * experiment_scaling_long[s];
  return x;
}

double VariantModel::posterior_expectation_poisson(size_t g, size_t s) const {
  double x = 0;
  for(size_t t = 0; t < T; ++t)
    x += phi(g, t) * theta(s, t);
  x *= spot_scaling[s] * experiment_scaling_long[s];
  return x;
}


double VariantModel::posterior_variance(size_t g, size_t s) const {
  double x = 0;
  double prod_ = spot_scaling[s] * experiment_scaling_long[s];
  for(size_t t = 0; t < T; ++t) {
    double prod = theta(s, t) * prod_;
    x += r(g, t) * prod / (prod + p(g, t)) / p(g, t) / p(g, t);
  }
  return x;
}

Matrix VariantModel::posterior_expectations() const {
  Matrix m(G, S);
  for(size_t g = 0; g < G; ++g)
    for(size_t s = 0; s < S; ++s)
      m(g, s) = posterior_expectation(g, s);
  return m;
}

Matrix VariantModel::posterior_expectations_poisson() const {
  Matrix m(G, S);
  for(size_t g = 0; g < G; ++g)
    for(size_t s = 0; s < S; ++s)
      m(g, s) = posterior_expectation_poisson(g, s);
  return m;
}

Matrix VariantModel::posterior_variances() const {
  Matrix m(G, S);
  for(size_t g = 0; g < G; ++g)
    for(size_t s = 0; s < S; ++s)
      m(g, s) = posterior_variance(g, s);
  return m;
}

void VariantModel::check_model(const IMatrix &counts) const {
  return;
  // check that the contributions add up to the observations
  /*
  for (size_t g = 0; g < G; ++g)
    for (size_t s = 0; s < S; ++s) {
      Int z = 0;
      for (size_t t = 0; t < T; ++t) z += contributions(g, s, t);
      if (z != counts(g, s))
        throw(runtime_error(
            "Contributions do not add up to observations for gene " +
            to_string(g) + " in spot " + to_string(s) + "."));
    }
  */

  // check that phi is positive
  for (size_t g = 0; g < G; ++g)
    for (size_t t = 0; t < T; ++t) {
      /*
      if (phi[g][t] == 0)
        throw(runtime_error("Phi is zero for gene " + to_string(g) +
                            " in factor " + to_string(t) + "."));
                            */
      if (phi(g, t) < 0)
        throw(runtime_error("Phi is negative for gene " + to_string(g) +
                            " in factor " + to_string(t) + "."));
    }

  // check that theta is positive
  for (size_t s = 0; s < S; ++s)
    for (size_t t = 0; t < T; ++t) {
      if (theta(s, t) == 0)
        throw(runtime_error("Theta is zero for spot " + to_string(s) +
                            " in factor " + to_string(t) + "."));
      if (theta(s, t) < 0)
        throw(runtime_error("Theta is negative for spot " + to_string(s) +
                            " in factor " + to_string(t) + "."));
    }

  // check that r and p are positive, and that p is < 1
  for (size_t g = 0; g < G; ++g)
    for (size_t t = 0; t < T; ++t) {
      if (p(g, t) < 0)
        throw(runtime_error("P[" + to_string(g) + "][" + to_string(t) +
                            "] is smaller zero: p=" + to_string(p(g, t)) +
                            "."));
      if (p(g, t) == 0)
        throw(runtime_error("P is zero for gene " + to_string(g) +
                            " in factor " + to_string(t) + "."));

      if (r(g, t) < 0)
        throw(runtime_error("R[" + to_string(g) + "][" + to_string(t) +
                            "] is smaller zero: r=" + to_string(r(g, t)) +
                            "."));
      if (r(g, t) == 0)
        throw(runtime_error("R is zero for gene " + to_string(g) +
                            " in factor " + to_string(t) + "."));
    }

  // check hyperparameters
  if (hyperparameters.phi_r_1 == 0)
    throw(runtime_error("The prior phi_r_1 is zero."));
  if (hyperparameters.phi_r_2 == 0)
    throw(runtime_error("The prior phi_r_2 is zero."));
  if (hyperparameters.phi_p_1 == 0)
    throw(runtime_error("The prior phi_p_1 is zero."));
  if (hyperparameters.phi_p_2 == 0)
    throw(runtime_error("The prior phi_p_2 is zero."));
  if (hyperparameters.alpha == 0)
    throw(runtime_error("The prior alpha is zero."));
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
        os << (t > 0 ? "\t" : "") << pfa.phi(g, t);
      os << endl;
    }

    size_t phi_zeros = 0;
    os << "Φ factor sums" << endl;
    for (size_t t = 0; t < pfa.T; ++t) {
      double sum = 0;
      for (size_t g = 0; g < pfa.G; ++g) {
        if (pfa.phi(g, t) == 0) phi_zeros++;
        sum += pfa.phi(g, t);
      }
      os << (t > 0 ? "\t" : "") << sum;
    }
    os << endl;
    os << "There are " << phi_zeros << " zeros in Φ. This corresponds to "
       << (100.0 * phi_zeros / pfa.T / pfa.G) << "%." << endl;

    os << "Θ" << endl;
    for (size_t s = 0; s < min<size_t>(pfa.S, 10); ++s) {
      for (size_t t = 0; t < pfa.T; ++t)
        os << (t > 0 ? "\t" : "") << pfa.theta(s, t);
      os << endl;
    }

    size_t theta_zeros = 0;
    os << "Θ factor sums" << endl;
    for (size_t t = 0; t < pfa.T; ++t) {
      double sum = 0;
      for (size_t s = 0; s < pfa.S; ++s) {
        if (pfa.theta(s, t) == 0) theta_zeros++;
        sum += pfa.theta(s, t);
      }
      os << (t > 0 ? "\t" : "") << sum;
    }
    os << endl;
    os << "There are " << theta_zeros << " zeros in Θ." << endl;

    os << "R" << endl;
    for (size_t g = 0; g < min<size_t>(pfa.G, 10); ++g) {
      for (size_t t = 0; t < pfa.T; ++t)
        os << (t > 0 ? "\t" : "") << pfa.r(g, t);
      os << endl;
    }

    size_t r_zeros = 0;
    for (size_t g = 0; g < pfa.G; ++g)
      for (size_t t = 0; t < pfa.T; ++t)
        if (pfa.r(g, t) == 0) r_zeros++;
    os << "There are " << r_zeros << " zeros in r. This corresponds to "
       << (100.0 * r_zeros / pfa.G / pfa.T) << "%." << endl;

    os << "P" << endl;
    for (size_t g = 0; g < min<size_t>(pfa.G, 10); ++g) {
      for (size_t t = 0; t < pfa.T; ++t)
        os << (t > 0 ? "\t" : "") << pfa.p(g, t);
      os << endl;
    }

    size_t p_zeros = 0;
    for (size_t g = 0; g < pfa.G; ++g)
      for (size_t t = 0; t < pfa.T; ++t)
        if (pfa.p(g, t) == 0) p_zeros++;
    os << "There are " << p_zeros << " zeros in p. This corresponds to "
       << (100.0 * p_zeros / pfa.G / pfa.T) << "%." << endl;

    os << "Spot scaling factors" << endl;
    for (size_t s = 0; s < pfa.S; ++s)
      os << (s > 0 ? "\t" : "") << pfa.spot_scaling[s];
    os << endl;
    size_t spot_scaling_zeros = 0;
    for (size_t s = 0; s < pfa.S; ++s)
      if (pfa.spot_scaling[s] == 0) spot_scaling_zeros++;
    os << "There are " << spot_scaling_zeros << " zeros in spot_scaling." << endl;
    os << Stats::summary(pfa.spot_scaling) << endl;

    if (pfa.parameters.activate_experiment_scaling) {
      os << "Experiment scaling factors" << endl;
      for (size_t e = 0; e < pfa.E; ++e)
        os << (e > 0 ? "\t" : "") << pfa.experiment_scaling[e];
      os << endl;
      size_t experiment_scaling_zeros = 0;
      for (size_t e = 0; e < pfa.E; ++e)
        if (pfa.experiment_scaling[e] == 0)
          spot_scaling_zeros++;
      os << "There are " << experiment_scaling_zeros
         << " zeros in experiment_scaling." << endl;
      os << Stats::summary(pfa.experiment_scaling) << endl;
    }
  }

  os << "R_theta factors" << endl;
  for (size_t t = 0; t < pfa.T; ++t)
    os << (t > 0 ? "\t" : "") << pfa.r_theta[t];
  os << endl;
  size_t r_theta_zeros = 0;
  for (size_t t = 0; t < pfa.T; ++t)
    if (pfa.r_theta[t] == 0) r_theta_zeros++;
  os << "There are " << r_theta_zeros << " zeros in R_theta." << endl;
  os << Stats::summary(pfa.r_theta) << endl;

  os << "P_theta factors" << endl;
  for (size_t t = 0; t < pfa.T; ++t)
    os << (t > 0 ? "\t" : "") << pfa.p_theta[t];
  os << endl;
  size_t p_theta_zeros = 0;
  for (size_t t = 0; t < pfa.T; ++t)
    if (pfa.p_theta[t] == 0) p_theta_zeros++;
  os << "There are " << p_theta_zeros << " zeros in P_theta." << endl;
  os << Stats::summary(pfa.p_theta) << endl;

  return os;
}
}
