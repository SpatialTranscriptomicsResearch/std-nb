#include "Experiment.hpp"
#include "Model.hpp"

using namespace std;

namespace PoissonFactorization {

Experiment::Experiment(Model *model_, const Counts &data_, size_t T_,
                       const Parameters &parameters_)
    : model(model_),
      G(data_.counts.n_rows),
      S(data_.counts.n_cols),
      T(T_),
      data(data_),
      coords(data.parse_coords()),
      parameters(parameters_),
      contributions_gene_type(G, T, arma::fill::zeros),
      contributions_spot_type(S, T, arma::fill::zeros),
      contributions_gene(rowSums<Vector>(data.counts)),
      contributions_spot(colSums<Vector>(data.counts)),
      phi_l(G, T, arma::fill::ones),
      phi_b(G, 1, arma::fill::ones),
      weights(S, T, parameters),
      field(Matrix(S, T, arma::fill::ones)),
      spot(S, arma::fill::ones) {
  LOG(debug) << "Experiment G = " << G << " S = " << S << " T = " << T;
  /* TODO consider to reactivate
  if (false) {
    // initialize:
    //  * contributions_gene_type
    //  * contributions_spot_type
    LOG(debug) << "Initializing contributions.";
    sample_contributions(c.counts);
  }
  */
  LOG(debug) << "Coords: " << coords;

  // initialize spot scaling factors
  {
    LOG(debug) << "Initializing spot scaling.";
    Float z = 0;
    for (size_t s = 0; s < S; ++s)
      z += spot(s) = contributions_spot(s);
    z /= S;
    for (size_t s = 0; s < S; ++s)
      spot(s) /= z;
  }

  if (not parameters.targeted(Target::theta))
    weights.matrix.ones();

  if (not parameters.targeted(Target::spot))
    spot.ones();
}

void Experiment::store(const string &prefix,
                       const vector<size_t> &order) const {
  auto factor_names = form_factor_names(T);
  auto &gene_names = data.row_names;
  auto &spot_names = data.col_names;

  string suffix = "";
  string extension = boost::filesystem::path(data.path).extension().c_str();
  if (extension == ".gz" or extension == ".bz2")
    suffix = extension;
  boost::filesystem::create_symlink(
      data.path, prefix + "counts" + FILENAME_ENDING + suffix);

#pragma omp parallel sections if (DO_PARALLEL)
  {
#pragma omp section
    write_matrix(phi_l, prefix + "feature-gamma_prior-r" + FILENAME_ENDING,
                 parameters.compression_mode, gene_names, factor_names, order);
#pragma omp section
    write_matrix(phi_b,
                 prefix + "baselinefeature-gamma_prior-r" + FILENAME_ENDING,
                 parameters.compression_mode, gene_names, {1, "Baseline"}, {});
#pragma omp section
    weights.store(prefix, spot_names, factor_names, order);
#pragma omp section
    write_vector(spot, prefix + "spot-scaling" + FILENAME_ENDING,
                 parameters.compression_mode, spot_names);
#pragma omp section
    write_matrix(field, prefix + "raw-field" + FILENAME_ENDING,
                 parameters.compression_mode, spot_names, factor_names, order);
#pragma omp section
    write_matrix(expected_spot_type(),
                 prefix + "expected-mix" + FILENAME_ENDING,
                 parameters.compression_mode, spot_names, factor_names, order);
#pragma omp section
    write_matrix(expected_gene_type(),
                 prefix + "expected-features" + FILENAME_ENDING,
                 parameters.compression_mode, gene_names, factor_names, order);
#pragma omp section
    {
      auto phi_marginal = marginalize_genes();
      auto f = field;
      f.each_row() %= phi_marginal.t();
      f.each_col() %= spot;
      write_matrix(f, prefix + "expected-field" + FILENAME_ENDING,
                   parameters.compression_mode, spot_names, factor_names,
                   order);
    }
#pragma omp section
    write_matrix(contributions_gene_type,
                 prefix + "contributions_gene_type" + FILENAME_ENDING,
                 parameters.compression_mode, gene_names, factor_names, order);
#pragma omp section
    write_matrix(contributions_spot_type,
                 prefix + "contributions_spot_type" + FILENAME_ENDING,
                 parameters.compression_mode, spot_names, factor_names, order);
#pragma omp section
    write_vector(contributions_gene,
                 prefix + "contributions_gene" + FILENAME_ENDING,
                 parameters.compression_mode, gene_names);
#pragma omp section
    write_vector(contributions_spot,
                 prefix + "contributions_spot" + FILENAME_ENDING,
                 parameters.compression_mode, spot_names);
  }
}

void Experiment::restore(const string &prefix) {
  contributions_gene_type = parse_file<Matrix>(
      prefix + "contributions_gene_type" + FILENAME_ENDING, read_matrix, "\t");
  contributions_spot_type = parse_file<Matrix>(
      prefix + "contributions_spot_type" + FILENAME_ENDING, read_matrix, "\t");
  contributions_gene
      = parse_file<Vector>(prefix + "contributions_gene" + FILENAME_ENDING,
                           read_vector<Vector>, "\t");
  contributions_spot
      = parse_file<Vector>(prefix + "contributions_spot" + FILENAME_ENDING,
                           read_vector<Vector>, "\t");

  phi_l = parse_file<Matrix>(prefix + "feature-gamma_prior-r" + FILENAME_ENDING,
                             read_matrix, "\t");
  phi_b = parse_file<Matrix>(
      prefix + "baselinefeature-gamma_prior-r" + FILENAME_ENDING, read_matrix,
      "\t");

  weights.restore(prefix);

  field = parse_file<Matrix>(prefix + "raw-field" + FILENAME_ENDING,
                             read_matrix, "\t");
  spot = parse_file<Vector>(prefix + "spot-scaling" + FILENAME_ENDING,
                            read_vector<Vector>, "\t");
}

Matrix Experiment::log_likelihood() const {
  Matrix l(G, S);
  const size_t K = min<size_t>(20, T);
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t g = 0; g < G; ++g) {
    Vector ps = model->phi_p.row(g).t();
    for (size_t t = 0; t < T; ++t)
      // NOTE conv neg bin has opposite interpretation of p
      ps[t] = 1 - neg_odds_to_prob(ps[t]);
    Vector rs(T);
    for (size_t s = 0; s < S; ++s) {
      for (size_t t = 0; t < T; ++t)
        rs[t] = model->phi_r(g, t) * phi_l(g, t) * theta(s, t) * spot(s);
      double x = convolved_negative_binomial(data.counts(g, s), K, rs, ps);
      LOG(debug) << "Computing log likelihood for g/s = " << g << "/" << s
                 << " counts = " << data.counts(g, s) << " l = " << x;
      l(g, s) += x;
    }
  }
  return l;
}

double neg_log_posterior(const Vector &y, size_t count, const Vector &r,
                         const Vector &p) {
  const size_t T = y.size();
  Vector x = count * gibbs(y);
  double l = 0;
  if (noisy) {
    LOG(trace) << "y = " << y;
    LOG(trace) << "x = " << x;
  }
  for (size_t t = 0; t < T; ++t)
    // TODO only compute relevant terms
    l += log_negative_binomial(x[t], r[t], p[t]);
  return -l;
}

Vector neg_grad_log_posterior(const Vector &y, size_t count, const Vector &r,
                              const Vector &p) {
  const size_t T = y.size();
  Vector x = count * gibbs(y);

  if (noisy) {
    LOG(trace) << "count = " << count;
    LOG(trace) << "y = " << y;
    LOG(trace) << "x = " << x;
  }

  Vector tmp(T, arma::fill::zeros);
  for (size_t t = 0; t < T; ++t)
    // TODO use digamma_diff
    tmp(t) = log(p[t]) + digamma(x(t) + r[t]) - digamma(x(t) + 1);

  if (noisy)
    LOG(trace) << "tmp = " << tmp;

  double z = 0;
  for (size_t t = 0; t < T; ++t)
    z += x(t) / count * tmp(t);

  Vector grad(T, arma::fill::zeros);
  for (size_t t = 0; t < T; ++t)
    grad(t) = -x(t) * (tmp(t) - z);

  if (noisy)
    LOG(debug) << "grad = " << grad;
  return grad;
}

/** sample count decomposition */
Vector Experiment::sample_contributions_gene_spot(size_t g, size_t s,
                                                  RNG &rng) const {
  Vector cnts(T, arma::fill::zeros);

  const size_t count = data.counts(g, s);

  if (count > 0) {
    if (T == 1) {
      cnts[0] = count;
    } else if (parameters.contributions_map) {
      Vector r(T);
      for (size_t t = 0; t < T; ++t)
        r[t] = model->phi_r(g, t) * phi_l(g, t) * phi_b(g) * theta(s, t)
               * spot(s);

      Vector p(T);
      for (size_t t = 0; t < T; ++t)
        p[t] = neg_odds_to_prob(model->phi_p(g, t));

      if (noisy) {
        for (size_t t = 0; t < T; ++t)
          LOG(debug) << "r = " << r[t];
        for (size_t t = 0; t < T; ++t)
          LOG(debug) << "p = " << p[t];
      }

      if (true) {
        for (size_t t = 0; t < T; ++t)
          cnts[t] = log(r[t] / model->phi_p(g, t));

        Vector mean;
        // Vector mean = count * gibbs(cnts);
        const size_t max_iter = (noisy ? 10 : 1);
        for (size_t iter = 0; iter < max_iter; ++iter) {
          mean = count * gibbs(cnts);
          if (noisy)
            LOG(verbose) << "cnts 0: " << (count * gibbs(cnts)).t();
          for (size_t i = 0; i < parameters.hmc_N; ++i) {
            if (noisy)
              LOG(debug) << "cnts " << i + 1 << ": "
                         << (count * gibbs(cnts).t());
            cnts = HMC::sample(cnts, neg_log_posterior, neg_grad_log_posterior,
                               parameters.hmc_L, parameters.hmc_epsilon, rng,
                               count, r, p);
            mean += count * gibbs(cnts);
          }
          mean /= parameters.hmc_N + 1;
          if (noisy) {
            double z = 0;
            for (auto &m : mean)
              z += m;
            LOG(verbose) << "cnts X iter = " << iter << " z = " << z
                         << " cnt=" << count;
            LOG(verbose) << "cnts X: " << (count * gibbs(cnts)).t();
            LOG(verbose) << "cnts m: " << mean.t();
          }
        }
        return mean;
      } else {
        for (size_t iter = 0; iter < 10; ++iter) {
          if (noisy)
            LOG(verbose) << "Iteration " << iter
                         << " cnts = " << count * gibbs(cnts);
          auto grad = neg_grad_log_posterior(cnts, count, r, p);
          if (noisy)
            LOG(verbose) << "Iteration " << iter << " grad = " << grad;

          // TODO reconsider activating this
          // this was mostly de-activated because it doesn't for the two-factor
          // case
          // it should work better for more than two factors
          if (false) {
            double l = 0;
            for (auto &x : grad)
              l += x * x;
            l = sqrt(l);

            grad /= l;
          } else
            grad *= 1e-2;

          if (noisy)
            LOG(verbose) << "Iteration " << iter << " grad = " << grad;

          // TODO check convergence
          cnts = cnts - grad;
        }
      }
      if (noisy)
        LOG(verbose) << "Final cnts = " << count * gibbs(cnts);
      cnts = count * gibbs(cnts);

    } else {
      auto log_posterior_difference = [&](const Vector &v, size_t i, size_t j,
                                          size_t n) {
        double l = 0;

        if (noisy)
          LOG(debug) << "i=" << i << " j=" << j << " n=" << n
                     << " v[i]=" << v[i] << " v[j]=" << v[j];

        const double r_i = model->phi_r(g, i) * phi_l(g, i);
        const double no_i = model->phi_p(g, i);
        const double prod_i = r_i * theta(s, i) * spot(s);

        const double r_j = model->phi_r(g, j) * phi_l(g, j);
        const double no_j = model->phi_p(g, j);
        const double prod_j = r_j * theta(s, j) * spot(s);

        // TODO determine continous-valued mean by optimizing the posterior
        // TODO handle infinities
        /*
        if(prod + v[t] == 0)
          return -numeric_limits<double>::infinity();
        */

        if (noisy)
          LOG(debug) << "r_i=" << r_i << " no_i=" << no_i
                     << " prod_i=" << prod_i << " r_j=" << r_j
                     << " no_j=" << no_j << " prod_j=" << prod_j;

        // subtract current score contributions
        l -= lgamma_diff_1p(v[i], prod_i) - v[i] * log(1 + no_i);
        l -= lgamma_diff_1p(v[j], prod_j) - v[j] * log(1 + no_j);

        // add proposed score contributions
        l += lgamma_diff_1p(v[i] - n, prod_i) - (v[i] - n) * log(1 + no_i);
        l += lgamma_diff_1p(v[j] + n, prod_j) - (v[j] + n) * log(1 + no_j);

        return l;
      };

      // TODO use full-conditional expected counts instead of one sample

      // sample x_gst for all t
      vector<double> mean_prob(T, 0);
      double z = 0;
      for (size_t t = 0; t < T; ++t)
        z += mean_prob[t] = phi_b(g) * phi_l(g, t) * model->phi_r(g, t)
                            / model->phi_p(g, t) * theta(s, t);
      for (size_t t = 0; t < T; ++t)
        mean_prob[t] /= z;
      auto icnts = sample_multinomial<size_t, IVector>(
          data.counts(g, s), begin(mean_prob), end(mean_prob), rng);
      for (size_t t = 0; t < T; ++t)
        cnts(t) = icnts(t);

      if (false and T > 1) {
        // perform several Metropolis-Hastings steps
        const size_t initial = 100;
        int n_iter = initial;
        while (n_iter--) {
          // modify
          size_t i = uniform_int_distribution<size_t>(0, T - 1)(rng);
          while (cnts[i] == 0)
            i = uniform_int_distribution<size_t>(0, T - 1)(rng);
          size_t j = uniform_int_distribution<size_t>(0, T - 1)(rng);
          while (i == j)
            j = uniform_int_distribution<size_t>(0, T - 1)(rng);
          size_t n = uniform_int_distribution<size_t>(1, cnts[i])(rng);

          // calculate score difference
          double l = log_posterior_difference(cnts, i, j, n);
          if (l > 0 or (isfinite(l)
                        and log(RandomDistribution::Uniform(rng))
                                    * parameters.temperature
                                <= l)) {
            // accept the candidate
            cnts[i] -= n;
            cnts[j] += n;
          }
        }
      }
    }
  }

  return cnts;
}

void Experiment::enforce_positive_parameters() {
  enforce_positive_and_warn("local field", field);
  enforce_positive_and_warn("spot", spot);
  enforce_positive_and_warn("phi_l", phi_l);
  enforce_positive_and_warn("phi_b", phi_b);
  weights.enforce_positive_parameters("weights");
}

Vector Experiment::marginalize_genes() const {
  Vector intensities(T, arma::fill::zeros);
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t t = 0; t < T; ++t) {
    double intensity = 0;
    for (size_t g = 0; g < G; ++g)
      intensity
          += phi_b(g) * phi_l(g, t) * model->phi_r(g, t) / model->phi_p(g, t);
    intensities[t] = intensity;
  }
  return intensities;
};

/** Calculate log posterior of theta with respect to the field */
Matrix Experiment::field_fitness_posterior(
    const Matrix &candidate_field) const {
  assert(candidate_field.n_rows == S);
  assert(candidate_field.n_cols == T);
  Matrix fit(S, T, arma::fill::zeros);
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t s = 0; s < S; ++s)
    for (size_t t = 0; t < T; ++t) {
      double prod = weights.prior.r(t) * candidate_field(s, t);
      fit(s, t) = (prod - 1) * log(theta(s, t))
                  - weights.prior.p(t) * theta(s, t) - lgamma(prod)
                  + log(weights.prior.p(t)) * prod;
    }
  return fit;
}

/** Calculate gradient of log posterior of theta with respect to the field */
Matrix Experiment::field_fitness_posterior_gradient(
    const Matrix &candidate_field) const {
  assert(candidate_field.n_rows == S);
  assert(candidate_field.n_cols == T);
  Matrix grad(S, T, arma::fill::zeros);
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t s = 0; s < S; ++s)
    for (size_t t = 0; t < T; ++t)
      grad(s, t) = weights.prior.r(t)
                   * (log(theta(s, t)) + log(weights.prior.p(t))
                      - digamma(weights.prior.r(t) * candidate_field(s, t)));
  return grad;
}

/** sample count decomposition */
Matrix Experiment::sample_contributions_gene(size_t g, RNG &rng) {
  LOG(debug) << "Sampling contributions for gene " << g;
  Matrix contributions(S, T, arma::fill::zeros);

  // reset contributions for those genes that are not dropped
  for (size_t t = 0; t < T; ++t)
    contributions_gene_type(g, t) = 0;

  for (size_t s = 0; s < S; ++s) {
    auto cnts = sample_contributions_gene_spot(g, s, rng);
    for (size_t t = 0; t < T; ++t) {
      contributions(s, t) = cnts[t];
      contributions_gene_type(g, t) += cnts[t];
    }
  }
  return contributions;
}

/** sample count decomposition */
Matrix Experiment::sample_contributions_spot(size_t s, RNG &rng) {
  LOG(debug) << "Sampling contributions for spot " << s;
  Matrix contributions(G, T, arma::fill::zeros);

  // reset contributions for those genes that are not dropped
  for (size_t t = 0; t < T; ++t)
    contributions_spot_type(s, t) = 0;

  for (size_t g = 0; g < G; ++g) {
    auto cnts = sample_contributions_gene_spot(g, s, rng);
    for (size_t t = 0; t < T; ++t) {
      contributions(g, t) = cnts[t];
      contributions_spot_type(s, t) += cnts[t];
    }
  }
  return contributions;
}

Vector Experiment::marginalize_spots() const {
  Vector intensities(T, arma::fill::zeros);
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t t = 0; t < T; ++t) {
    double intensity = 0;
    for (size_t s = 0; s < S; ++s)
      intensity += theta(s, t) * spot[s];
    intensities[t] = intensity;
  }
  return intensities;
}

Matrix Experiment::explained_gene_type() const {
  Vector theta_t = marginalize_spots();
  Matrix explained(G, T, arma::fill::zeros);
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t g = 0; g < G; ++g)
    for (size_t t = 0; t < T; ++t)
      explained(g, t)
          = model->phi_r(g, t) * phi_b(g) / model->phi_p(g, t) * theta_t(t);
  return explained;
}

Matrix Experiment::expected_gene_type() const {
  return phi_l % explained_gene_type();
}

// TODO never used - consider to remove
Vector Experiment::explained_gene() const {
  Vector theta_t = marginalize_spots();
  Vector explained(G, arma::fill::zeros);
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t g = 0; g < G; ++g)
    for (size_t t = 0; t < T; ++t)
      explained(g) += phi_l(g, t) * model->phi_r(g, t) * phi_b(g)
                      / model->phi_p(g, t) * theta_t(t);
  return explained;
};

Matrix Experiment::explained_spot_type() const {
  Matrix m = Matrix(S, T, arma::fill::ones);
  for (size_t t = 0; t < T; ++t) {
    Float x = 0;
    for (size_t g = 0; g < G; ++g)
      x += phi_l(g, t) * model->phi_r(g, t) * phi_b(g) / model->phi_p(g, t);
    for (size_t s = 0; s < S; ++s)
      m(s, t) *= x * spot(s);
  }
  return m;
}

Matrix Experiment::expected_spot_type() const {
  return weights.matrix % explained_spot_type();
}

vector<vector<size_t>> Experiment::active_factors(double threshold) const {
  auto w = expected_spot_type();
  vector<vector<size_t>> vs;
  for (size_t s = 0; s < S; ++s) {
    vector<size_t> v;
    for (size_t t = 0; t < T; ++t)
      if (w(s, t) > threshold)
        v.push_back(t);
    vs.push_back(v);
  }
  return vs;
}

ostream &operator<<(ostream &os, const Experiment &experiment) {
  os << "Experiment "
     << "G = " << experiment.G << " "
     << "S = " << experiment.S << " "
     << "T = " << experiment.T << endl;

  if (verbosity >= Verbosity::debug) {
    // TODO TODO fix features
    // print_matrix_head(os, experiment.baseline_feature.matrix, "Baseline Φ");
    // print_matrix_head(os, experiment.features.matrix, "Φ");
    print_matrix_head(os, experiment.weights.matrix, "Θ");
    /* TODO reactivate
    os << experiment.baseline_feature.prior;
    os << experiment.features.prior;
    os << experiment.weights.prior;

    print_vector_head(os, experiment.spot, "Spot scaling factors");
    */
  }

  return os;
}

Experiment operator*(const Experiment &a, const Experiment &b) {
  Experiment experiment = a;

  experiment.contributions_gene_type %= b.contributions_gene_type;
  experiment.contributions_spot_type %= b.contributions_spot_type;
  experiment.contributions_gene %= b.contributions_gene;
  experiment.contributions_spot %= b.contributions_spot;

  experiment.spot %= b.spot;

  experiment.phi_l %= b.phi_l;
  experiment.phi_b %= b.phi_b;
  experiment.weights.matrix %= b.weights.matrix;

  return experiment;
}

Experiment operator+(const Experiment &a, const Experiment &b) {
  Experiment experiment = a;

  experiment.contributions_gene_type += b.contributions_gene_type;
  experiment.contributions_spot_type += b.contributions_spot_type;
  experiment.contributions_gene += b.contributions_gene;
  experiment.contributions_spot += b.contributions_spot;

  experiment.spot += b.spot;

  experiment.phi_l += b.phi_l;
  experiment.phi_b += b.phi_b;
  experiment.weights.matrix += b.weights.matrix;

  return experiment;
}

Experiment operator-(const Experiment &a, const Experiment &b) {
  Experiment experiment = a;

  experiment.contributions_gene_type -= b.contributions_gene_type;
  experiment.contributions_spot_type -= b.contributions_spot_type;
  experiment.contributions_gene -= b.contributions_gene;
  experiment.contributions_spot -= b.contributions_spot;

  experiment.spot -= b.spot;

  experiment.phi_l -= b.phi_l;
  experiment.phi_b -= b.phi_b;
  experiment.weights.matrix -= b.weights.matrix;

  return experiment;
}

Experiment operator*(const Experiment &a, double x) {
  Experiment experiment = a;

  experiment.contributions_gene_type *= x;
  experiment.contributions_spot_type *= x;
  experiment.contributions_gene *= x;
  experiment.contributions_spot *= x;

  experiment.spot *= x;

  experiment.phi_l *= x;
  experiment.phi_b *= x;
  experiment.weights.matrix *= x;

  return experiment;
}

Experiment operator/(const Experiment &a, double x) {
  Experiment experiment = a;

  experiment.contributions_gene_type /= x;
  experiment.contributions_spot_type /= x;
  experiment.contributions_gene /= x;
  experiment.contributions_spot /= x;

  experiment.spot /= x;

  experiment.phi_l /= x;
  experiment.phi_b /= x;
  experiment.weights.matrix /= x;

  return experiment;
}

Experiment operator-(const Experiment &a, double x) {
  Experiment experiment = a;

  experiment.contributions_gene_type -= x;
  experiment.contributions_spot_type -= x;
  experiment.contributions_gene -= x;
  experiment.contributions_spot -= x;

  experiment.spot -= x;

  experiment.phi_l -= x;
  experiment.phi_b -= x;
  experiment.weights.matrix -= x;

  return experiment;
}
}
