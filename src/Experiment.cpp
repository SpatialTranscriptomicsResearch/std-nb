#include "Experiment.hpp"
#include "Model.hpp"
#include "gamma_func.hpp"
#include "hamiltonian_monte_carlo.hpp"

using namespace std;

namespace STD {

Experiment::Experiment(Model *model_, const Counts &counts_, size_t T_,
                       const Parameters &parameters_)
    : model(model_),
      G(counts_.num_genes()),
      S(counts_.num_samples()),
      T(T_),
      counts(counts_),
      coords(counts.parse_coords()),
      parameters(parameters_),
      lambda(Matrix::Ones(G, T)),
      beta(Matrix::Ones(G, 1)),
      theta(Matrix::Ones(S, T)),
      field(Matrix::Ones(S, T)),
      spot(Vector::Ones(S)),
      contributions_gene_type(Matrix::Zero(G, T)),
      contributions_spot_type(Matrix::Zero(S, T)),
      contributions_gene(rowSums<Vector>(*counts.matrix)),
      contributions_spot(colSums<Vector>(*counts.matrix)) {
  LOG(debug) << "Experiment G = " << G << " S = " << S << " T = " << T;
  /* TODO consider to initialize:
   * contributions_gene_type
   * contributions_spot_type
   */
  LOG(debug) << "Coords: " << coords;

  if (parameters.targeted(Target::spot)) {
    LOG(debug) << "Initializing spot scaling.";
    spot = contributions_spot;
    // divide by mean
    spot *= S / spot.sum();
  }

  // TODO initialize theta using parameters, model->mix_prior
  if (parameters.targeted(Target::theta))
    for (auto &x : theta)
      // TODO introduce parameter for constant
      x = exp(0.5 * std::normal_distribution<double>()(EntropySource::rng));
}

void Experiment::store(const string &prefix,
                       const vector<size_t> &order) const {
  auto factor_names = form_factor_names(T);
  auto &gene_names = counts.row_names;
  auto &spot_names = counts.col_names;

  string suffix = "";
  string extension = boost::filesystem::path(counts.path).extension().c_str();
  if (extension == ".gz" or extension == ".bz2")
    suffix = extension;
  boost::filesystem::create_symlink(
      boost::filesystem::canonical(counts.path),
      prefix + "counts" + FILENAME_ENDING + suffix);

#pragma omp parallel sections if (DO_PARALLEL)
  {
#pragma omp section
    write_matrix(lambda, prefix + "feature-lambda" + FILENAME_ENDING,
                 parameters.compression_mode, gene_names, factor_names, order);
#pragma omp section
    write_matrix(beta, prefix + "feature-beta" + FILENAME_ENDING,
                 parameters.compression_mode, gene_names, {1, "Baseline"}, {});
#pragma omp section
    write_matrix(theta, prefix + "theta" + FILENAME_ENDING,
                 parameters.compression_mode, spot_names, factor_names, order);
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
      for (size_t s = 0; s < S; ++s)
        f.row(s).array() /= phi_marginal.array();
      for (size_t t = 0; t < T; ++t)
        f.col(t).array() /= spot.array();
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
  lambda = parse_file<Matrix>(prefix + "feature-lambda" + FILENAME_ENDING,
                              read_matrix, "\t");
  beta = parse_file<Matrix>(prefix + "feature-beta" + FILENAME_ENDING,
                            read_matrix, "\t");

  theta = parse_file<Matrix>(prefix + "theta" + FILENAME_ENDING, read_matrix,
                             "\t");
  field = parse_file<Matrix>(prefix + "raw-field" + FILENAME_ENDING,
                             read_matrix, "\t");
  spot = parse_file<Vector>(prefix + "spot-scaling" + FILENAME_ENDING,
                            read_vector<Vector>, "\t");

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
}

Matrix Experiment::log_likelihood() const {
  Matrix l(G, S);
  const size_t K = min<size_t>(20, T);
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t g = 0; g < G; ++g) {
    Vector ps = model->negodds_rho.row(g);
    for (size_t t = 0; t < T; ++t)
      // NOTE conv neg bin has opposite interpretation of p
      ps[t] = 1 - neg_odds_to_prob(ps[t]);
    Vector rs(T);
    for (size_t s = 0; s < S; ++s) {
      for (size_t t = 0; t < T; ++t)
        rs[t] = model->gamma(g, t) * lambda(g, t) * beta(g) * theta(s, t)
                * spot(s);
      double x = convolved_negative_binomial(counts(g, s), K, rs, ps);
      LOG(debug) << "Computing log likelihood for g/s = " << g << "/" << s
                 << " counts = " << counts(g, s) << " l = " << x;
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

  Vector tmp(T);
  for (size_t t = 0; t < T; ++t)
    // TODO use digamma_diff
    tmp(t) = log(p(t)) + digamma(x(t) + r(t)) - digamma(x(t) + 1);

  if (noisy)
    LOG(trace) << "tmp = " << tmp;

  double z = 0;
  for (size_t t = 0; t < T; ++t)
    z += x(t) / count * tmp(t);

  Vector grad(T);
  for (size_t t = 0; t < T; ++t)
    grad(t) = -x(t) * (tmp(t) - z);

  if (noisy)
    LOG(debug) << "grad = " << grad;
  return grad;
}

/** sample count decomposition */
Vector Experiment::sample_contributions_gene_spot(size_t g, size_t s,
                                                  RNG &rng) const {
  Vector cnts = Vector::Zero(T);

  const auto count = counts(g, s);

  if (count == 0)
    return cnts;

  if (T == 1) {
    cnts[0] = count;
    return cnts;
  }

  switch (parameters.sample_method) {
    case Sampling::Method::Mean: {
      double z = 0;
      for (size_t t = 0; t < T; ++t)
        z += cnts[t] = beta(g) * lambda(g, t) * model->gamma(g, t) * theta(s, t)
                       / model->negodds_rho(g, t);
      for (size_t t = 0; t < T; ++t)
        cnts[t] *= counts(g, s) / z;
      return cnts;
    } break;
    case Sampling::Method::Multinomial: {
      double z = 0;
      for (size_t t = 0; t < T; ++t)
        z += cnts[t] = beta(g) * lambda(g, t) * model->gamma(g, t) * theta(s, t)
                       / model->negodds_rho(g, t);
      for (size_t t = 0; t < T; ++t)
        cnts[t] /= z;
      auto icnts
          = sample_multinomial(counts(g, s), begin(cnts), end(cnts), rng);
      for (size_t t = 0; t < T; ++t)
        cnts[t] = icnts[t];
      return cnts;
    } break;
    case Sampling::Method::MH:
      throw(runtime_error("Sampling method not implemented: MH."));
      break;
    case Sampling::Method::HMC:
      throw(runtime_error("Sampling method not implemented: HMC."));
      break;
    case Sampling::Method::RPROP:
      throw(runtime_error("Sampling method not implemented: RPROP."));
      break;
  }
  if (false) {
    if (parameters.contributions_map) {
      Vector r(T);
      for (size_t t = 0; t < T; ++t)
        r[t] = model->gamma(g, t) * lambda(g, t) * beta(g) * theta(s, t)
               * spot(s);

      Vector p(T);
      for (size_t t = 0; t < T; ++t)
        p[t] = neg_odds_to_prob(model->negodds_rho(g, t));

      if (noisy) {
        for (size_t t = 0; t < T; ++t)
          LOG(debug) << "r = " << r[t];
        for (size_t t = 0; t < T; ++t)
          LOG(debug) << "p = " << p[t];
      }

      if (true) {
        for (size_t t = 0; t < T; ++t)
          cnts[t] = log(r[t] / model->negodds_rho(g, t));

        Vector mean(T);
        const size_t max_iter = (noisy ? 10 : 1);
        for (size_t iter = 0; iter < max_iter; ++iter) {
          mean = count * gibbs(cnts);
          if (noisy)
            LOG(verbose) << "cnts 0: " << (count * gibbs(cnts)).transpose();
          for (size_t i = 0; i < parameters.hmc_N; ++i) {
            if (noisy)
              LOG(debug) << "cnts " << i + 1 << ": "
                         << (count * gibbs(cnts).transpose());
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
            LOG(verbose) << "cnts X: " << (count * gibbs(cnts)).transpose();
            LOG(verbose) << "cnts m: " << mean.transpose();
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
          // case it should work better for more than two factors
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

        const double r_i = model->gamma(g, i) * lambda(g, i);
        const double no_i = model->negodds_rho(g, i);
        const double prod_i = r_i * theta(s, i) * spot(s);

        const double r_j = model->gamma(g, j) * lambda(g, j);
        const double no_j = model->negodds_rho(g, j);
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
      vector<double> expected_prob(T, 0);
      double z = 0;
      for (size_t t = 0; t < T; ++t)
        z += expected_prob[t] = beta(g) * lambda(g, t) * model->gamma(g, t)
                                / model->negodds_rho(g, t) * theta(s, t);
      for (size_t t = 0; t < T; ++t)
        expected_prob[t] /= z;
      auto icnts = sample_multinomial(counts(g, s), begin(expected_prob),
                                      end(expected_prob), rng);
      for (size_t t = 0; t < T; ++t)
        // cnts(t) = icnts(t);
        // TODO deactivate or change back
        cnts[t] = counts(g, s) * expected_prob[t];

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
  enforce_positive_and_warn("lambda", lambda);
  enforce_positive_and_warn("beta", beta);
  enforce_positive_and_warn("theta", theta);
  enforce_positive_and_warn("local field", field);
  enforce_positive_and_warn("spot", spot);
}

/** Calculate log posterior of theta with respect to the field */
Matrix Experiment::field_fitness_posterior() const {
  Matrix fit = Matrix::Zero(S, T);
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t s = 0; s < S; ++s)
    for (size_t t = 0; t < T; ++t) {
      double prod = model->mix_prior.r(t) * field(s, t);
      fit(s, t) = (prod - 1) * log(theta(s, t))
                  - model->mix_prior.p(t) * theta(s, t) - lgamma(prod)
                  + log(model->mix_prior.p(t)) * prod;
    }
  return fit;
}

/** Calculate gradient of log posterior of theta with respect to the field */
Matrix Experiment::field_fitness_posterior_gradient() const {
  Matrix grad = Matrix::Zero(S, T);
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t s = 0; s < S; ++s)
    for (size_t t = 0; t < T; ++t)
      grad(s, t) = model->mix_prior.r(t)
                   * (log(theta(s, t)) + log(model->mix_prior.p(t))
                      - digamma(model->mix_prior.r(t) * field(s, t)));
  return grad;
}

/** sample count decomposition */
Matrix Experiment::sample_contributions_gene(size_t g, RNG &rng) {
  LOG(debug) << "Sampling contributions for gene " << g;
  Matrix contributions = Matrix::Zero(S, T);

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
  Matrix contributions = Matrix::Zero(G, T);

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

Vector Experiment::marginalize_genes() const {
  Vector intensities = Vector::Zero(T);
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t t = 0; t < T; ++t) {
    double intensity = 0;
    for (size_t g = 0; g < G; ++g)
      intensity += beta(g) * lambda(g, t) * model->gamma(g, t)
                   / model->negodds_rho(g, t);
    intensities[t] = intensity;
  }
  return intensities;
};

Vector Experiment::marginalize_spots() const {
  Vector intensities(T);
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t t = 0; t < T; ++t) {
    double intensity = 0;
    for (size_t s = 0; s < S; ++s)
      intensity += theta(s, t) * spot(s);
    intensities(t) = intensity;
  }
  return intensities;
}

Matrix Experiment::expected_gene_type() const {
  Vector marginal = marginalize_spots();
  Matrix expected = Matrix::Zero(G, T);
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t g = 0; g < G; ++g)
    for (size_t t = 0; t < T; ++t)
      expected(g, t) = model->gamma(g, t) * lambda(g, t) * beta(g)
                       / model->negodds_rho(g, t) * marginal(t);
  return expected;
}

Matrix Experiment::expected_spot_type() const {
  Matrix m(S, T);
  for (size_t t = 0; t < T; ++t) {
    Float x = 0;
#pragma omp parallel for if (DO_PARALLEL)
    for (size_t g = 0; g < G; ++g)
      x += model->gamma(g, t) * lambda(g, t) * beta(g)
           / model->negodds_rho(g, t);
#pragma omp parallel for if (DO_PARALLEL)
    for (size_t s = 0; s < S; ++s)
      m(s, t) = x * theta(s, t) * spot(s);
  }
  return m;
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

size_t Experiment::size() const {
  size_t s = 0;
  if (parameters.targeted(Target::lambda))
    s += lambda.size();
  if (parameters.targeted(Target::beta))
    s += beta.size();
  if (parameters.targeted(Target::theta))
    s += theta.size();
  if (parameters.targeted(Target::field))
    s += field.size();
  if (parameters.targeted(Target::spot))
    s += spot.size();
  return s;
}

void Experiment::set_zero() {
  lambda.setZero();
  beta.setZero();
  theta.setZero();
  field.setZero();
  spot.setZero();
}

Vector Experiment::vectorize() const {
  Vector v(size());
  auto iter = begin(v);
  if (parameters.targeted(Target::lambda))
    for (auto &x : lambda)
      *iter++ = x;
  if (parameters.targeted(Target::beta))
    for (auto &x : beta)
      *iter++ = x;
  if (parameters.targeted(Target::theta))
    for (auto &x : theta)
      *iter++ = x;
  if (parameters.targeted(Target::field))
    for (auto &x : field)
      *iter++ = x;
  if (parameters.targeted(Target::spot))
    for (auto &x : spot)
      *iter++ = x;

  assert(iter == end(v));

  return v;
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
    print_matrix_head(os, experiment.theta, "Θ");
    /* TODO reactivate
    os << experiment.baseline_feature.prior;
    os << experiment.features.prior;

    print_vector_head(os, experiment.spot, "Spot scaling factors");
    */
  }

  return os;
}

Experiment operator*(const Experiment &a, const Experiment &b) {
  Experiment experiment = a;

  experiment.contributions_gene_type.array()
      *= b.contributions_gene_type.array();
  experiment.contributions_spot_type.array()
      *= b.contributions_spot_type.array();
  experiment.contributions_gene.array() *= b.contributions_gene.array();
  experiment.contributions_spot.array() *= b.contributions_spot.array();

  experiment.spot.array() *= b.spot.array();

  experiment.lambda.array() *= b.lambda.array();
  experiment.beta.array() *= b.beta.array();
  experiment.theta.array() *= b.theta.array();

  return experiment;
}

Experiment operator+(const Experiment &a, const Experiment &b) {
  Experiment experiment = a;

  experiment.contributions_gene_type += b.contributions_gene_type;
  experiment.contributions_spot_type += b.contributions_spot_type;
  experiment.contributions_gene += b.contributions_gene;
  experiment.contributions_spot += b.contributions_spot;

  experiment.spot += b.spot;

  experiment.lambda += b.lambda;
  experiment.beta += b.beta;
  experiment.theta += b.theta;

  return experiment;
}

Experiment operator-(const Experiment &a, const Experiment &b) {
  Experiment experiment = a;

  experiment.contributions_gene_type -= b.contributions_gene_type;
  experiment.contributions_spot_type -= b.contributions_spot_type;
  experiment.contributions_gene -= b.contributions_gene;
  experiment.contributions_spot -= b.contributions_spot;

  experiment.spot -= b.spot;

  experiment.lambda -= b.lambda;
  experiment.beta -= b.beta;
  experiment.theta -= b.theta;

  return experiment;
}

Experiment operator*(const Experiment &a, double x) {
  Experiment experiment = a;

  experiment.contributions_gene_type *= x;
  experiment.contributions_spot_type *= x;
  experiment.contributions_gene *= x;
  experiment.contributions_spot *= x;

  experiment.spot *= x;

  experiment.lambda *= x;
  experiment.beta *= x;
  experiment.theta *= x;

  return experiment;
}

Experiment operator/(const Experiment &a, double x) {
  Experiment experiment = a;

  experiment.contributions_gene_type /= x;
  experiment.contributions_spot_type /= x;
  experiment.contributions_gene /= x;
  experiment.contributions_spot /= x;

  experiment.spot /= x;

  experiment.lambda /= x;
  experiment.beta /= x;
  experiment.theta /= x;

  return experiment;
}

Experiment operator-(const Experiment &a, double x) {
  Experiment experiment = a;

  experiment.contributions_gene_type.array() -= x;
  experiment.contributions_spot_type.array() -= x;
  experiment.contributions_gene.array() -= x;
  experiment.contributions_spot.array() -= x;

  experiment.spot.array() -= x;

  experiment.lambda.array() -= x;
  experiment.beta.array() -= x;
  experiment.theta.array() -= x;

  return experiment;
}
}
