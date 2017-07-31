#include "Experiment.hpp"
#include "Model.hpp"
#include "gamma_func.hpp"
#include "hamiltonian_monte_carlo.hpp"
#include "rprop.hpp"

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
      theta(Matrix::Ones(S, T)),
      field(Matrix::Ones(S, T)),
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
    write_matrix(theta, prefix + "theta" + FILENAME_ENDING,
                 parameters.compression_mode, spot_names, factor_names, order);
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
    /* TODO cov var
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
    */
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
  theta = parse_file<Matrix>(prefix + "theta" + FILENAME_ENDING, read_matrix,
                             "\t");
  field = parse_file<Matrix>(prefix + "raw-field" + FILENAME_ENDING,
                             read_matrix, "\t");

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

/* TODO covariates reactivate likelihood
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
*/

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

// NOTE: scalar covariates are multiplied into this table
Matrix Experiment::compute_gene_type_table(
    Coefficient::Variable variable) const {
  Matrix gt = Matrix::Ones(G, T);

  // TODO cov make more efficient
  for (auto &idx : coeff_idxs)
    if (model->coeffs[idx].variable == variable
        and not model->coeffs[idx].spot_dependent())
      for (size_t g = 0; g < G; ++g)
        for (size_t t = 0; t < T; ++t)
          gt(g, t) *= model->coeffs[idx].get(g, t, 0);

  return gt;
}

// NOTE: scalar covariates are NOT multiplied into this table
Matrix Experiment::compute_spot_type_table(
    Coefficient::Variable variable) const {
  Matrix st;
  if (variable == Coefficient::Variable::rate)
    st = theta;
  else
    st = Matrix::Ones(S, T);

  // TODO cov make more efficient
  for (auto &idx : coeff_idxs)
    if (model->coeffs[idx].variable == variable
        and model->coeffs[idx].spot_dependent())
      for (size_t s = 0; s < S; ++s)
        for (size_t t = 0; t < T; ++t)
          st(s, t) *= model->coeffs[idx].get(0, t, s);

  return st;
}

/** sample count decomposition */
Vector Experiment::sample_contributions_gene_spot(
    size_t g, size_t s, const Matrix &rate_gt, const Matrix &rate_st,
    const Matrix &variance_gt, const Matrix &variance_st, RNG &rng) const {
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
        z += cnts[t] = rate_gt(g, t) * rate_st(s, t)
                       / (variance_gt(g, t) * variance_st(s, t));
      for (size_t t = 0; t < T; ++t)
        cnts[t] *= count / z;
      return cnts;
    } break;
    case Sampling::Method::Multinomial: {
      double z = 0;
      for (size_t t = 0; t < T; ++t)
        z += cnts[t] = rate_gt(g, t) * rate_st(s, t)
                       / (variance_gt(g, t) * variance_st(s, t));
      for (size_t t = 0; t < T; ++t)
        cnts[t] /= z;
      auto icnts = sample_multinomial(count, begin(cnts), end(cnts), rng);
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
    case Sampling::Method::RPROP: {
      throw(runtime_error("Sampling method not quite implemented: RPROP."));
      /*
      Vector log_rho(T);
      for (size_t t = 0; t < T; ++t)
        log_rho[t] = neg_odds_to_log_prob(model->negodds_rho(g, t));

      Vector r(T);
      double z = 0;
      for (size_t t = 0; t < T; ++t) {
        r[t] = rate_gt(g, t) * st(s, t);
        z += cnts[t] = r[t] / model->negodds_rho(g, t);
      }
      for (size_t t = 0; t < T; ++t)
        cnts[t] *= count / z;

      Vector k(T);
      auto fnc = [&](const Vector &log_k, Vector &grad) {
        // LOG(debug) << "inter log = " << log_k.transpose();
        double score = 0;
        double Z = 0;
        for (size_t t = 0; t < T; ++t)
          Z += k[t] = exp(log_k(t));
        for (size_t t = 0; t < T; ++t)
          k[t] *= count / Z;
        double sum = 0;
        for (size_t t = 0; t < T; ++t)
          sum += grad[t] = k[t] * (digamma_diff(k[t], r[t]) + log_rho[t]);
        // LOG(debug) << "inter exp = " << k.transpose();
        for (size_t t = 0; t < T; ++t)
          grad[t] -= k[t] / count * sum;
        // LOG(debug) << "grad = " << grad.transpose();
        return score;
      };
      Vector grad(cnts.size());
      Vector prev_sign(Vector::Zero(cnts.size()));
      Vector rates(cnts.size());
      rates.fill(parameters.grad_alpha);
      const size_t sample_iterations
          = 20;  // TODO make into CLI configurable parameter
      double fx = 0;
      // LOG(debug) << "total = " << count;
      // LOG(debug) << "start = " << cnts.transpose();
      for (size_t t = 0; t < T; ++t)
        cnts[t] = log(cnts[t]);
      // LOG(debug) << "start = " << cnts.transpose();
      for (size_t iter = 0; iter < sample_iterations; ++iter) {
        fx = fnc(cnts, grad);
        rprop_update(grad, prev_sign, rates, cnts, parameters.rprop);
      }
      z = 0;
      for (size_t t = 0; t < T; ++t)
        z += cnts[t] = exp(cnts[t]);
      for (size_t t = 0; t < T; ++t)
        cnts[t] *= count / z;

      // LOG(debug) << "final = " << cnts.transpose();
      */
    } break;
  }

  /*
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
        // if(prod + v[t] == 0)
        //   return -numeric_limits<double>::infinity();

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
  */

  return cnts;
}

void Experiment::enforce_positive_parameters() {
  enforce_positive_and_warn("theta", theta);
  enforce_positive_and_warn("local field", field);
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

/* TODO cov var
Vector Experiment::marginalize_genes() const {

  Matrix gt = compute_gene_type_table(Coefficient::Variable::rate).array()
              / compute_gene_type_table(Coefficient::Variable::variance).array();
  Vector gt_cs = colSums<Vector>(gt);

  Vector intensities = Vector::Zero(T);
  Matrix gt = compute_gene_type_table(Coefficient::Variable::rate).array()
              / compute_gene_type_table(Coefficient::Variable::variance).array();
  // Matrix st = compute_spot_type_table(Coefficient::Variable::rate).array()
  //             / compute_spot_type_table(Coefficient::Variable::variance).array();
  for (size_t t = 0; t < T; ++t) {
    double intensity = 0;
#pragma omp parallel for reduction(+ : intensity) if (DO_PARALLEL)
    for (size_t g = 0; g < G; ++g)
        intensity += gt(g, t);
    // TODO cov var; the following seemed originally like the correct thing to do
    //   for (size_t s = 0; s < S; ++s)
    //     intensity += gt(g, t) * st(s, t);
    intensities[t] = intensity;
  }
  return intensities;
};
*/

Matrix Experiment::expectation() const {
  Matrix mean(G, S);
  Matrix rate_gt = compute_gene_type_table(Coefficient::Variable::rate);
  Matrix variance_gt = compute_gene_type_table(Coefficient::Variable::variance);
  Matrix mean_gt = rate_gt.array() / variance_gt.array();

  Matrix rate_st = compute_spot_type_table(Coefficient::Variable::rate);
  Matrix variance_st = compute_spot_type_table(Coefficient::Variable::variance);
  Matrix mean_st = rate_st.array() / variance_st.array();
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t g = 0; g < G; ++g)
    for (size_t s = 0; s < S; ++s) {
      double x = 0;
      for (size_t t = 0; t < T; ++t)
        x += mean_gt(g, t) * mean_st(s, t);
      mean(g, s) = x;
    }
  return mean;
}

Matrix Experiment::variance() const {
  Matrix var(G, S);
  Matrix rate_gt = compute_gene_type_table(Coefficient::Variable::rate);
  Matrix rate_st = compute_spot_type_table(Coefficient::Variable::rate);
  Matrix variance_gt = compute_gene_type_table(Coefficient::Variable::variance);
  Matrix variance_st = compute_spot_type_table(Coefficient::Variable::variance);
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t g = 0; g < G; ++g)
    for (size_t s = 0; s < S; ++s) {
      double x = 0;
      for (size_t t = 0; t < T; ++t) {
        double no = variance_gt(g, t) * variance_st(s, t);
        x += rate_gt(g, t) * rate_st(s, t) / no / odds_to_prob(no);
      }
      var(g, s) = x;
    }
  return var;
}

Matrix Experiment::expected_gene_type() const {
  Matrix st
      = compute_spot_type_table(Coefficient::Variable::rate).array()
        / compute_spot_type_table(Coefficient::Variable::variance).array();
  Vector st_cs = colSums<Vector>(st);

  Matrix expected
      = compute_gene_type_table(Coefficient::Variable::rate).array()
        / compute_gene_type_table(Coefficient::Variable::variance).array();

  for (size_t t = 0; t < T; ++t)
    expected.col(t) *= st_cs[t];
  return expected;
}

Matrix Experiment::expected_spot_type() const {
  Matrix gt
      = compute_gene_type_table(Coefficient::Variable::rate).array()
        / compute_gene_type_table(Coefficient::Variable::variance).array();
  Vector gt_cs = colSums<Vector>(gt);

  Matrix expected
      = compute_spot_type_table(Coefficient::Variable::rate).array()
        / compute_spot_type_table(Coefficient::Variable::variance).array();

  for (size_t t = 0; t < T; ++t)
    expected.col(t) *= gt_cs[t];
  return expected;
}

size_t Experiment::size() const {
  size_t s = 0;
  if (parameters.targeted(Target::theta))
    s += theta.size();
  if (parameters.targeted(Target::field))
    s += field.size();
  return s;
}

void Experiment::setZero() {
  theta.setZero();
  field.setZero();
}

Vector Experiment::vectorize() const {
  Vector v(size());
  auto iter = begin(v);
  if (parameters.targeted(Target::theta))
    for (auto &x : theta)
      *iter++ = x;
  if (parameters.targeted(Target::field))
    for (auto &x : field)
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

Experiment operator+(const Experiment &a, const Experiment &b) {
  Experiment experiment = a;

  experiment.contributions_gene_type += b.contributions_gene_type;
  experiment.contributions_spot_type += b.contributions_spot_type;
  experiment.contributions_gene += b.contributions_gene;
  experiment.contributions_spot += b.contributions_spot;

  experiment.theta += b.theta;

  return experiment;
}
}
