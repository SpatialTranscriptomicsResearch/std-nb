#include "Experiment.hpp"
#include <LBFGS.h>
#include "Model.hpp"
#include "aux.hpp"
#include "gamma_func.hpp"
#include "hamiltonian_monte_carlo.hpp"
#include "io.hpp"
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
    write_matrix(expected_spot_type(),
                 prefix + "expected-mix" + FILENAME_ENDING,
                 parameters.compression_mode, spot_names, factor_names, order);
#pragma omp section
    write_matrix(expected_gene_type(),
                 prefix + "expected-features" + FILENAME_ENDING,
                 parameters.compression_mode, gene_names, factor_names, order);
#pragma omp section
    write_matrix(compute_gene_type_table(rate_coeff_idxs),
                 prefix + "rate_gene_type" + FILENAME_ENDING,
                 parameters.compression_mode, gene_names, factor_names, order);
#pragma omp section
    write_matrix(compute_gene_type_table(odds_coeff_idxs),
                 prefix + "odds_gene_type" + FILENAME_ENDING,
                 parameters.compression_mode, gene_names, factor_names, order);
#pragma omp section
    write_matrix(compute_spot_type_table(rate_coeff_idxs),
                 prefix + "rate_spot_type" + FILENAME_ENDING,
                 parameters.compression_mode, spot_names, factor_names, order);
#pragma omp section
    write_matrix(compute_spot_type_table(odds_coeff_idxs),
                 prefix + "odds_spot_type" + FILENAME_ENDING,
                 parameters.compression_mode, spot_names, factor_names, order);
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

/** sample count decomposition */
Vector Experiment::sample_contributions_gene_spot(
    size_t g, size_t s, const Matrix &rate_gt, const Matrix &rate_st,
    const Matrix &odds_gt, const Matrix &odds_st, RNG &rng) const {
  Vector cnts = Vector::Zero(T);

  const auto count = counts(g, s);

  if (count == 0)
    return cnts;

  if (T == 1) {
    cnts[0] = count;
    return cnts;
  }

  Vector rate(T);
  for (size_t t = 0; t < T; ++t)
    rate(t) = rate_gt(g, t) * rate_st(s, t);

  Vector odds(T);
  for (size_t t = 0; t < T; ++t)
    odds(t) = odds_gt(g, t) * odds_st(s, t);

  Vector proportions(T);
  {
    double z = 0;
    for (size_t t = 0; t < T; ++t)
      z += proportions(t) = rate(t) * odds(t);
    for (size_t t = 0; t < T; ++t)
      proportions(t) /= z;
  }

  switch (parameters.sample_method) {
    case Sampling::Method::Mean:
      return proportions * count;
    case Sampling::Method::Multinomial:
      return sample_multinomial<Vector>(count, begin(proportions),
                                        end(proportions), rng);
    default:
      break;
  }

  auto unlog = [&](const Vector &log_k) {
    double max_log_k = -std::numeric_limits<double>::infinity();
    for (size_t t = 0; t < T; ++t)
      if (log_k(t) > max_log_k)
        max_log_k = log_k(t);

    double z = 0;
    Vector k(T);
    for (size_t t = 0; t < T; ++t)
      z += k(t) = exp(log_k(t) - max_log_k);
    for (size_t t = 0; t < T; ++t)
      k(t) *= count / z;
    return k;
  };

  Vector log_p(T);
  for (size_t t = 0; t < T; ++t)
    log_p(t) = odds_to_log_prob(odds(t));

  // compute the count-dependent likelihood contribution
  auto eval = [&](const Vector &k) {
    double score = 0;
    for (size_t t = 0; t < T; ++t)
      score += lgamma(rate(t) + k(t)) - lgamma(k(t) + 1) + k(t) * log_p(t);
    // - lgamma(rate(t))
    return score;
  };

  auto compute_gradient = [&](const Vector &log_k, Vector &grad) {
    grad = Vector(T);
    Vector k = unlog(log_k);
    double sum = 0;
    for (size_t t = 0; t < T; ++t)
      sum += grad(t) = k(t) * (digamma_diff_1p(k(t), rate(t)) + log_p(t));
    for (size_t t = 0; t < T; ++t)
      grad(t) = k(t) / count * (grad(t) - sum);
  };

  auto fnc = [&](const Vector &log_k, Vector &grad) {
    compute_gradient(log_k, grad);
    double score = -eval(unlog(log_k));
    LOG(verbose) << "count = " << count << " fnc = " << score;
    return score;
  };

  switch (parameters.sample_method) {
    case Sampling::Method::Trial: {
      double best_score = -std::numeric_limits<double>::infinity();
      Vector best_cnts = Vector::Zero(T);
      for (size_t i = 0; i < parameters.sample_iterations; ++i) {
        Vector trial_cnts = sample_multinomial<Vector>(
            count, begin(proportions), end(proportions), rng);
        double trial_score = eval(trial_cnts);

        if (trial_score > best_score) {
          best_score = trial_score;
          best_cnts = trial_cnts;
        }
      }
      return best_cnts;
    }
    case Sampling::Method::TrialMean: {
      double total_score = 0;
      Vector mean_cnts = Vector::Zero(T);
      for (size_t i = 0; i < parameters.sample_iterations; ++i) {
        Vector trial_cnts = sample_multinomial<Vector>(
            count, begin(proportions), end(proportions), rng);
        double score = exp(eval(trial_cnts));
        mean_cnts += score * trial_cnts;
        total_score += score;
      }
      return mean_cnts / total_score;
    }
    case Sampling::Method::MH:
      throw(runtime_error("Sampling method not implemented: MH."));
      break;
    case Sampling::Method::HMC:
      throw(runtime_error("Sampling method not implemented: HMC."));
      break;
    case Sampling::Method::RPROP: {
      cnts = (proportions * count).array().log();

      Vector grad(cnts.size());
      Vector prev_sign(Vector::Zero(cnts.size()));
      Vector rates(cnts.size());
      rates.fill(parameters.grad_alpha);
      for (size_t iter = 0; iter < parameters.sample_iterations; ++iter) {
        compute_gradient(cnts, grad);
        rprop_update(grad, prev_sign, rates, cnts, parameters.rprop);
      }
      return unlog(cnts);
    } break;
    case Sampling::Method::lBFGS: {
      cnts = (proportions * count).array().log();

      using namespace LBFGSpp;
      LBFGSParam<double> param;
      param.epsilon = parameters.lbfgs_epsilon;
      // TODO make into separate CLI configurable parameter
      param.max_iterations = parameters.lbfgs_iter;
      // Create solver and function object
      LBFGSSolver<double> solver(param);

      double fx = std::numeric_limits<double>::infinity();
      int niter = solver.minimize(fnc, cnts, fx);

      LOG(verbose) << "lBFGS performed " << niter
                   << " iterations and reached score " << fx;
      return unlog(cnts);
    } break;
    default:
      break;
  }

  throw std::runtime_error("Error: this point should not be reached.");
  return cnts;
}

// NOTE: scalar covariates are multiplied into this table
Matrix Experiment::compute_gene_type_table(const vector<size_t> &idxs) const {
  Matrix gt = Matrix::Ones(G, T);

  for (auto &idx : idxs)
    if (not model->coeffs[idx].spot_dependent())
      for (size_t g = 0; g < G; ++g)
        for (size_t t = 0; t < T; ++t)
          gt(g, t) *= model->coeffs[idx].get(g, t, 0);

  return gt;
}

// NOTE: scalar covariates are NOT multiplied into this table
Matrix Experiment::compute_spot_type_table(const vector<size_t> &idxs) const {
  Matrix st = Matrix::Ones(S, T);

  for (auto &idx : idxs)
    if (model->coeffs[idx].spot_dependent())
      for (size_t s = 0; s < S; ++s)
        for (size_t t = 0; t < T; ++t)
          st(s, t) *= model->coeffs[idx].get(0, t, s);

  return st;
}

/* TODO cov var
Vector Experiment::marginalize_genes() const {

  Matrix gt = compute_gene_type_table(Coefficient::Variable::rate).array()
              / compute_gene_type_table(Coefficient::Variable::odds).array();
  Vector gt_cs = colSums<Vector>(gt);

  Vector intensities = Vector::Zero(T);
  Matrix gt = compute_gene_type_table(Coefficient::Variable::rate).array()
              / compute_gene_type_table(Coefficient::Variable::odds).array();
  // Matrix st = compute_spot_type_table(Coefficient::Variable::rate).array()
  //             / compute_spot_type_table(Coefficient::Variable::odds).array();
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
  Matrix rate_gt = compute_gene_type_table(rate_coeff_idxs);
  Matrix odds_gt = compute_gene_type_table(odds_coeff_idxs);
  Matrix mean_gt = rate_gt.array() * odds_gt.array();

  Matrix rate_st = compute_spot_type_table(rate_coeff_idxs);
  Matrix odds_st = compute_spot_type_table(odds_coeff_idxs);
  Matrix mean_st = rate_st.array() * odds_st.array();
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
  Matrix rate_gt = compute_gene_type_table(rate_coeff_idxs);
  Matrix rate_st = compute_spot_type_table(rate_coeff_idxs);
  Matrix odds_gt = compute_gene_type_table(odds_coeff_idxs);
  Matrix odds_st = compute_spot_type_table(odds_coeff_idxs);
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t g = 0; g < G; ++g)
    for (size_t s = 0; s < S; ++s) {
      double x = 0;
      for (size_t t = 0; t < T; ++t) {
        double odds = odds_gt(g, t) * odds_st(s, t);
        x += rate_gt(g, t) * rate_st(s, t) * odds / neg_odds_to_prob(odds);
      }
      var(g, s) = x;
    }
  return var;
}

Matrix Experiment::expected_gene_type() const {
  Matrix st = compute_spot_type_table(rate_coeff_idxs).array()
              * compute_spot_type_table(odds_coeff_idxs).array();
  Vector st_cs = colSums<Vector>(st);

  Matrix expected = compute_gene_type_table(rate_coeff_idxs).array()
                    * compute_gene_type_table(odds_coeff_idxs).array();

  for (size_t t = 0; t < T; ++t)
    expected.col(t) *= st_cs[t];
  return expected;
}

Matrix Experiment::expected_spot_type() const {
  Matrix gt = compute_gene_type_table(rate_coeff_idxs).array()
              * compute_gene_type_table(odds_coeff_idxs).array();
  Vector gt_cs = colSums<Vector>(gt);

  Matrix expected = compute_spot_type_table(rate_coeff_idxs).array()
                    * compute_spot_type_table(odds_coeff_idxs).array();

  for (size_t t = 0; t < T; ++t)
    expected.col(t) *= gt_cs[t];
  return expected;
}

ostream &operator<<(ostream &os, const Experiment &experiment) {
  os << "Experiment "
     << "G = " << experiment.G << " "
     << "S = " << experiment.S << " "
     << "T = " << experiment.T << endl;
  return os;
}

Experiment operator+(const Experiment &a, const Experiment &b) {
  Experiment experiment = a;

  experiment.contributions_gene_type += b.contributions_gene_type;
  experiment.contributions_spot_type += b.contributions_spot_type;
  experiment.contributions_gene += b.contributions_gene;
  experiment.contributions_spot += b.contributions_spot;

  return experiment;
}
}
