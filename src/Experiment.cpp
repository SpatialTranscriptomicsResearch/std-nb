#include "Experiment.hpp"
#include "Model.hpp"
#include "aux.hpp"
#include "gamma_func.hpp"
#include "hamiltonian_monte_carlo.hpp"
#include "io.hpp"
#include "metropolis_hastings.hpp"
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
      scale_ratio(counts.matrix->sum() / S),
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
  LOG(trace) << "Coords: " << coords;
}

void Experiment::ensure_dimensions() const {
  for (auto &idxs : {rate_coeff_idxs, odds_coeff_idxs})
    for (auto &coeff_idx : idxs) {
      int nrow = 0;
      int ncol = 0;
      Coefficient &coeff = *model->coeffs[coeff_idx];
      switch (coeff.kind) {
        case Coefficient::Kind::scalar:
          nrow = ncol = 1;
          break;
        case Coefficient::Kind::gene:
          nrow = G;
          ncol = 1;
          break;
        case Coefficient::Kind::spot:
          nrow = S;
          ncol = 1;
          break;
        case Coefficient::Kind::type:
          nrow = T;
          ncol = 1;
          break;
        case Coefficient::Kind::gene_type:
          nrow = G;
          ncol = T;
          break;
        case Coefficient::Kind::spot_type:
          nrow = S;
          ncol = T;
          break;
      }
      if (coeff.values.rows() != nrow or coeff.values.cols() != ncol)
        throw std::runtime_error("Error: mismatched dimension on coefficient "
                                 + to_string(coeff_idx) + ": " + to_string(nrow)
                                 + "x" + to_string(ncol) + " vs "
                                 + coeff.to_string());
    }
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

/** sample count decomposition */
Vector Experiment::sample_contributions_gene_spot(size_t g, size_t s,
                                                  const Vector &rate,
                                                  const Vector &odds,
                                                  RNG &rng) const {
  Vector cnts = Vector::Zero(T);

  const auto actual_count = counts(g, s);
  size_t count = actual_count;
  if (parameters.adjust_seq_depth)
    count = std::binomial_distribution<size_t>(count, 1 / scale_ratio)(rng);
  if (parameters.downsample < 1)
    count
        = std::binomial_distribution<size_t>(count, parameters.downsample)(rng);

  if (count == 0)
    return cnts;

  if (T == 1) {
    cnts[0] = count;
    return cnts;
  }

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
  auto eval = [&](const Vector &k) -> double {
    double score = 0;
    for (size_t t = 0; t < T; ++t)
      score += lgamma(rate(t) + k(t)) - lgamma(k(t) + 1) + k(t) * log_p(t);
    // - lgamma(rate(t))
    return score;
  };

  auto compute_gradient = [&](const Vector &log_k, Vector &grad) -> void {
    grad = Vector(T);
    Vector k = unlog(log_k);
    double sum = 0;
    for (size_t t = 0; t < T; ++t)
      sum += grad(t) = k(t) * (digamma_diff_1p(k(t), rate(t)) + log_p(t));
    for (size_t t = 0; t < T; ++t)
      grad(t) = k(t) / count * (grad(t) - sum);
  };

  auto fnc = [&](const Vector &log_k, Vector &grad) -> double {
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
    } break;
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
    } break;
    case Sampling::Method::MH: {
      cnts = proportions.array().log();
      std::normal_distribution<double> normal_dist(0, 1);
      auto generate = [&normal_dist](const Vector &v, RNG &rng_) -> Vector {
        Vector w = v;
        for (auto &x : w)
          x += normal_dist(rng_);
        return w;
      };
      MetropolisHastings sampler(parameters.temperature);
      return unlog(sampler.sample(
          cnts, parameters.sample_iterations, rng, generate,
          [&](const Vector &log_k) -> double { return eval(unlog(log_k)); }));
    } break;
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
    default:
      break;
  }

  throw std::runtime_error("Error: this point should not be reached.");
  return cnts;
}

ostream &operator<<(ostream &os, const Experiment &experiment) {

  size_t reads = experiment.counts.matrix->sum();
  os << "Experiment "
     << "G = " << experiment.G << " "
     << "S = " << experiment.S << " "
     << "T = " << experiment.T << " "
     << "R = " << reads << " -> "
     << "R/S = " << 1.0 * reads / experiment.S << endl;
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
}  // namespace STD
