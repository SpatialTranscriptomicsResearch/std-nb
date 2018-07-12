#include "Model.hpp"

#include <map>
#include <memory>
#include <unordered_set>

#include "adagrad.hpp"
#include "adam.hpp"
#include "aux.hpp"
#include "gamma_func.hpp"
#include "io.hpp"
#include "pdist.hpp"
#include "rprop.hpp"
#include "sampling.hpp"

using namespace spec_parser;
using namespace std;

using spec_parser::expression::balance;
using spec_parser::expression::deriv;
using spec_parser::expression::eval;
using spec_parser::expression::simplify;

namespace STD {

namespace {

#include "model_aux.cpp"

size_t iter_cnt = 0;

template <typename T>
using ExprPtr = std::shared_ptr<spec_parser::expression::Exp<T>>;

template <typename T>
void compile_expression_and_derivs(const ExprPtr<T> &expr,
                                   const std::string &tag) {
  spec_parser::expression::codegen(simplify(balance(simplify(expr))), tag,
                                   collect_variables(expr));
  for (auto variable : collect_variables(expr)) {
    auto deriv_expr = simplify(balance(simplify(deriv(variable, expr))));
    spec_parser::expression::codegen(
        deriv_expr, tag + "-" + to_string(*variable), collect_variables(expr));
  }
}

bool initialized_jit = false;

Model::Model(const vector<Counts> &c, size_t T_, const Design::Design &design_,
             const ModelSpec &model_spec_, const Parameters &parameters_,
             bool initialize, bool construct_gp)
    : G(max_row_number(c)),
      T(T_),
      E(0),
      S(0),
      model_spec(model_spec_),
      design(design_),
      module_name("std-module"),  // TODO use unique module names
      rate_fnc(),
      odds_fnc(),
      rate_derivs(),
      odds_derivs(),
      experiments(),
      parameters(parameters_),
      contributions_gene_type(Matrix::Zero(G, T)),
      contributions_gene(Vector::Zero(G)) {
  if (not(initialized_jit)) {
    JIT::init_runtime(module_name, parameters.output_directory + "/");

    compile_expression_and_derivs(model_spec.rate_expr, "rate");
    compile_expression_and_derivs(model_spec.odds_expr, "odds");

    JIT::finalize_module(module_name);
    initialized_jit = true;
  }

  rate_fnc = JIT::get_function("rate");
  odds_fnc = JIT::get_function("odds");
  for (auto variable : collect_variables(model_spec.rate_expr))
    rate_derivs.push_back(JIT::get_function("rate-" + to_string(*variable)));
  for (auto variable : collect_variables(model_spec.odds_expr))
    odds_derivs.push_back(JIT::get_function("odds-" + to_string(*variable)));

  for (auto &counts : c)
    add_experiment(counts);
  update_contributions();

  LOG(debug) << "Model G = " << G << " T = " << T << " E = " << E;

  add_covariates();

  ensure_dimensions();

  coeff_debug_dump("INITIAL");
  if (construct_gp)
    construct_GPs();
  // coeff_debug_dump("BEFORE");
  // remove_redundant_terms();
  coeff_debug_dump("AFTER");

  if (initialize)
    for (auto &coeff : coeffs) {
      coeff->sample();
      if (coeff->type == Coefficient::Type::gp_coord
          and not parameters.gp.free_mean)
        dynamic_pointer_cast<Coefficient::Spatial::Coord>(coeff)
            ->subtract_mean();
    }

  size_t index = 0;
  for (auto coeff : coeffs)
    LOG(debug) << index++ << " " << *coeff << ": "
               << coeff->info.to_string(design.covariates);

  auto fnc = [](const Experiment &a, const Experiment &b) -> bool {
    return a.scale_ratio < b.scale_ratio;
  };
  double min_ratio = std::min_element(begin(experiments), end(experiments), fnc)
                         ->scale_ratio;
  for (auto &experiment : experiments)
    experiment.scale_ratio /= min_ratio;

  if (parameters.gp.center)
    center();

  verify_model(*this);
}

Model Model::clone() const {
  // TODO reactivate coeffs make efficient
  // avoid recompilation
  // avoid recalculation of spectral decomposition
  vector<Counts> counts;
  for (auto experiment : experiments)
    counts.push_back(experiment.counts);
  Model model(counts, T, design, model_spec, parameters, false, false);
  for (size_t idx = 0; idx < coeffs.size(); ++idx)
    if (coeffs[idx]->type == Coefficient::Type::gp_coord) {
      auto coord_coeff
          = dynamic_pointer_cast<Coefficient::Spatial::Coord>(coeffs[idx]);
      auto model_coord_coeff
          = dynamic_pointer_cast<Coefficient::Spatial::Coord>(
              model.coeffs[idx]);
      model_coord_coeff->gp = coord_coeff->gp;
    }
  return model;
}

void Model::ensure_dimensions() const {
  for (auto &experiment : experiments)
    experiment.ensure_dimensions();
}

void Model::setZero() {
  for (auto &coeff : coeffs)
    coeff->values.setZero();
}

size_t Model::number_variable() const {
  size_t s = 0;
  for (auto &coeff : coeffs)
    s += coeff->number_variable();
  return s;
}

size_t Model::size() const {
  size_t s = 0;
  for (auto &coeff : coeffs)
    s += coeff->size();
  return s;
}

Vector Model::vectorize() const {
  Vector v(size());
  auto iter = begin(v);

  for (auto &coeff : coeffs)
    for (auto &x : coeff->vectorize())
      *iter++ = x;

  assert(iter == end(v));

  return v;
}

void Model::from_vector(const Vector &v) {
  auto iter = begin(v);
  for (auto &coeff : coeffs)
    coeff->from_vector(iter);
}

pair<Matrix, Matrix> Model::compute_mean_and_var(size_t e) const {
  LOG(debug) << "Computing means and variances";

  const auto &exp = experiments[e];

  Matrix Mean = Matrix::Zero(G, exp.S);
  Matrix Var = Matrix::Zero(G, exp.S);

  const size_t num_rate_coeffs = rate_derivs.size();
  const size_t num_odds_coeffs = odds_derivs.size();

#pragma omp parallel if (DO_PARALLEL)
  {
    Matrix mean = Matrix::Zero(G, exp.S);
    Matrix var = Matrix::Zero(G, exp.S);
    double rate, odds;
    std::vector<std::vector<double>> rate_coeff_arrays, odds_coeff_arrays;
    for (size_t t = 0; t < T; ++t) {
      rate_coeff_arrays.push_back(std::vector<double>(num_rate_coeffs));
      odds_coeff_arrays.push_back(std::vector<double>(num_odds_coeffs));
    }

#pragma omp for schedule(guided)
    for (size_t g = 0; g < G; ++g)
      for (size_t s = 0; s < exp.S; ++s) {
        for (size_t t = 0; t < T; ++t) {
          for (size_t i = 0; i < num_rate_coeffs; ++i)
            rate_coeff_arrays[t][i] = exp.rate_coeffs[i]->get_actual(g, t, s);
          for (size_t i = 0; i < num_odds_coeffs; ++i)
            odds_coeff_arrays[t][i] = exp.odds_coeffs[i]->get_actual(g, t, s);

          rate = std::exp(rate_fnc(rate_coeff_arrays[t].data()));
          odds = std::exp(odds_fnc(odds_coeff_arrays[t].data()));
          mean(g, s) += rate * odds;
          var(g, s) += rate * odds * (1 + odds);
        }
      }

#pragma omp critical
    {
      Mean = Mean + mean;
      Var = Var + var;
    }
  }

  return {Mean, Var};
}

Model Model::compute_gradient(double &score) const {
  LOG(debug) << "Computing gradient";

  score = 0;
  Model gradient = clone();
  gradient.setZero();
  gradient.contributions_gene_type.setZero();
  for (auto &experiment : gradient.experiments) {
    experiment.contributions_spot_type.setZero();
    experiment.contributions_gene_type.setZero();
  }

  const size_t num_rate_coeffs = rate_derivs.size();
  const size_t num_odds_coeffs = odds_derivs.size();

#pragma omp parallel if (DO_PARALLEL)
  {
    Model grad = gradient.clone();
    auto rng = EntropySource::rngs[omp_get_thread_num()];
    double score_ = 0;
    Vector rate(T), odds(T);
    std::vector<std::vector<double>> rate_coeff_arrays, odds_coeff_arrays;
    for (size_t t = 0; t < T; ++t) {
      rate_coeff_arrays.push_back(std::vector<double>(num_rate_coeffs));
      odds_coeff_arrays.push_back(std::vector<double>(num_odds_coeffs));
    }

#pragma omp for schedule(guided)
    for (size_t g = 0; g < G; ++g)
      for (size_t e = 0; e < E; ++e)
        for (size_t s = 0; s < experiments[e].S; ++s)
          if (RandomDistribution::Uniform(rng)
              >= parameters.dropout_gene_spot) {
            const auto &exp = experiments[e];

            for (size_t t = 0; t < T; ++t) {
              for (size_t i = 0; i < num_rate_coeffs; ++i)
                rate_coeff_arrays[t][i]
                    = exp.rate_coeffs[i]->get_actual(g, t, s);
              for (size_t i = 0; i < num_odds_coeffs; ++i)
                odds_coeff_arrays[t][i]
                    = exp.odds_coeffs[i]->get_actual(g, t, s);

              rate(t) = std::exp(rate_fnc(rate_coeff_arrays[t].data()));
              odds(t) = std::exp(odds_fnc(odds_coeff_arrays[t].data()));
            }

            double total_rate = std::accumulate(begin(rate), end(rate), 0.0);
            assert(std::all_of(begin(odds), end(odds), [&odds](const auto &x) {
              return x == odds[0];
            }));
            double total_odds = odds[0];
            register_gradient(g, e, s, total_rate, total_odds, grad, rate, odds,
                              rate_coeff_arrays, odds_coeff_arrays, rng);
            double p = odds_to_prob(total_odds);
            score_ += log_negative_binomial(exp.counts(g, s), total_rate, p);
          }

#pragma omp critical
    {
      gradient = gradient + grad;
      score += score_;
    }
  }

  gradient.update_contributions();

  for (size_t i = 0; i < coeffs.size(); ++i)
    if (coeffs[i]->type != Coefficient::Type::gp_coord
        or iter_cnt >= parameters.gp.first_iteration)
      score += coeffs[i]->compute_gradient(gradient.coeffs[i]);

  return gradient;
}

void Model::register_gradient(
    size_t g, size_t e, size_t s, double total_rate, double total_odds,
    Model &gradient, const Vector &rate, const Vector &odds,
    const std::vector<std::vector<double>> &rate_coeffs,
    const std::vector<std::vector<double>> &odds_coeffs, RNG &rng) const {
  const double K = experiments[e].counts(g, s);
  double k = K;
  if (parameters.adjust_seq_depth)
    k = std::binomial_distribution<size_t>(k,
                                           1 / experiments[e].scale_ratio)(rng);
  if (parameters.downsample < 1)
    k = std::binomial_distribution<size_t>(k, parameters.downsample)(rng);
  const double r
      = total_rate
        * (parameters.adjust_seq_depth ? 1 / experiments[e].scale_ratio : 1);
  const double p = odds_to_prob(total_odds);
  const double log_one_minus_p = neg_odds_to_log_prob(total_odds);

  // const double rate_term = r * (log_one_minus_p + digamma_diff(r, k));
  const double rate_term = log_one_minus_p + digamma_diff(r, k);
  const double odds_term = k - p * (r + k);

  for (size_t t = 0; t < T; ++t) {
    gradient.experiments[e].contributions_gene_type(g, t)
        += rate(t) / total_rate * k;
    gradient.experiments[e].contributions_spot_type(s, t)
        += rate(t) / total_rate * k;
  }

  for (size_t t = 0; t < T; ++t) {
    {  // loop over rate covariates
      auto deriv_iter = begin(rate_derivs);
      for (size_t idx = 0; deriv_iter != end(rate_derivs);
           ++idx, ++deriv_iter) {
        gradient.experiments[e].rate_coeffs[idx]->get_raw(g, t, s)
            += rate_term * rate(t) * (*deriv_iter)(rate_coeffs[t].data());
      }
    }
    {  // loop over odds covariates
      auto deriv_iter = begin(odds_derivs);
      for (size_t idx = 0; deriv_iter != end(odds_derivs);
           ++idx, ++deriv_iter) {
        gradient.experiments[e].odds_coeffs[idx]->get_raw(g, t, s)
            += odds_term * (*deriv_iter)(odds_coeffs[t].data());
      }
    }
  }
}

void Model::center() {
  for (auto &coeff : coeffs)
    if (coeff->type == Coefficient::Type::gp_points)
      for (int t = 0; t < coeff->values.cols(); ++t)
        coeff->values.col(t)
            = coeff->values.col(t).array() - coeff->values.col(t).mean();
}

void Model::gradient_update(
    size_t num_iterations,
    std::function<bool(const CoefficientPtr)> is_included) {
  LOG(verbose) << "Performing gradient update iterations";
  for (auto coeff : coeffs)
    if (is_included(coeff))
      LOG(debug) << "Optimizing " << coeff->to_string();

  size_t current_iteration = 0;

  auto eval_and_compute_gradient = [&](const Vector &x, Vector &grad) {
    if (((iter_cnt++) % parameters.report_interval) == 0) {
      const size_t iteration_num_digits
          = 1 + floor(log(num_iterations) / log(10));
      store("iter" + to_string_embedded(iter_cnt - 1, iteration_num_digits)
            + "/");
    }

    // deactivate stochasticity in last iteration for correct contribution stats
    auto temp_parameters = parameters;
    if (++current_iteration == num_iterations) {
      parameters.dropout_gene_spot = 0;
      parameters.adjust_seq_depth = false;
      parameters.downsample = 1;
    }

    from_vector(x.array());

    if (parameters.gp.center)
      center();

    double score = 0;
    Model model_grad = compute_gradient(score);
    for (auto coeff : model_grad.coeffs)
      LOG(debug) << coeff << " grad = " << Stats::summary(coeff->values);

    // restore parameters from before deactivating stochasticity
    parameters = temp_parameters;

    // set gradient to zero for fixed coefficients
    for (auto coeff : model_grad.coeffs)
      if (coeff->type == Coefficient::Type::fixed)
        coeff->values.setZero();

    // set gradient to zero for coefficients that are not included
    for (auto coeff : model_grad.coeffs) {
      if (not is_included(coeff))
        coeff->values.fill(0);
    }

    grad = model_grad.vectorize();
    contributions_gene_type = model_grad.contributions_gene_type;
    for (size_t e = 0; e < E; ++e) {
      experiments[e].contributions_spot_type
          = model_grad.experiments[e].contributions_spot_type;
      experiments[e].contributions_gene_type
          = model_grad.experiments[e].contributions_gene_type;
    }

    LOG(info) << "Iteration " << iter_cnt << ", score: " << score;
    LOG(debug) << "x: " << endl << Stats::summary(x);
    LOG(debug) << "grad: " << endl << Stats::summary(grad);

    return score;
  };

  Vector x = vectorize().array();

  double fx;
  switch (parameters.optim_method) {
    case Optimize::Method::RPROP: {
      Vector grad;
      Vector prev_sign(Vector::Zero(x.size()));
      Vector rates(x.size());
      rates.fill(parameters.grad_alpha);
      for (size_t iter = 0; iter < num_iterations; ++iter) {
        fx = eval_and_compute_gradient(x, grad);
        rprop_update(grad, prev_sign, rates, x, parameters.rprop);
      }
    } break;
    case Optimize::Method::Gradient: {
      double alpha = parameters.grad_alpha;
      for (size_t iter = 0; iter < num_iterations; ++iter) {
        Vector grad;
        fx = eval_and_compute_gradient(x, grad);
        x = x + alpha * grad;
        LOG(debug) << "iter " << iter << " alpha: " << alpha;
        LOG(debug) << "iter " << iter << " fx: " << fx;
        LOG(debug) << "iter " << iter << " x: " << endl << Stats::summary(x);

        alpha *= parameters.grad_anneal;
      }
    } break;
    case Optimize::Method::AdaGrad: {
      Vector grad;
      Array agrad;
      Array ax;
      Array scale(Array::Zero(x.size()));
      for (size_t iter = 0; iter < num_iterations; ++iter) {
        fx = eval_and_compute_gradient(x, grad);
        agrad = grad.array();
        ax = x.array();
        adagrad_update(agrad, scale, ax, parameters.adagrad);
        x = ax.matrix();
      }
    } break;
    case Optimize::Method::Adam: {
      Vector grad;
      Array agrad;
      Array ax;
      Array mom1(Array::Zero(x.size()));
      Array mom2(Array::Zero(x.size()));
      auto updater = parameters.adam_nesterov_momentum ? nadam_update<Array>
                                                       : adam_update<Array>;
      for (size_t iter = 1; iter <= num_iterations; ++iter) {
        fx = eval_and_compute_gradient(x, grad);
        agrad = grad.array();
        ax = x.array();
        updater(agrad, mom1, mom2, ax, iter, parameters.adam);
        x = ax.matrix();
      }
    } break;
  }
  LOG(info) << "Final f(x) = " << fx;

  from_vector(x.array());
}

void Model::update_contributions() {
  contributions_gene_type.setZero();
  contributions_gene.setZero();
  for (auto &experiment : experiments)
#pragma omp parallel for if (DO_PARALLEL)
    for (size_t g = 0; g < G; ++g) {
      contributions_gene(g) += experiment.contributions_gene(g);
      for (size_t t = 0; t < T; ++t)
        contributions_gene_type(g, t)
            += experiment.contributions_gene_type(g, t);
    }
}

void Model::add_experiment(const Counts &counts) {
  experiments.push_back({this, counts, T, parameters});
  E++;
  S += experiments.back().S;
}

ostream &operator<<(ostream &os, const Model &model) {
  size_t n_variable = model.number_variable();
  os << "Spatial Transcriptome Deconvolution "
     << "G = " << model.G << " "
     << "T = " << model.T << " "
     << "E = " << model.E << " "
     << "S = " << model.S << endl
     << model.size() << " parameters, " << n_variable << " variable" << endl
     << "G * S = " << (model.G * model.S) << " -> "
     << 100.0 * n_variable / (model.G * model.S) << "%." << endl;

  for (auto &experiment : model.experiments)
    os << experiment;

  return os;
}

Model operator+(const Model &a, const Model &b) {
  Model model = a;

  model.contributions_gene_type += b.contributions_gene_type;
  model.contributions_gene += b.contributions_gene;

  for (size_t i = 0; i < a.coeffs.size(); ++i)
    model.coeffs[i]->values.array() += b.coeffs[i]->values.array();
  for (size_t e = 0; e < model.E; ++e)
    model.experiments[e] = model.experiments[e] + b.experiments[e];

  return model;
}
}  // namespace
