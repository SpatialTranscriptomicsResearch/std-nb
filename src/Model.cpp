#include "Model.hpp"

#include <map>
#include <memory>
#include <unordered_set>

#include <LBFGS.h>

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

Model::Model(const vector<Counts> &c, size_t T_, const Design::Design &design_,
             const ModelSpec &model_spec_, const Parameters &parameters_)
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
  JIT::init_runtime(module_name, parameters.output_directory + "/");

  compile_expression_and_derivs(model_spec.rate_expr, "rate");
  compile_expression_and_derivs(model_spec.odds_expr, "odds");

  JIT::finalize_module(module_name);

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
  add_gp_proxies();
  // coeff_debug_dump("BEFORE");
  // remove_redundant_terms();
  coeff_debug_dump("AFTER");

  size_t index = 0;
  for (auto coeff : coeffs)
    LOG(verbose) << index++ << " " << *coeff << ": "
                 << coeff->info.to_string(design.covariates);

  verify_model(*this);

  // TODO cov spot initialize spot scaling:
  // linear in number of counts, scaled so that mean = 1
}

void Model::ensure_dimensions() const {
  for (auto &experiment : experiments)
    experiment.ensure_dimensions();
}

void Model::setZero() {
  for (auto &coeff : coeffs)
    coeff->values.setZero();
}

size_t Model::number_parameters() const {
  size_t s = 0;
  for (auto &coeff : coeffs)
    s += coeff->number_parameters();
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

Model Model::compute_gradient(double &score) const {
  LOG(debug) << "Computing gradient";

  score = 0;
  Model gradient = *this;
  std::transform(
      begin(coeffs), end(coeffs), begin(gradient.coeffs),
      [](const auto &x) { return std::make_shared<Coefficient>(*x); });
  gradient.setZero();
  gradient.contributions_gene_type.setZero();
  for (auto &experiment : gradient.experiments) {
    experiment.contributions_spot_type.setZero();
    experiment.contributions_gene_type.setZero();
  }

#pragma omp parallel if (DO_PARALLEL)
  {
    Model grad = gradient;
    std::transform(
        begin(gradient.coeffs), end(gradient.coeffs), begin(grad.coeffs),
        [](const auto &x) { return std::make_shared<Coefficient>(*x); });
    auto rng = EntropySource::rngs[omp_get_thread_num()];
    double score_ = 0;
    Vector rate(T), odds(T);
    // TODO ensure all experiments have the same number of coefficients
    // currently, this could be violated due to redundancy removal
    size_t num_rate_coeffs = rate_derivs.size();  // TODO see above
    size_t num_odds_coeffs = odds_derivs.size();  // TODO see above
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
                    = coeffs[exp.rate_coeff_idxs[i]]->get_actual(g, t, s);
              for (size_t i = 0; i < num_odds_coeffs; ++i)
                odds_coeff_arrays[t][i]
                    = coeffs[exp.odds_coeff_idxs[i]]->get_actual(g, t, s);

              rate(t) = std::exp(rate_fnc(rate_coeff_arrays[t].data()));
              odds(t) = std::exp(odds_fnc(odds_coeff_arrays[t].data()));
            }

            auto cnts
                = exp.sample_contributions_gene_spot(g, s, rate, odds, rng);
            if (exp.counts(g, s) > 0)
              for (size_t t = 0; t < T; ++t) {
                register_gradient(g, e, s, t, cnts, grad, rate, odds,
                                  rate_coeff_arrays[t], odds_coeff_arrays[t]);
                double p = odds_to_prob(odds(t));
                score_ += log_negative_binomial(cnts[t], rate(t), p);
              }
            else
              for (size_t t = 0; t < T; ++t) {
                register_gradient_zero_count(g, e, s, t, cnts, grad, rate, odds,
                                             rate_coeff_arrays[t],
                                             odds_coeff_arrays[t]);
                double log_one_minus_p = neg_odds_to_log_prob(odds(t));
                score_ += log_negative_binomial_zero_log_one_minus_p(
                    rate(t), log_one_minus_p);
              }
          }

#pragma omp critical
    {
      gradient = gradient + grad;
      score += score_;
    }
  }

  gradient.update_contributions();

  for (size_t i = 0; i < coeffs.size(); ++i)
    if (coeffs[i]->distribution != Coefficient::Distribution::gp_proxy
        or iter_cnt >= parameters.gp.first_iteration)
      score += coeffs[i]->compute_gradient(coeffs, gradient.coeffs, i);

  return gradient;
}

void Model::register_gradient(size_t g, size_t e, size_t s, size_t t,
                              const Vector &cnts, Model &gradient,
                              const Vector &rate, const Vector &odds,
                              const std::vector<double> &rate_coeffs,
                              const std::vector<double> &odds_coeffs) const {
  gradient.experiments[e].contributions_gene_type(g, t) += cnts[t];
  gradient.experiments[e].contributions_spot_type(s, t) += cnts[t];

  const double k = cnts[t];
  const double r = rate(t);
  const double o = odds(t);
  const double p = odds_to_prob(o);
  const double log_one_minus_p = neg_odds_to_log_prob(o);

  const double rate_term = r * (log_one_minus_p + digamma_diff(r, k));
  const double odds_term = k - p * (r + k);

  // loop over rate covariates
  auto deriv_iter = begin(rate_derivs);
  for (size_t idx = 0; deriv_iter != end(rate_derivs); ++idx, ++deriv_iter) {
    size_t coeff_idx = experiments[e].rate_coeff_idxs[idx];
    gradient.coeffs[coeff_idx]->get_raw(g, t, s)
        += rate_term * (*deriv_iter)(rate_coeffs.data());
  }
  // loop over odds covariates
  deriv_iter = begin(odds_derivs);
  for (size_t idx = 0; deriv_iter != end(odds_derivs); ++idx, ++deriv_iter) {
    size_t coeff_idx = experiments[e].odds_coeff_idxs[idx];
    gradient.coeffs[coeff_idx]->get_raw(g, t, s)
        += odds_term * (*deriv_iter)(odds_coeffs.data());
  }
}

void Model::register_gradient_zero_count(
    size_t g, size_t e, size_t s, size_t t, const Vector &cnts, Model &gradient,
    const Vector &rate, const Vector &odds,
    const std::vector<double> &rate_coeffs,
    const std::vector<double> &odds_coeffs) const {
  const double r = rate(t);
  const double o = odds(t);
  const double p = odds_to_prob(o);
  const double log_one_minus_p = neg_odds_to_log_prob(o);

  const double rate_term = r * log_one_minus_p;
  const double odds_term = -p * r;

  // loop over rate covariates
  auto deriv_iter = begin(rate_derivs);
  for (size_t idx = 0; deriv_iter != end(rate_derivs); ++idx, ++deriv_iter) {
    size_t coeff_idx = experiments[e].rate_coeff_idxs[idx];
    gradient.coeffs[coeff_idx]->get_raw(g, t, s)
        += rate_term * (*deriv_iter)(rate_coeffs.data());
  }
  // loop over odds covariates
  deriv_iter = begin(odds_derivs);
  for (size_t idx = 0; deriv_iter != end(odds_derivs); ++idx, ++deriv_iter) {
    size_t coeff_idx = experiments[e].odds_coeff_idxs[idx];
    gradient.coeffs[coeff_idx]->get_raw(g, t, s)
        += odds_term * (*deriv_iter)(odds_coeffs.data());
  }
}

void Model::gradient_update(size_t num_iterations, std::function<bool(const Coefficient&)> is_included) {
  LOG(verbose) << "Performing gradient update iterations";

  size_t current_iteration = 0;

  auto fnc = [&](const Vector &x, Vector &grad) {
    if (((iter_cnt++) % parameters.report_interval) == 0) {
      const size_t iteration_num_digits
          = 1 + floor(log(num_iterations) / log(10));
      store("iter" + to_string_embedded(iter_cnt - 1, iteration_num_digits)
            + "/");
    }

    // deactivate dropout in the last iteration for correct contribution stats
    double dropout_temp = parameters.dropout_gene_spot;
    if (++current_iteration == num_iterations)
      parameters.dropout_gene_spot = 0;

    from_vector(x.array());
    double score = 0;
    Model model_grad = compute_gradient(score);
    for (auto &coeff : model_grad.coeffs)
      LOG(debug) << coeff << " grad = " << Stats::summary(coeff->values);

    parameters.dropout_gene_spot = dropout_temp;

    for (auto &coeff : model_grad.coeffs)
      if (coeff->distribution == Coefficient::Distribution::fixed)
        coeff->values.setZero();

    // set gradient to zero for coefficients that are not included
    for (auto &coeff : model_grad.coeffs) {
      if (not is_included(*coeff))
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

    if (parameters.optim_method == Optimize::Method::lBFGS) {
      // as for lBFGS we want to minimize, we have to negate
      score = -score;
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
        fx = fnc(x, grad);
        rprop_update(grad, prev_sign, rates, x, parameters.rprop);
      }
    } break;
    case Optimize::Method::Gradient: {
      double alpha = parameters.grad_alpha;
      for (size_t iter = 0; iter < num_iterations; ++iter) {
        Vector grad;
        fx = fnc(x, grad);
        x = x + alpha * grad;
        LOG(debug) << "iter " << iter << " alpha: " << alpha;
        LOG(debug) << "iter " << iter << " fx: " << fx;
        LOG(debug) << "iter " << iter << " x: " << endl << Stats::summary(x);

        alpha *= parameters.grad_anneal;
      }
    } break;
    case Optimize::Method::lBFGS: {
      using namespace LBFGSpp;
      LBFGSParam<double> param;
      param.epsilon = parameters.lbfgs_epsilon;
      param.max_iterations = parameters.lbfgs_iter;
      // Create solver and function object
      LBFGSSolver<double> solver(param);

      int niter = solver.minimize(fnc, x, fx);

      LOG(verbose) << "lBFGS performed " << niter << " iterations";
    } break;
    case Optimize::Method::AdaGrad: {
      Vector grad;
      Array agrad;
      Array ax;
      Array scale(Array::Zero(x.size()));
      for (size_t iter = 0; iter < num_iterations; ++iter) {
        fx = fnc(x, grad);
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
      for (size_t iter = 1; iter <= num_iterations; ++iter) {
        fx = fnc(x, grad);
        agrad = grad.array();
        ax = x.array();
        adam_update(agrad, mom1, mom2, ax, iter, parameters.adam);
        x = ax.matrix();
      }
    } break;
  }
  LOG(info) << "Final f(x) = " << fx;

  from_vector(x.array());
}

// computes a matrix M(g,t) =
//   gamma(g,t) sum_e beta(e,g) lambda(e,g,t) sum_s theta(e,s,t) sigma(e,s)
Matrix Model::expected_gene_type() const {
  Matrix m = Matrix::Zero(G, T);
  /* TODO: needs rewrite
  for (auto &experiment : experiments)
    m += experiment.expected_gene_type();
  */
  return m;
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
  size_t n_params = model.number_parameters();
  os << "Spatial Transcriptome Deconvolution "
     << "G = " << model.G << " "
     << "T = " << model.T << " "
     << "E = " << model.E << " "
     << "S = " << model.S << endl
     << model.size() << " parameters, " << n_params << " variable" << endl
     << "G * S = " << (model.G * model.S) << " -> "
     << 100.0 * n_params / (model.G * model.S) << "%." << endl;

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
}  // namespace STD
