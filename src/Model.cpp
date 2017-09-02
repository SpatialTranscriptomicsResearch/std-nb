#include "Model.hpp"
#include <LBFGS.h>
#include <map>
#include "aux.hpp"
#include "gamma_func.hpp"
#include "io.hpp"
#include "pdist.hpp"
#include "rprop.hpp"
#include "sampling.hpp"

using namespace std;

namespace STD {

CovariateInformation drop_covariate(const CovariateInformation &info,
                                    const Design &design,
                                    const std::string &cov_label) {
  auto mod_info = info;
  for (size_t idx = 0; idx < info.idxs.size(); ++idx)
    if (design.covariates[info.idxs[idx]].label == cov_label) {
      mod_info.idxs.erase(begin(mod_info.idxs) + idx);
      mod_info.vals.erase(begin(mod_info.vals) + idx);
    }
  return mod_info;
}

CovariateInformation drop_covariates(
    CovariateInformation info, const Design &design,
    const std::vector<std::string> &cov_labels) {
  for (auto &cov_label : cov_labels)
    info = drop_covariate(info, design, cov_label);
  return info;
}

std::vector<Coefficient>::iterator Model::find_coefficient(
    const std::string &label, Coefficient::Variable variable,
    Coefficient::Kind kind, Coefficient::Distribution distribution,
    const CovariateInformation &info) {
  return find_if(begin(coeffs), end(coeffs), [&](const Coefficient &coeff) {
    return coeff.label == label and coeff.variable == variable
           and coeff.kind == kind and coeff.distribution == distribution
           and coeff.info == info;
  });
}

// returns a vector of indices of the coefficients for each sample
void Model::add_covariate_terms(const Formula::Term &term,
                                Coefficient::Variable variable) {
  LOG(debug) << "Treating next " << to_string(variable) << " formula term.";

  Coefficient::Kind kind = determine_kind(term);
  vector<size_t> cov_idxs = design.determine_covariate_idxs(term);

  Coefficient::Distribution distribution = choose_distribution(
      variable, kind, parameters.distribution_mode, parameters.gp.use);

  LOG(debug) << "Coefficient::Kind = " << to_string(kind);

  const string label = to_string(variable);

  for (size_t e = 0; e < E; ++e) {
    CovariateInformation info
        = {cov_idxs, design.get_covariate_value_idxs(e, cov_idxs)};
    auto coeff_iter
        = find_coefficient(label, variable, kind, distribution, info);
    if (coeff_iter == end(coeffs)) {
      LOG(debug) << "Creating new coefficient";
      // this covariate value combination was not previously used
      Coefficient covterm(G, T, experiments[e].S, label, variable, kind,
                          distribution, info);
      coeffs.push_back(covterm);
      coeff_iter = prev(end(coeffs));
    }
    coeff_iter->experiment_idxs.push_back(e);
    size_t idx = distance(begin(coeffs), coeff_iter);
    experiments[e].coeff_idxs(variable).push_back(idx);

    LOG(verbose) << "idx = " << idx;

    if (distribution == Coefficient::Distribution::log_gp) {
      auto gp_coord_info = drop_covariates(
          info, design, {DesignNS::spot_label, DesignNS::section_label});
      auto gp_coord_kind = kind & ~Coefficient::Kind::spot;
      auto gp_coord_coeff_iter = find_coefficient(
          label, variable, gp_coord_kind,
          Coefficient::Distribution::log_gp_coord, gp_coord_info);
      if (gp_coord_coeff_iter == end(coeffs)) {
        LOG(verbose) << "Adding GP coordinate system";
        Coefficient gp_coord_coeff(G, T, 0, label, variable, gp_coord_kind,
                                   Coefficient::Distribution::log_gp_coord,
                                   gp_coord_info);
        coeffs.push_back(gp_coord_coeff);
        gp_coord_coeff_iter = prev(end(coeffs));
      }
      LOG(verbose) << "Updating GP coordinate system"
                   << distance(begin(coeffs), gp_coord_coeff_iter);
      gp_coord_coeff_iter->experiment_idxs.push_back(e);
      gp_coord_coeff_iter->prior_idxs.push_back(idx);

      size_t gp_coord_idx = distance(begin(coeffs), gp_coord_coeff_iter);

      auto gp_info
          = drop_covariates(info, design,
                            {DesignNS::spot_label, DesignNS::section_label,
                             DesignNS::coordsys_label});
      auto gp_kind = kind & ~Coefficient::Kind::spot;
      auto gp_coeff_iter
          = find_coefficient(label, variable, gp_kind,
                             Coefficient::Distribution::log_gp_proxy, gp_info);
      if (gp_coeff_iter == end(coeffs)) {
        LOG(verbose) << "Adding GP proxy";
        Coefficient gp_coeff(G, T, 0, label, variable, gp_kind,
                             Coefficient::Distribution::log_gp_proxy, gp_info);
        coeffs.push_back(gp_coeff);
        gp_coeff_iter = prev(end(coeffs));
      }
      LOG(verbose) << "Updating GP proxy "
                   << distance(begin(coeffs), gp_coeff_iter);
      gp_coeff_iter->experiment_idxs.push_back(e);
      if (std::find(begin(gp_coeff_iter->prior_idxs),
                    end(gp_coeff_iter->prior_idxs), gp_coord_idx)
          == end(gp_coeff_iter->prior_idxs))
        gp_coeff_iter->prior_idxs.push_back(gp_coord_idx);
    }
  }
}

void Model::add_gp_proxies() {
  LOG(verbose) << "Constructing GP proxies";
  for (size_t idx = 0; idx < coeffs.size(); ++idx)
    if (coeffs[idx].distribution == Coefficient::Distribution::log_gp_proxy) {
      LOG(verbose) << "Constructing GP proxy " << idx;
      for (auto &coord_coeff_idx : coeffs[idx].prior_idxs) {
        LOG(verbose) << "using coordinate system coefficient "
                     << coord_coeff_idx;
        auto &coord_coeff = coeffs[coord_coeff_idx];
        auto exp_idxs = coord_coeff.experiment_idxs;
        auto prior_idxs = coord_coeff.prior_idxs;
        size_t n = 0;
        for (size_t e : exp_idxs)
          n += experiments[e].S;
        size_t ncol = experiments[exp_idxs[0]].coords.cols();
        LOG(verbose) << "n = " << n;
        Matrix m = Matrix::Zero(n, ncol);
        size_t i = 0;
        for (size_t e : exp_idxs) {
          for (size_t s = 0; s < experiments[e].S; ++s)
            for (size_t j = 0; j < ncol; ++j)
              m(i + s, j) = experiments[e].coords(s, j);
          i += experiments[e].S;
        }
        coord_coeff.gp = GP::GaussianProcess(m, parameters.gp.length_scale);
      }
    }
}

void Model::add_prior_coefficients() {
  const size_t n = coeffs.size();
  for (size_t idx = 0; idx < n; ++idx) {
    if (coeffs[idx].distribution == Coefficient::Distribution::log_gp_coord
        or coeffs[idx].distribution == Coefficient::Distribution::log_gp_proxy)
      continue;

    const size_t current_size = coeffs.size();
    coeffs[idx].prior_idxs.push_back(current_size);
    coeffs[idx].prior_idxs.push_back(current_size + 1);
    CovariateInformation info = coeffs[idx].info;
    Coefficient::Variable variable = Coefficient::Variable::prior;
    string label = to_string(variable);
    Coefficient covterm(0, 0, 0, label, variable, Coefficient::Kind::scalar,
                        Coefficient::Distribution::fixed, info);
    covterm.experiment_idxs = coeffs[idx].experiment_idxs;

    covterm.get(0, 0, 0)
        = parameters.hyperparameters.get_param(coeffs[idx].distribution, 0);
    coeffs.push_back(covterm);

    covterm.get(0, 0, 0)
        = parameters.hyperparameters.get_param(coeffs[idx].distribution, 1);
    coeffs.push_back(covterm);
  }
}

Model::Model(const vector<Counts> &c, size_t T_, const Design &design_,
             const Parameters &parameters_)
    : G(max_row_number(c)),
      T(T_),
      E(0),
      S(0),
      design(design_),
      experiments(),
      parameters(parameters_),
      contributions_gene_type(Matrix::Zero(G, T)),
      contributions_gene(Vector::Zero(G)) {
  for (auto &counts : c)
    add_experiment(counts);
  update_contributions();

  LOG(debug) << "Model G = " << G << " T = " << T << " E = " << E;

  for (auto &term : parameters.rate_formula.terms)
    add_covariate_terms(term, Coefficient::Variable::rate);
  for (auto &term : parameters.variance_formula.terms)
    add_covariate_terms(term, Coefficient::Variable::odds);

  coeff_debug_dump("INITIAL");
  add_gp_proxies();
  coeff_debug_dump("BEFORE");
  remove_redundant_terms();
  coeff_debug_dump("AFTER");
  add_prior_coefficients();
  coeff_debug_dump("FINAL");

  // TODO cov spot initialize spot scaling:
  // linear in number of counts, scaled so that mean = 1

  enforce_positive_parameters(parameters.min_value);
}

void Model::coeff_debug_dump(const string &tag) const {
  for (auto coeff : coeffs)
    LOG(debug) << tag << " " << coeff << ": "
               << coeff.info.to_string(design.covariates);
  auto fnc = [&](const string &s, size_t idx, size_t e) {
    LOG(debug) << tag << " " << s << " experiment " << e << " " << idx << " "
               << coeffs[idx] << ": "
               << coeffs[idx].info.to_string(design.covariates);
  };
  for (size_t e = 0; e < E; ++e) {
    for (auto idx : experiments[e].rate_coeff_idxs)
      fnc("rate", idx, e);
    for (auto idx : experiments[e].odds_coeff_idxs)
      fnc("odds", idx, e);
  }
}

vector<size_t> find_redundant(const vector<vector<size_t>> &v) {
  using inv_map_t = multimap<vector<size_t>, size_t>;
  inv_map_t m;
  vector<size_t> redundant;
  for (size_t i = 0; i < v.size(); ++i) {
    auto key = v[i];
    if (not key.empty()) {
      pair<vector<size_t>, size_t> entry = {key, i};
      m.insert(entry);
      if (m.count(key) > 1)
        redundant.push_back(i);
    }
  }
  return redundant;
}

void Model::remove_redundant_terms() {
  using Variable = Coefficient::Variable;
  using Kind = Coefficient::Kind;
  for (auto variable : {Variable::rate, Variable::odds})
    for (auto kind : {Kind::scalar, Kind::gene, Kind::type, Kind::spot,
                      Kind::gene_type, Kind::spot_type})
      remove_redundant_terms(variable, kind);
}

// TODO covariates: add redundant term labels
void Model::remove_redundant_terms(Coefficient::Variable variable,
                                   Coefficient::Kind kind) {
  vector<vector<size_t>> cov2groups(coeffs.size());
  for (size_t e = 0; e < E; ++e)
    for (auto idx : experiments[e].coeff_idxs(variable))
      if (coeffs[idx].kind == kind)
        cov2groups[idx].push_back(e);
  auto redundant = find_redundant(cov2groups);
  sort(begin(redundant), end(redundant));  // needed?
  size_t removed = 0;
  for (auto r : redundant) {
    LOG(verbose) << "Removing " << r << ": " << coeffs[r] << ": "
                 << coeffs[r - removed].info.to_string(design.covariates);
    coeffs.erase(begin(coeffs) + r - removed);
    removed++;
  }
  for (size_t e = 0; e < E; ++e) {
    for (auto v : {Coefficient::Variable::rate, Coefficient::Variable::odds}) {
      auto &idxs = experiments[e].coeff_idxs(v);
      for (auto r : redundant)
        idxs.erase(remove(begin(idxs), end(idxs), r), end(idxs));
      for (auto &idx : idxs)
        for (auto r : redundant)
          if (idx > r)
            idx--;
    }
  }
}

template <typename V>
vector<size_t> get_order(const V &v) {
  size_t N = v.size();
  vector<size_t> order(N);
  iota(begin(order), end(order), 0);
  sort(begin(order), end(order),
       [&v](size_t a, size_t b) { return v[a] > v[b]; });
  return order;
}

void Model::store(const string &prefix_, bool mean_and_var,
                  bool reorder) const {
  string prefix = parameters.output_directory + prefix_;
  {
    using namespace boost::filesystem;
    if (not((exists(prefix) and is_directory(prefix))
            or create_directory(prefix)))
      throw(std::runtime_error("Couldn't create directory " + prefix));
  }
  auto type_names = form_factor_names(T);
  auto &gene_names = experiments.begin()->counts.row_names;

  auto exp_gene_type = expected_gene_type();
  vector<size_t> order;
  if (reorder) {
    auto cs = colSums<Vector>(exp_gene_type);
    order = get_order(cs);
  }

#pragma omp parallel sections if (DO_PARALLEL)
  {
#pragma omp section
    {
      // TODO use parse-able format
      ofstream ofs(prefix + "hyperparameters.txt");
      ofs << parameters.hyperparameters;
    }
#pragma omp section
    write_matrix(exp_gene_type, prefix + "expected-features" + FILENAME_ENDING,
                 parameters.compression_mode, gene_names, type_names, order);

// TODO cov perhaps write out a single file for the scalar covariates
#pragma omp section
    {
      map<pair<Coefficient::Variable, Coefficient::Kind>, size_t> kind_counts;
      for (auto &coeff : coeffs) {
        auto iter = kind_counts.insert({{coeff.variable, coeff.kind}, 0});
        vector<string> spot_names;
        if (coeff.spot_dependent())
          for (auto idx : coeff.experiment_idxs)
            spot_names.insert(begin(spot_names),
                              begin(experiments[idx].counts.col_names),
                              end(experiments[idx].counts.col_names));
        coeff.store(prefix + "covariate-" + to_string(coeff.variable) + "-"
                        + to_token(coeff.kind) + "-"
                        + to_string_embedded(iter.first->second++, 2) + "_"
                        + coeff.info.to_string(design.covariates)
                        + FILENAME_ENDING,
                    parameters.compression_mode, gene_names, spot_names,
                    type_names, order);
      }
    }

#pragma omp section
    write_matrix(contributions_gene_type,
                 prefix + "contributions_gene_type" + FILENAME_ENDING,
                 parameters.compression_mode, gene_names, type_names, order);
#pragma omp section
    write_vector(contributions_gene,
                 prefix + "contributions_gene" + FILENAME_ENDING,
                 parameters.compression_mode, gene_names);
  }
  for (size_t e = 0; e < E; ++e) {
    string exp_prefix = prefix + "experiment"
                        + to_string_embedded(e, EXPERIMENT_NUM_DIGITS) + "-";
    experiments[e].store(exp_prefix, order);
  }
  if (mean_and_var)
    for (size_t e = 0; e < E; ++e) {
      write_matrix(experiments[e].expectation(),
                   prefix + "experiment"
                       + to_string_embedded(e, EXPERIMENT_NUM_DIGITS) + "-"
                       + "expected_counts" + FILENAME_ENDING,
                   parameters.compression_mode, gene_names,
                   experiments[e].counts.col_names);
      write_matrix(experiments[e].variance(),
                   prefix + "experiment"
                       + to_string_embedded(e, EXPERIMENT_NUM_DIGITS) + "-"
                       + "variance_counts" + FILENAME_ENDING,
                   parameters.compression_mode, gene_names,
                   experiments[e].counts.col_names);
    }
}

/* TODO covariates enable loading of subsets of covariates */
void Model::restore(const string &prefix) {
  {
    map<Coefficient::Kind, size_t> kind_counts;
    for (auto &coeff : coeffs) {
      auto iter
          = kind_counts.insert(pair<Coefficient::Kind, size_t>(coeff.kind, 0));
      coeff.restore(prefix + "covariate-" + to_string(coeff.variable) + "-"
                    + to_token(coeff.kind) + "-"
                    + to_string_embedded(iter.first->second++, 2) + "_"
                    + coeff.info.to_string(design.covariates)
                    + FILENAME_ENDING);
    }
  }

  contributions_gene_type = parse_file<Matrix>(
      prefix + "contributions_gene_type" + FILENAME_ENDING, read_matrix, "\t");
  contributions_gene
      = parse_file<Vector>(prefix + "contributions_gene" + FILENAME_ENDING,
                           read_vector<Vector>, "\t");

  for (size_t e = 0; e < E; ++e) {
    string exp_prefix = prefix + "experiment"
                        + to_string_embedded(e, EXPERIMENT_NUM_DIGITS) + "-";
    experiments[e].restore(exp_prefix);
  }
}

void Model::setZero() {
  for (auto &coeff : coeffs)
    coeff.values.setZero();
}

size_t Model::number_parameters() const {
  size_t s = 0;
  for (auto &coeff : coeffs)
    s += coeff.number_parameters();
  return s;
}

size_t Model::size() const {
  size_t s = 0;
  for (auto &coeff : coeffs)
    s += coeff.size();
  return s;
}

Vector Model::vectorize() const {
  Vector v(size());
  auto iter = begin(v);

  for (auto &coeff : coeffs)
    for (auto &x : coeff.vectorize())
      *iter++ = x;

  assert(iter == end(v));

  return v;
}

void Model::from_vector(const Vector &v) {
  auto iter = begin(v);
  for (auto &coeff : coeffs)
    coeff.from_vector(iter);
}

Model Model::compute_gradient(double &score) const {
  LOG(debug) << "Computing gradient";

  vector<Matrix> rate_gt, rate_st;
  vector<Matrix> odds_gt, odds_st;

  for (auto &experiment : experiments) {
    rate_gt.push_back(
        experiment.compute_gene_type_table(experiment.rate_coeff_idxs));
    rate_st.push_back(
        experiment.compute_spot_type_table(experiment.rate_coeff_idxs));
    odds_gt.push_back(
        experiment.compute_gene_type_table(experiment.odds_coeff_idxs));
    odds_st.push_back(
        experiment.compute_spot_type_table(experiment.odds_coeff_idxs));
  }

  score = 0;
  Model gradient = *this;
  gradient.setZero();
  gradient.contributions_gene_type.setZero();
  for (auto &experiment : gradient.experiments) {
    experiment.contributions_spot_type.setZero();
    experiment.contributions_gene_type.setZero();
  }

#pragma omp parallel if (DO_PARALLEL)
  {
    Model grad = gradient;
    auto rng = EntropySource::rngs[omp_get_thread_num()];
    double score_ = 0;

#pragma omp for
    for (size_t g = 0; g < G; ++g)
      for (size_t e = 0; e < E; ++e)
        for (size_t s = 0; s < experiments[e].S; ++s)
          if (RandomDistribution::Uniform(rng)
              >= parameters.dropout_gene_spot) {
            auto cnts = experiments[e].sample_contributions_gene_spot(
                g, s, rate_gt[e], rate_st[e], odds_gt[e], odds_st[e], rng);
            for (size_t t = 0; t < T; ++t) {
              double r = rate_gt[e](g, t) * rate_st[e](s, t);
              double odds = odds_gt[e](g, t) * odds_st[e](s, t);
              double p = odds_to_prob(odds);
              score += log_negative_binomial(cnts[t], r, p);
            }
            register_gradient(g, e, s, cnts, grad, rate_gt[e], rate_st[e],
                              odds_gt[e], odds_st[e]);
          }

#pragma omp critical
    {
      gradient = gradient + grad;
      score += score_;
    }
  }

  gradient.update_contributions();

  for (size_t i = 0; i < coeffs.size(); ++i)
    score += coeffs[i].compute_gradient(coeffs, gradient.coeffs, i);

  return gradient;
}

void Model::register_gradient(size_t g, size_t e, size_t s, const Vector &cnts,
                              Model &gradient, const Matrix &rate_gt,
                              const Matrix &rate_st, const Matrix &odds_gt,
                              const Matrix &odds_st) const {
  for (size_t t = 0; t < T; ++t)
    gradient.experiments[e].contributions_gene_type(g, t) += cnts[t];
  for (size_t t = 0; t < T; ++t)
    gradient.experiments[e].contributions_spot_type(s, t) += cnts[t];

  for (size_t t = 0; t < T; ++t) {
    const double k = cnts[t];
    const double r = rate_gt(g, t) * rate_st(s, t);
    const double odds = odds_gt(g, t) * odds_st(s, t);
    const double p = odds_to_prob(odds);
    const double log_one_minus_p = neg_odds_to_log_prob(odds);

    const double rate_term = r * (log_one_minus_p + digamma_diff(r, k));
    const double odds_term = k - p * (r + k);

    for (auto &idx : gradient.experiments[e].rate_coeff_idxs)
      gradient.coeffs[idx].get(g, t, s) += rate_term;
    for (auto &idx : gradient.experiments[e].odds_coeff_idxs)
      gradient.coeffs[idx].get(g, t, s) += odds_term;
  }
}

// calculate parameter's likelihood
double Model::param_likel() const {
  double score = 0;
  // TODO covariates likelihood
  /*
  if (parameters.targeted(Target::gamma)) {
    const double a = parameters.hyperparameters.gamma_1;
    const double b = parameters.hyperparameters.gamma_2;
#pragma omp parallel for if (DO_PARALLEL)
    for (size_t g = 0; g < G; ++g)
      for (size_t t = 0; t < T; ++t)
        score += log_gamma_rate(gamma(g, t), a, b);
  }

  if (parameters.targeted(Target::lambda)) {
    const double a = parameters.hyperparameters.lambda_1;
    const double b = parameters.hyperparameters.lambda_2;
    for (auto &coord_sys : coordinate_systems)
      for (auto e : coord_sys.members)
#pragma omp parallel for if (DO_PARALLEL)
        for (size_t g = 0; g < G; ++g)
          for (size_t t = 0; t < T; ++t)
            score += log_gamma_rate(experiments[e].lambda(g, t), a, b);
  }

  if (parameters.targeted(Target::beta)) {
    const double a = parameters.hyperparameters.beta_1;
    const double b = parameters.hyperparameters.beta_2;
    for (auto &coord_sys : coordinate_systems)
      for (auto e : coord_sys.members)
#pragma omp parallel for if (DO_PARALLEL)
        for (size_t g = 0; g < G; ++g)
          score += log_gamma_rate(experiments[e].beta(g), a, b);
  }

  if (parameters.targeted(Target::theta))
    for (auto &coord_sys : coordinate_systems)
      for (auto e : coord_sys.members)
#pragma omp parallel for if (DO_PARALLEL)
        for (size_t s = 0; s < experiments[e].S; ++s)
          for (size_t t = 0; t < T; ++t) {
            const double a = experiments[e].field(s, t) * mix_prior.r(t);
            const double b = mix_prior.p(t);
            score += log_gamma_rate(experiments[e].theta(s, t), a, b);
          }

  if (parameters.targeted(Target::rho)) {
    const double a = parameters.hyperparameters.rho_1;
    const double b = parameters.hyperparameters.rho_2;
#pragma omp parallel for if (DO_PARALLEL)
    for (size_t g = 0; g < G; ++g)
      for (size_t t = 0; t < T; ++t) {
        const double no = negodds_rho(g, t);
        score += log_beta_neg_odds(no, a, b);
      }
  }
  */

  return score;
}

size_t iter_cnt = 0;

void Model::gradient_update() {
  LOG(verbose) << "Performing gradient update iterations";

  auto fnc = [&](const Vector &x, Vector &grad) {
    if (((iter_cnt++) % parameters.report_interval) == 0) {
      const size_t iteration_num_digits
          = 1 + floor(log(parameters.grad_iterations) / log(10));
      store("iter" + to_string_embedded(iter_cnt - 1, iteration_num_digits)
            + "/");
    }

    from_vector(x.array().exp());
    enforce_positive_parameters(parameters.min_value);
    double score = 0;
    Model model_grad = compute_gradient(score);
    for (auto &coeff : model_grad.coeffs)
      LOG(debug) << coeff << " grad = " << Stats::summary(coeff.values);

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
    LOG(verbose) << "x: " << endl << Stats::summary(x);
    LOG(verbose) << "grad: " << endl << Stats::summary(grad);

    return score;
  };

  Vector x = vectorize().array().log();

  double fx;
  switch (parameters.optim_method) {
    case Optimize::Method::RPROP: {
      Vector grad;
      Vector prev_sign(Vector::Zero(x.size()));
      Vector rates(x.size());
      rates.fill(parameters.grad_alpha);
      for (size_t iter = 0; iter < parameters.grad_iterations; ++iter) {
        fx = fnc(x, grad);
        rprop_update(grad, prev_sign, rates, x, parameters.rprop);
        enforce_positive_and_warn("RPROP log params", x,
                                  log(parameters.min_value),
                                  parameters.warn_lower_limit);
      }
    } break;
    case Optimize::Method::Gradient: {
      double alpha = parameters.grad_alpha;
      for (size_t iter = 0; iter < parameters.grad_iterations; ++iter) {
        Vector grad;
        fx = fnc(x, grad);
        x = x + alpha * grad;
        enforce_positive_and_warn("GradOpt log params", x,
                                  log(parameters.min_value),
                                  parameters.warn_lower_limit);
        LOG(verbose) << "iter " << iter << " alpha: " << alpha;
        LOG(verbose) << "iter " << iter << " fx: " << fx;
        LOG(verbose) << "iter " << iter << " x: " << endl << Stats::summary(x);

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
  }
  LOG(verbose) << "Final f(x) = " << fx;

  from_vector(x.array().exp());
}

void Model::enforce_positive_parameters(double min_value) {
  for (size_t i = 0; i < coeffs.size(); ++i)
    enforce_positive_and_warn(
        to_string(coeffs[i].kind) + " " + to_string(coeffs[i].variable)
            + " covariate " + to_string_embedded(i, 3),
        coeffs[i].values, min_value, parameters.warn_lower_limit);
}

/* TODO covariates reactivate likelihood
double Model::log_likelihood(const string &prefix) const {
  double l = 0;
  for (size_t e = 0; e < E; ++e) {
    Matrix m = experiments[e].log_likelihood();

    auto &gene_names = experiments[e].counts.row_names;
    auto &spot_names = experiments[e].counts.col_names;

    string exp_prefix = prefix + "experiment"
                        + to_string_embedded(e, EXPERIMENT_NUM_DIGITS) + "-";
    write_matrix(m, exp_prefix + "loglikelihood" + FILENAME_ENDING,
                 parameters.compression_mode, gene_names, spot_names);
    write_matrix(*experiments[e].counts.matrix,
                 exp_prefix + "loglikelihood-counts" + FILENAME_ENDING,
                 parameters.compression_mode, gene_names, spot_names);
    for (auto &x : m)
      l += x;
  }
  return l;
}
*/

// computes a matrix M(g,t) =
//   gamma(g,t) sum_e beta(e,g) lambda(e,g,t) sum_s theta(e,s,t) sigma(e,s)
Matrix Model::expected_gene_type() const {
  Matrix m = Matrix::Zero(G, T);
  for (auto &experiment : experiments)
    m += experiment.expected_gene_type();
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
    model.coeffs[i].values.array() += b.coeffs[i].values.array();
  for (size_t e = 0; e < model.E; ++e)
    model.experiments[e] = model.experiments[e] + b.experiments[e];

  return model;
}
}
