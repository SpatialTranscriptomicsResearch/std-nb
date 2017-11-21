#include "Model.hpp"

#include <map>
#include <memory>
#include <unordered_set>

#include <LBFGS.h>

#include "aux.hpp"
#include "gamma_func.hpp"
#include "io.hpp"
#include "pdist.hpp"
#include "rprop.hpp"
#include "sampling.hpp"

using namespace spec_parser;
using namespace std;

using spec_parser::expression::eval;
using spec_parser::expression::deriv;

namespace STD {

namespace {

size_t iter_cnt = 0;

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

CovariateInformation get_covariate_info(
    const Design& design, const set<string>& covariates, size_t experiment)
{
  auto covariates_ = covariates;

  using namespace DesignNS;

  // spot dependency implies section dependency
  if (covariates_.find(spot_label) != covariates_.end()) {
    covariates_.insert(section_label);
  }

  vector<size_t> cov_idxs = design.determine_covariate_idxs(covariates_);
  vector<size_t> cov_vals;
  for (auto &covariate_idx : cov_idxs)
    cov_vals.push_back(
        design.dataset_specifications[experiment].covariate_values[covariate_idx]);

  return CovariateInformation { cov_idxs, cov_vals };
}

// Removes trailing zeros in a numeric string's decimals.
string remove_trailing_zeros(const string& str) {
  if (str.find('.') == string::npos) {
    return str;
  }
  auto pos = str.end();
  while (*(--pos) != '.') {
    if (*pos != '0') {
      pos += 1;
      break;
    }
  }
  return string(str.begin(), pos);
}

void verify_model(const Model& m) {
  {  // check for overspecification
    static const auto input_dim
        = Coefficient::Kind::gene | Coefficient::Kind::spot;
    for (auto &x : m.coeffs) {
      if ((x->kind & input_dim) == input_dim) {
        throw runtime_error(
            "Error: coefficient '" + x->label
            + "' has dimensionality greater than or equal to the input data.");
      }
    }
  }

  {  // check for cycles in model spec
    auto check_cycles = [&m](size_t root) {
      vector<bool> visited(m.coeffs.size());
      function<void(size_t)> go = [&m, &visited, &go](size_t x) {
        if (visited[x]) {
          throw runtime_error(
              "Error: cyclic model specifications are currently not "
              "supported.");
        }
        visited[x] = true;
        for (auto &p : m.coeffs[x]->prior_idxs) {
          go(p);
        }
        visited[x] = false;
      };
      return go(root);
    };
    unordered_set<size_t> coeffs;
    for (auto &e : m.experiments) {
      coeffs.insert(e.rate_coeff_idxs.begin(), e.rate_coeff_idxs.end());
      coeffs.insert(e.odds_coeff_idxs.begin(), e.odds_coeff_idxs.end());
    }
    for (auto &x : coeffs) {
      check_cycles(x);
    }
  }
}

}  // namespace

std::vector<CoefficientPtr>::iterator Model::find_coefficient(const Coefficient::Id& cid) {
  return find_if(begin(coeffs), end(coeffs), [&](const CoefficientPtr &coeff) {
    return coeff->label == cid.name and coeff->kind == cid.kind
           and coeff->distribution == cid.dist and coeff->info == cid.info;
  });
}

size_t Model::register_coefficient(
    const unordered_map<string, ModelSpec::Variable>& variable_map,
    string id,
    size_t experiment)
{
  // Register coefficient if it doesn't already exist and return its index in
  // the coeffs vector.
  auto do_registration = [this](
      const Coefficient::Id& cid,
      size_t _G,
      size_t _T,
      size_t _S,
      std::function<void(size_t)> on_add
      ) -> size_t {
    auto it = find_coefficient(cid);
    size_t idx;
    if (it != end(coeffs)) {
      idx = distance(begin(coeffs), it);
    } else {
      idx = coeffs.size();
      LOG(debug) << "Adding new coefficient for " << cid.name << " (" << idx
                 << ").";
      coeffs.emplace_back(std::make_shared<Coefficient>(_G, _T, _S, cid));
      on_add(idx);
    }
    return idx;
  };

  auto register_fixed = [&](double value) {
    Coefficient::Id cid{
      .name = id,
      .kind = Coefficient::Kind::scalar,
      .dist = Coefficient::Distribution::fixed,
      .info = CovariateInformation{},
    };
    return do_registration(
        cid, 0, 0, 0, [&](size_t idx) { coeffs[idx]->get_raw(0, 0, 0) = value; });
  };

  auto register_random = [&]() {
    auto it = variable_map.find(id);
    if (it == variable_map.end()) {
      throw runtime_error("Unknown variable id '" + id + "'.");
    }
    auto variable = it->second;

    auto info = get_covariate_info(design, variable->covariates, experiment);
    auto kind = determine_kind(variable->covariates);

    if (variable->distribution == nullptr) {
      auto dist = parameters.default_distribution;
      LOG(verbose) << id
                   << " does not have a distribution specification. Using "
                   << to_string(dist) << " as per defaults.";
      variable->distribution = make_shared<Distribution>(dist,
          vector<string> {
              remove_trailing_zeros(to_string(
                    parameters.hyperparameters.get_param(dist, 0))),
              remove_trailing_zeros(to_string(
                    parameters.hyperparameters.get_param(dist, 1))),
          });
    }

    Coefficient::Id cid{
      .name = id,
      .kind = kind,
      .dist = variable->distribution->type,
      .info = info,
    };

    size_t idx = do_registration(
        cid, G, T, experiments[experiment].S, [&](size_t _idx) {
          size_t i = 1;
          for (auto& prior : variable->distribution->arguments) {
            size_t prior_idx = register_coefficient(
                variable_map, prior, experiment);
            coeffs[_idx]->prior_idxs.push_back(prior_idx);
            LOG(debug) << "Prior " << i++ << " of " << id << " (" << _idx
                       << ") is " << prior << " (" << prior_idx << ").";
          }
        });

    if (variable->distribution->type == Coefficient::Distribution::gp) {
      auto gp_kind = kind & ~Coefficient::Kind::spot;

      // Create or update coordinate system
      auto gp_coord_info = drop_covariates(
          info, design, { DesignNS::spot_label, DesignNS::section_label });
      auto gp_coord_id = Coefficient::Id{
        .name = id + "-gp-coord",
        .kind = gp_kind,
        .dist = Coefficient::Distribution::gp_coord,
        .info = gp_coord_info,
      };
      auto gp_coord_idx
          = do_registration(gp_coord_id, G, T, 0, [](size_t _gp_coord_idx) {
              LOG(debug) << "Added new GP coordinate system (" << _gp_coord_idx
                         << ").";
            });
      LOG(debug) << "Updating GP coordinate system (" << gp_coord_idx << ").";
      coeffs[gp_coord_idx]->experiment_idxs.push_back(experiment);
      coeffs[gp_coord_idx]->prior_idxs.push_back(idx);

      // Create or update GP proxy
      auto gp_proxy_info = drop_covariates(info, design,
          { DesignNS::spot_label, DesignNS::section_label,
              DesignNS::coordsys_label });
      auto gp_id = Coefficient::Id{
        .name = id + "-gp-proxy",
        .kind = gp_kind,
        .dist = Coefficient::Distribution::gp_proxy,
        .info = gp_proxy_info,
      };
      auto gp_proxy_idx
          = do_registration(gp_id, G, T, 0, [](size_t _gp_proxy_idx) {
              LOG(debug) << "Added new GP proxy (" << _gp_proxy_idx << ").";
            });
      LOG(debug) << "Updating GP proxy (" << gp_proxy_idx << ").";
      Coefficient& gp_coeff = *coeffs[gp_proxy_idx];
      gp_coeff.experiment_idxs.push_back(experiment);
      if (std::find(begin(gp_coeff.prior_idxs), end(gp_coeff.prior_idxs),
              gp_coord_idx)
          == end(gp_coeff.prior_idxs)) {
        gp_coeff.prior_idxs.push_back(gp_coord_idx);
      }
    }

    return idx;
  };

  double value;
  try {
    value = stod(id);
  } catch (const invalid_argument&) {
    return register_random();
  }
  return register_fixed(value);
}

void Model::add_covariates(const ModelSpec &model_spec) {
  auto rate_variables = collect_variables(model_spec.rate_expr);
  auto odds_variables = collect_variables(model_spec.odds_expr);
  for (size_t e = 0; e < E; ++e) {
    LOG(debug) << "Registering coefficients for experiment " << e;

    for(auto &variable: rate_variables) {
      auto idx = register_coefficient(model_spec.variables, variable->full_id(), e);
      coeffs[idx]->experiment_idxs.push_back(e);
      experiments[e].rate_coeff_idxs.push_back(idx);
    }

    for(auto &variable: odds_variables) {
      auto idx = register_coefficient(model_spec.variables, variable->full_id(), e);
      coeffs[idx]->experiment_idxs.push_back(e);
      experiments[e].odds_coeff_idxs.push_back(idx);
    }
  }
}

void Model::add_gp_proxies() {
  LOG(debug) << "Constructing GP proxies";
  for (size_t idx = 0; idx < coeffs.size(); ++idx)
    if (coeffs[idx]->distribution == Coefficient::Distribution::gp_proxy) {
      LOG(debug) << "Constructing GP proxy " << idx << ": " << *coeffs[idx];
      for (auto &coord_coeff_idx : coeffs[idx]->prior_idxs) {
        assert(coeffs[coord_coeff_idx]->distribution
               == Coefficient::Distribution::gp_coord);
        LOG(debug) << "using coordinate system coefficient " << coord_coeff_idx
                   << ": " << *coeffs[coord_coeff_idx];
        auto &coord_coeff = *coeffs[coord_coeff_idx];
        auto exp_idxs = coord_coeff.experiment_idxs;
        auto prior_idxs = coord_coeff.prior_idxs;
        size_t n = 0;
        for (size_t e : exp_idxs)
          n += experiments[e].S;
        size_t ncol = experiments[*exp_idxs.begin()].coords.cols();
        LOG(debug) << "n = " << n;
        Matrix m = Matrix::Zero(n, ncol);
        size_t i = 0;
        for (size_t e : exp_idxs) {
          for (size_t s = 0; s < experiments[e].S; ++s)
            for (size_t j = 0; j < ncol; ++j)
              m(i + s, j) = experiments[e].coords(s, j);
          i += experiments[e].S;
        }
        LOG(debug) << "m.dimesions = " << m.rows() << "x" << m.cols();
        coeffs[coord_coeff_idx]->gp = make_shared<GP::GaussianProcess>(
            GP::GaussianProcess(m, parameters.gp.length_scale));
      }
    }
}

template <typename T>
using ExprPtr = std::shared_ptr<spec_parser::expression::Exp<T>>;

template <typename T>
void compile_expression_and_derivs(const ExprPtr<T> &expr,
                                   const std::string &tag) {
  spec_parser::expression::codegen(expr, tag);
  for (auto variable : collect_variables(expr)) {
    auto deriv_expr = deriv(variable, expr);
    spec_parser::expression::codegen(deriv_expr,
                                     tag + "-" + to_string(*variable));
  }
}

Model::Model(const vector<Counts> &c, size_t T_, const Design &design_,
             const ModelSpec &model_spec, const Parameters &parameters_)
    : G(max_row_number(c)),
      T(T_),
      E(0),
      S(0),
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
  JIT::init_runtime(module_name);

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

  add_covariates(model_spec);

  coeff_debug_dump("INITIAL");
  add_gp_proxies();
  coeff_debug_dump("BEFORE");
  remove_redundant_terms();
  coeff_debug_dump("AFTER");

  verify_model(*this);

  // TODO cov spot initialize spot scaling:
  // linear in number of counts, scaled so that mean = 1
}

void Model::coeff_debug_dump(const string &tag) const {
  size_t index = 0;
  for (auto coeff : coeffs)
    LOG(debug) << tag << " " << index++ << " " << coeff << ": "
               << coeff->info.to_string(design.covariates);
  auto fnc = [&](const string &s, size_t idx, size_t e) {
    LOG(debug) << tag << " " << s << " experiment " << e << " " << idx << " "
               << *coeffs[idx] << ": "
               << coeffs[idx]->info.to_string(design.covariates);
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
  using Kind = Coefficient::Kind;
  for (auto kind : {Kind::scalar, Kind::gene, Kind::type, Kind::spot,
                    Kind::gene_type, Kind::spot_type})
    remove_redundant_terms(kind);
}

// TODO covariates: add redundant term labels
void Model::remove_redundant_terms(Coefficient::Kind kind) {
  vector<vector<size_t>> cov2groups_rate(coeffs.size());
  for (size_t e = 0; e < E; ++e)
    for (auto idx : experiments[e].rate_coeff_idxs)
      if (coeffs[idx]->kind == kind)
        cov2groups_rate[idx].push_back(e);
  remove_redundant_terms_sub(cov2groups_rate);

  vector<vector<size_t>> cov2groups_odds(coeffs.size());
  for (size_t e = 0; e < E; ++e)
    for (auto idx : experiments[e].odds_coeff_idxs)
      if (coeffs[idx]->kind == kind)
        cov2groups_rate[idx].push_back(e);
  remove_redundant_terms_sub(cov2groups_odds);
}

void Model::remove_redundant_terms_sub(
    const vector<vector<size_t>> &cov2groups) {
  // TODO print warning in case coefficients are used in both rate and odds eqs
  auto redundant = find_redundant(cov2groups);
  sort(begin(redundant), end(redundant));
  reverse(begin(redundant), end(redundant));

  // drop redundant coefficients
  for (auto r : redundant) {
    LOG(debug) << "Removing " << r << ": " << *coeffs[r] << ": "
               << coeffs[r]->info.to_string(design.covariates);
    coeffs.erase(begin(coeffs) + r);
  }

  // fix prior_idxs for dropped redundant coefficients
  for (auto &coeff : coeffs) {
    auto &idxs = coeff->prior_idxs;
    for (auto r : redundant)
      idxs.erase(remove(begin(idxs), end(idxs), r), end(idxs));
    for (auto &idx : idxs)
      for (auto r : redundant)
        if (idx > r)
          idx--;
  }

  // fix experiment.rate_coeff_idxs and experiment.odds_coeff_idxs for dropped
  // redundant coefficients
  for (size_t e = 0; e < E; ++e) {
    for (auto idxs :
         {&experiments[e].rate_coeff_idxs, &experiments[e].odds_coeff_idxs}) {
      for (auto r : redundant)
        idxs->erase(remove(begin(*idxs), end(*idxs), r), end(*idxs));
      for (auto &idx : *idxs)
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
    auto cs = colSums<Vector>(contributions_gene_type);
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
      for (auto &coeff : coeffs) {
        vector<string> spot_names;
        if (coeff->spot_dependent())
          for (auto idx : coeff->experiment_idxs)
            spot_names.insert(begin(spot_names),
                              begin(experiments[idx].counts.col_names),
                              end(experiments[idx].counts.col_names));
        coeff->store(prefix + "covariate-" + storage_type(coeff->kind) + "-"
                        + coeff->label + "-"
                        + coeff->info.to_string(design.covariates)
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
    for (auto &coeff : coeffs) {
      coeff->restore(
          prefix + "covariate-" + storage_type(coeff->kind) + "-" + coeff->label
          + "-" + coeff->info.to_string(design.covariates) + FILENAME_ENDING);
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
                    = coeffs[experiments[e].rate_coeff_idxs[i]]->get_actual(
                        g, t, s);
              for (size_t i = 0; i < num_odds_coeffs; ++i)
                odds_coeff_arrays[t][i]
                    = coeffs[experiments[e].odds_coeff_idxs[i]]->get_actual(
                        g, t, s);

              rate(t) = std::exp(rate_fnc(rate_coeff_arrays[t].data()));
              odds(t) = std::exp(odds_fnc(odds_coeff_arrays[t].data()));
            }
            // TODO: optimization for counts == 0
            auto cnts
                = exp.sample_contributions_gene_spot(g, s, rate, odds, rng);
            for (size_t t = 0; t < T; ++t) {
              register_gradient(g, e, s, t, cnts, grad, rate, odds,
                                rate_coeff_arrays[t], odds_coeff_arrays[t]);
              double p = odds_to_prob(odds(t));
              score_ += log_negative_binomial(cnts[t], rate(t), p);
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

void Model::register_gradient_zero_count(size_t g, size_t e, size_t s,
                                         Model &gradient, const Matrix &rate_gt,
                                         const Matrix &rate_st,
                                         const Matrix &odds_gt,
                                         const Matrix &odds_st) const {
  // TODO
}

// calculate parameter's likelihood
double Model::param_likel() const {
  double score = 0;
  // TODO covariates likelihood
  /*
    const double a = parameters.hyperparameters.gamma_1;
    const double b = parameters.hyperparameters.gamma_2;
#pragma omp parallel for if (DO_PARALLEL)
    for (size_t g = 0; g < G; ++g)
      for (size_t t = 0; t < T; ++t)
        score += log_gamma_rate(gamma(g, t), a, b);

    const double a = parameters.hyperparameters.lambda_1;
    const double b = parameters.hyperparameters.lambda_2;
    for (auto &coord_sys : coordinate_systems)
      for (auto e : coord_sys.members)
#pragma omp parallel for if (DO_PARALLEL)
        for (size_t g = 0; g < G; ++g)
          for (size_t t = 0; t < T; ++t)
            score += log_gamma_rate(experiments[e].lambda(g, t), a, b);

    const double a = parameters.hyperparameters.beta_1;
    const double b = parameters.hyperparameters.beta_2;
    for (auto &coord_sys : coordinate_systems)
      for (auto e : coord_sys.members)
#pragma omp parallel for if (DO_PARALLEL)
        for (size_t g = 0; g < G; ++g)
          score += log_gamma_rate(experiments[e].beta(g), a, b);

    for (auto &coord_sys : coordinate_systems)
      for (auto e : coord_sys.members)
#pragma omp parallel for if (DO_PARALLEL)
        for (size_t s = 0; s < experiments[e].S; ++s)
          for (size_t t = 0; t < T; ++t) {
            const double a = experiments[e].field(s, t) * mix_prior.r(t);
            const double b = mix_prior.p(t);
            score += log_gamma_rate(experiments[e].theta(s, t), a, b);
          }

    const double a = parameters.hyperparameters.rho_1;
    const double b = parameters.hyperparameters.rho_2;
#pragma omp parallel for if (DO_PARALLEL)
    for (size_t g = 0; g < G; ++g)
      for (size_t t = 0; t < T; ++t) {
        const double no = negodds_rho(g, t);
        score += log_beta_neg_odds(no, a, b);
      }
  */

  return score;
}

void Model::gradient_update() {
  LOG(verbose) << "Performing gradient update iterations";

  auto fnc = [&](const Vector &x, Vector &grad) {
    if (((iter_cnt++) % parameters.report_interval) == 0) {
      const size_t iteration_num_digits
          = 1 + floor(log(parameters.grad_iterations) / log(10));
      store("iter" + to_string_embedded(iter_cnt - 1, iteration_num_digits)
            + "/");
    }

    from_vector(x.array());
    double score = 0;
    Model model_grad = compute_gradient(score);
    for (auto &coeff : model_grad.coeffs)
      LOG(debug) << coeff << " grad = " << Stats::summary(coeff->values);

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

  Vector x = vectorize().array();

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
      }
    } break;
    case Optimize::Method::Gradient: {
      double alpha = parameters.grad_alpha;
      for (size_t iter = 0; iter < parameters.grad_iterations; ++iter) {
        Vector grad;
        fx = fnc(x, grad);
        x = x + alpha * grad;
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

  from_vector(x.array());
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
