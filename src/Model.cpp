#include "Model.hpp"
#include <LBFGS.h>
#include <map>
#include "gamma_func.hpp"
#include "rprop.hpp"

using namespace std;

namespace STD {

vector<double> unpack_values(
    const vector<pair<Model::CovariateInformation, double>> &x) {
  const size_t n = x.size();
  vector<double> v(n);
  for (size_t i = 0; i < n; ++i)
    v[i] = x[i].second;
  return v;
}

vector<string> unpack_labels(
    const vector<pair<Model::CovariateInformation, double>> &x,
    const Covariates &covariates) {
  const size_t n = x.size();
  vector<string> v(n);
  for (size_t i = 0; i < n; ++i)
    v[i] = x[i].first.to_string(covariates);
  return v;
}

void pack_values(const vector<double> &v,
                 vector<pair<Model::CovariateInformation, double>> &x) {
  const size_t n = x.size();
  for (size_t i = 0; i < n; ++i)
    x[i].second = v[i];
}

Model::Model(const vector<Counts> &c, size_t T_, const Formula &formula_,
             const Design &design_, const Parameters &parameters_,
             bool same_coord_sys)
    : G(max_row_number(c)),
      T(T_),
      E(0),
      S(0),
      formula(formula_),
      design(design_),
      experiments(),
      parameters(parameters_),
      negodds_rho(Matrix::Ones(G, T)),
      mix_prior(sum_cols(c), T, parameters),
      contributions_gene_type(Matrix::Zero(G, T)),
      contributions_gene(Vector::Zero(G)) {
  size_t coord_sys = 0;
  for (auto &counts : c)
    add_experiment(counts, same_coord_sys ? 0 : coord_sys++);
  update_contributions();

  LOG(debug) << "Model G = " << G << " T = " << T << " E = " << E;

  for (auto &term : formula.terms) {
    LOG(debug) << "Treating next formula term.";
    vector<size_t> cov_idxs;  // indices of covariates in this term

    bool gene_dependent = false;
    bool type_dependent = false;

    // determine cov_idxs, gene_dependent and type_dependent
    for (auto &covariate_label : term) {
      LOG(debug) << "Treating covariate label: " << covariate_label;
      if (to_lower(covariate_label) == "gene")
        gene_dependent = true;
      else if (to_lower(covariate_label) == "type")
        type_dependent = true;
      else {
        auto cov_iter
            = find_if(begin(design.covariates), end(design.covariates),
                      [&](const Covariate &covariate) {
                        return covariate.label == covariate_label;
                      });
        if (cov_iter == end(design.covariates)) {
          throw(runtime_error("Error: a covariate mentioned in the formula '"
                              + covariate_label
                              + "' is not found in the design."));
        } else {
          cov_idxs.push_back(distance(begin(design.covariates), cov_iter));
        }
      }
    }

    LOG(debug) << "gene_dependent = " << gene_dependent;
    LOG(debug) << "type_dependent = " << type_dependent;

    map<vector<size_t>, size_t> covvalues2idx;
    for (size_t e = 0; e < E; ++e) {
      vector<size_t> cov_values;
      for (auto &cov_idx : cov_idxs)
        cov_values.push_back(
            design.dataset_specifications[e].covariate_values[cov_idx]);
      auto iter = covvalues2idx.find(cov_values);
      size_t idx;
      if (iter != end(covvalues2idx)) {
        idx = iter->second;
        LOG(debug) << "Found previous covariate value combination: " << idx;
      } else {
        CovariateInformation info = {cov_idxs, cov_values};
        // this covariate value combination was not previously used
        if (gene_dependent and type_dependent) {
          idx = covariates_gene_type.size();
          LOG(debug) << "Creating new " << G << "x" << T
                     << " gene-type matrix: " << idx;
          covariates_gene_type.push_back({info, Matrix::Ones(G, T)});
        } else if (gene_dependent and not type_dependent) {
          idx = covariates_gene.size();
          LOG(debug) << "Creating new " << G << " gene vector: " << idx;
          covariates_gene.push_back({info, Vector::Ones(G)});
        } else if (not gene_dependent and type_dependent) {
          idx = covariates_type.size();
          LOG(debug) << "Creating new " << T << " type vector: " << idx;
          covariates_type.push_back({info, Vector::Ones(T)});
        } else if (not gene_dependent and not type_dependent) {
          idx = covariates_scalar.size();
          LOG(debug) << "Creating new scalar: " << idx;
          covariates_scalar.push_back({info, 1});
        }
        covvalues2idx[cov_values] = idx;
      }
      if (gene_dependent and type_dependent)
        experiments[e].covariates_gene_type.push_back(idx);
      else if (gene_dependent and not type_dependent)
        experiments[e].covariates_gene.push_back(idx);
      else if (not gene_dependent and type_dependent)
        experiments[e].covariates_type.push_back(idx);
      else if (not gene_dependent and not type_dependent)
        experiments[e].covariates_scalar.push_back(idx);
    }
  }

  remove_redundant_terms();

  {
    // TODO covariates initialize
    for (auto &x : covariates_scalar)
      x.second = 1;
    // x.second = exp(0.1 * std::normal_distribution<double>()(EntropySource::rng));
    for (auto &covariate : covariates_gene)
      for (auto &x : covariate.second)
        x = 1;
    // x = exp(0.1 * std::normal_distribution<double>()(EntropySource::rng));
    for (auto &covariate : covariates_type)
      for (auto &x : covariate.second)
        x = 1;
    // x = exp(0.1 * std::normal_distribution<double>()(EntropySource::rng));
    for (auto &covariate : covariates_gene_type)
      for (auto &x : covariate.second)
        x = exp(0.1 * std::normal_distribution<double>()(EntropySource::rng));
  }

  initialize_coordinate_systems(1);

  enforce_positive_parameters(parameters.min_value);
}

vector<size_t> find_redundant(const vector<vector<size_t>> &v) {
  using inv_map_t = multimap<vector<size_t>, size_t>;
  inv_map_t m;
  vector<size_t> redundant;
  for (size_t i = 0; i < v.size(); ++i) {
    auto key = v[i];
    pair<vector<size_t>, size_t> entry = {v[i], i};
    m.insert(entry);
    if (m.count(key) > 1)
      redundant.push_back(i);
  }
  return redundant;
}

// TODO covariates: add redundant term labels
void Model::remove_redundant_terms() {
  {
    vector<vector<size_t>> cov2groups(covariates_scalar.size());
    for (size_t e = 0; e < E; ++e)
      for (auto x : experiments[e].covariates_scalar)
        cov2groups[x].push_back(e);
    auto redundant = find_redundant(cov2groups);
    sort(begin(redundant), end(redundant));  // needed?
    size_t removed = 0;
    for (auto r : redundant) {
      LOG(verbose) << "Removing scalar covariate " << r << ": "
                   << covariates_scalar[r - removed].first.to_string(
                          design.covariates);
      covariates_scalar.erase(begin(covariates_scalar) + r - removed);
      removed++;
    }
    for (size_t e = 0; e < E; ++e) {
      for (auto r : redundant)
        experiments[e].covariates_scalar.erase(
            remove(begin(experiments[e].covariates_scalar),
                   end(experiments[e].covariates_scalar), r),
            end(experiments[e].covariates_scalar));
      for (auto &x : experiments[e].covariates_scalar)
        for (auto r : redundant)
          if (x > r)
            x--;
    }
  }

  {
    vector<vector<size_t>> cov2groups(covariates_gene.size());
    for (size_t e = 0; e < E; ++e)
      for (auto x : experiments[e].covariates_gene)
        cov2groups[x].push_back(e);
    auto redundant = find_redundant(cov2groups);
    sort(begin(redundant), end(redundant));  // needed?
    size_t removed = 0;
    for (auto r : redundant) {
      LOG(verbose) << "Removing gene-dependent covariate " << r << ": "
                   << covariates_gene[r - removed].first.to_string(
                          design.covariates);
      covariates_gene.erase(begin(covariates_gene) + r - removed);
      removed++;
    }
    for (size_t e = 0; e < E; ++e) {
      for (auto r : redundant)
        experiments[e].covariates_gene.erase(
            remove(begin(experiments[e].covariates_gene),
                   end(experiments[e].covariates_gene), r),
            end(experiments[e].covariates_gene));
      for (auto &x : experiments[e].covariates_gene)
        for (auto r : redundant)
          if (x > r)
            x--;
    }
  }

  {
    vector<vector<size_t>> cov2groups(covariates_type.size());
    for (size_t e = 0; e < E; ++e)
      for (auto x : experiments[e].covariates_type)
        cov2groups[x].push_back(e);
    auto redundant = find_redundant(cov2groups);
    sort(begin(redundant), end(redundant));  // needed?
    size_t removed = 0;
    for (auto r : redundant) {
      LOG(verbose) << "Removing type-dependent covariate " << r << ": "
                   << covariates_type[r - removed].first.to_string(
                          design.covariates);
      covariates_type.erase(begin(covariates_type) + r - removed);
      removed++;
    }
    for (size_t e = 0; e < E; ++e) {
      for (auto r : redundant)
        experiments[e].covariates_type.erase(
            remove(begin(experiments[e].covariates_type),
                   end(experiments[e].covariates_type), r),
            end(experiments[e].covariates_type));
      for (auto &x : experiments[e].covariates_type)
        for (auto r : redundant)
          if (x > r)
            x--;
    }
  }

  {
    vector<vector<size_t>> cov2groups(covariates_gene_type.size());
    for (size_t e = 0; e < E; ++e)
      for (auto x : experiments[e].covariates_gene_type)
        cov2groups[x].push_back(e);
    auto redundant = find_redundant(cov2groups);
    sort(begin(redundant), end(redundant));  // needed?
    size_t removed = 0;
    for (auto r : redundant) {
      LOG(verbose) << "Removing gene- and type-dependent covariate " << r
                   << ": "
                   << covariates_gene_type[r - removed].first.to_string(
                          design.covariates);
      covariates_gene_type.erase(begin(covariates_gene_type) + r - removed);
      removed++;
    }
    for (size_t e = 0; e < E; ++e) {
      for (auto r : redundant)
        experiments[e].covariates_gene_type.erase(
            remove(begin(experiments[e].covariates_gene_type),
                   end(experiments[e].covariates_gene_type), r),
            end(experiments[e].covariates_gene_type));
      for (auto &x : experiments[e].covariates_gene_type)
        for (auto r : redundant)
          if (x > r)
            x--;
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
  auto factor_names = form_factor_names(T);
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
                 parameters.compression_mode, gene_names, factor_names, order);

#pragma omp section
    {
      write_vector(unpack_values(covariates_scalar),
                   prefix + "covariate-scalar" + FILENAME_ENDING,
                   parameters.compression_mode,
                   unpack_labels(covariates_scalar, design.covariates));
    }
#pragma omp section
    for (size_t i = 0; i < covariates_gene.size(); ++i)
      write_vector(covariates_gene[i].second,
                   prefix + "covariate-gene-" + to_string_embedded(i, 2) + "_"
                       + covariates_gene[i].first.to_string(design.covariates)
                       + FILENAME_ENDING,
                   parameters.compression_mode, gene_names);
#pragma omp section
    for (size_t i = 0; i < covariates_type.size(); ++i)
      write_vector(covariates_type[i].second,
                   prefix + "covariate-type-" + to_string_embedded(i, 2) + "_"
                       + covariates_gene[i].first.to_string(design.covariates)
                       + FILENAME_ENDING,
                   parameters.compression_mode, factor_names);
#pragma omp section
    for (size_t i = 0; i < covariates_gene_type.size(); ++i)
      write_matrix(
          covariates_gene_type[i].second,
          prefix + "covariate-gene-type-" + to_string_embedded(i, 2) + "_"
              + covariates_gene[i].first.to_string(design.covariates)
              + FILENAME_ENDING,
          parameters.compression_mode, gene_names, factor_names, order);
#pragma omp section
    write_matrix(negodds_rho, prefix + "feature-negodds_rho" + FILENAME_ENDING,
                 parameters.compression_mode, gene_names, factor_names, order);
#pragma omp section
    {
      const size_t C = coordinate_systems.size();
      const size_t num_digits = 1 + floor(log(C) / log(10));
      for (size_t c = 0; c < C; ++c) {
        vector<string> rn;
        for (size_t n = 0; n < coordinate_systems[c].N; ++n)
          rn.push_back(to_string(n));
        write_matrix(coordinate_systems[c].field,
                     prefix + "field" + to_string_embedded(c, num_digits)
                         + FILENAME_ENDING,
                     parameters.compression_mode, rn, factor_names, order);
      }
    }
#pragma omp section
    mix_prior.store(prefix + "theta_prior", factor_names, order);
#pragma omp section
    write_matrix(contributions_gene_type,
                 prefix + "contributions_gene_type" + FILENAME_ENDING,
                 parameters.compression_mode, gene_names, factor_names, order);
#pragma omp section
    write_vector(contributions_gene,
                 prefix + "contributions_gene" + FILENAME_ENDING,
                 parameters.compression_mode, gene_names);
#pragma omp section
    {
      auto print_field_matrix = [&](const string &path, const Vector &weights) {
        ofstream ofs(path);
        ofs << "coord_sys\tpoint_idx";
        for (size_t d = 0; d < coordinate_systems[0].mesh.dim; ++d)
          ofs << "\tx" << d;
        for (size_t t = 0; t < T; ++t)
          ofs << "\tFactor " << t + 1;
        ofs << endl;

        size_t coord_sys_idx = 0;
        for (size_t c = 0; c < coordinate_systems.size(); ++c) {
          for (size_t n = 0; n < coordinate_systems[c].N; ++n) {
            // print coordinate system index
            ofs << coord_sys_idx << "\t" << n;
            // print coordinate
            for (size_t d = 0; d < coordinate_systems[c].mesh.dim; ++d)
              ofs << "\t" << coordinate_systems[c].mesh.points[n][d];
            // print weighted field values of factors in order
            for (size_t t = 0; t < T; ++t)
              ofs << "\t"
                  << coordinate_systems[c].field(n, order[t])
                         * weights[order[t]];
            ofs << endl;
          }
          coord_sys_idx++;
        }
      };
      Vector w = Vector::Ones(T);
      // print un-weighted field
      print_field_matrix(prefix + "field" + FILENAME_ENDING, w);

      // TODO covariates store expected fields
      /*
      // NOTE we ignore local features and local baseline
      w = mix_prior.r.array() / mix_prior.p.array();
      Vector meanColSums = colSums<Vector>(gamma.array() / negodds_rho.array());
      for (size_t t = 0; t < T; ++t)
        w(t) *= meanColSums(t);
      // print weighted field
      print_field_matrix(prefix + "expfield" + FILENAME_ENDING, w);
      */
    }
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
  vector<double> v = parse_file<std::vector<double>>(
      prefix + "covariate-scalar" + FILENAME_ENDING,
      read_vector<std::vector<double>>, "\t");
  pack_values(v, covariates_scalar);
  for (size_t i = 0; i < covariates_gene.size(); ++i)
    covariates_gene[i].second = parse_file<Vector>(
        prefix + "covariate-gene-" + to_string_embedded(i, 2) + "_"
            + covariates_gene[i].first.to_string(design.covariates)
            + FILENAME_ENDING,
        read_vector<Vector>, "\t");
  for (size_t i = 0; i < covariates_type.size(); ++i)
    covariates_type[i].second = parse_file<Vector>(
        prefix + "covariate-type-" + to_string_embedded(i, 2) + "_"
            + covariates_gene[i].first.to_string(design.covariates)
            + FILENAME_ENDING,
        read_vector<Vector>, "\t");
  for (size_t i = 0; i < covariates_gene_type.size(); ++i)
    covariates_gene_type[i].second = parse_file<Matrix>(
        prefix + "covariate-gene-type-" + to_string_embedded(i, 2) + "_"
            + covariates_gene[i].first.to_string(design.covariates)
            + FILENAME_ENDING,
        read_matrix, "\t");

  negodds_rho = parse_file<Matrix>(
      prefix + "feature-negodds_rho" + FILENAME_ENDING, read_matrix, "\t");

  {
    const size_t C = coordinate_systems.size();
    const size_t num_digits = 1 + floor(log(C) / log(10));
    for (size_t c = 0; c < C; ++c)
      coordinate_systems[c].field = parse_file<Matrix>(
          prefix + "field" + to_string_embedded(c, num_digits)
              + FILENAME_ENDING,
          read_matrix, "\t");
  }

  mix_prior.restore(prefix + "theta_prior");

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

Matrix Model::field_fitness_posterior_gradient() const {
  Matrix grad(S, T);
  size_t cumul = 0;
  for (auto &experiment : experiments) {
    grad.middleRows(cumul, experiment.S)
        = experiment.field_fitness_posterior_gradient();
    cumul += experiment.S;
  }
  return grad;
}

void Model::set_zero() {
  for (auto &y : covariates_scalar)
    y.second = 0;
  for (auto &y : covariates_gene)
    y.second.setZero();
  for (auto &y : covariates_type)
    y.second.setZero();
  for (auto &y : covariates_gene_type)
    y.second.setZero();

  negodds_rho.setZero();
  for (auto &coord_sys : coordinate_systems)
    coord_sys.field.setZero();
  mix_prior.r.setZero();
  mix_prior.p.setZero();
  for (auto &experiment : experiments)
    experiment.set_zero();
}

size_t Model::size() const {
  size_t s = 0;

  if (parameters.targeted(Target::covariates_scalar))
    s += covariates_scalar.size();
  if (parameters.targeted(Target::covariates_gene))
    for (auto &y : covariates_gene)
      s += y.second.size();
  if (parameters.targeted(Target::covariates_type))
    for (auto &y : covariates_type)
      s += y.second.size();
  if (parameters.targeted(Target::covariates_gene_type))
    for (auto &y : covariates_gene_type)
      s += y.second.size();

  if (parameters.targeted(Target::gamma_prior))
    s += 2;
  if (parameters.targeted(Target::rho_prior))
    s += 2;
  if (parameters.targeted(Target::rho))
    s += negodds_rho.size();
  if (parameters.targeted(Target::theta_prior))
    s += mix_prior.r.size() + mix_prior.p.size();
  if (parameters.targeted(Target::field))
    for (auto &coord_sys : coordinate_systems)
      s += coord_sys.field.size();
  for (auto &experiment : experiments)
    s += experiment.size();
  return s;
}

Vector Model::vectorize() const {
  Vector v(size());
  auto iter = begin(v);

  if (parameters.targeted(Target::covariates_scalar))
    for (auto &y : covariates_scalar)
      *iter++ = y.second;
  if (parameters.targeted(Target::covariates_gene))
    for (auto &y : covariates_gene)
      for (auto &z : y.second)
        *iter++ = z;
  if (parameters.targeted(Target::covariates_type))
    for (auto &y : covariates_type)
      for (auto &z : y.second)
        *iter++ = z;
  if (parameters.targeted(Target::covariates_gene_type))
    for (auto &y : covariates_gene_type)
      for (auto &z : y.second)
        *iter++ = z;

  if (parameters.targeted(Target::gamma_prior)) {
    *iter++ = parameters.hyperparameters.gamma_1;
    *iter++ = parameters.hyperparameters.gamma_2;
  }

  if (parameters.targeted(Target::rho_prior)) {
    *iter++ = parameters.hyperparameters.rho_1;
    *iter++ = parameters.hyperparameters.rho_2;
  }

  if (parameters.targeted(Target::rho))
    for (auto &x : negodds_rho)
      *iter++ = x;

  if (parameters.targeted(Target::field))
    for (auto &coord_sys : coordinate_systems)
      for (auto &x : coord_sys.field)
        *iter++ = x;

  if (parameters.targeted(Target::theta_prior)) {
    for (auto &x : mix_prior.r)
      *iter++ = x;
    for (auto &x : mix_prior.p)
      *iter++ = x;
  }

  for (auto &experiment : experiments)
    for (auto &x : experiment.vectorize())
      *iter++ = x;

  assert(iter == end(v));

  return v;
}

Model Model::compute_gradient(double &score) const {
  LOG(verbose) << "Computing gradient";

  std::vector<Matrix> gt;
  std::vector<Matrix> st;

  for (auto &coord_sys : coordinate_systems)
    for (auto e : coord_sys.members) {
      gt.push_back(experiments[e].compute_gene_type_table());
      st.push_back(experiments[e].compute_spot_type_table());
    }

  score = 0;
  Model gradient = *this;
  gradient.set_zero();
  gradient.contributions_gene_type.setZero();
  for (auto &experiment : gradient.experiments) {
    experiment.contributions_spot_type.setZero();
    experiment.contributions_gene_type.setZero();
  }
  if (parameters.targeted(Target::covariates_scalar)
      or parameters.targeted(Target::covariates_gene)
      or parameters.targeted(Target::covariates_type)
      or parameters.targeted(Target::covariates_gene_type)
      or parameters.targeted(Target::rho) or parameters.targeted(Target::theta)
      or parameters.targeted(Target::spot))
#pragma omp parallel if (DO_PARALLEL)
  {
    Model grad = gradient;
    auto rng = EntropySource::rngs[omp_get_thread_num()];
    double score_ = 0;

#pragma omp for
    for (size_t g = 0; g < G; ++g)
      for (auto &coord_sys : coordinate_systems)
        for (auto e : coord_sys.members)
          for (size_t s = 0; s < experiments[e].S; ++s)
            if (RandomDistribution::Uniform(rng)
                >= parameters.dropout_gene_spot) {
              auto cnts = experiments[e].sample_contributions_gene_spot(
                  g, s, gt[e], st[e], rng);
              for (size_t t = 0; t < T; ++t)
                score += log_negative_binomial(
                    cnts[t], gt[e](g, t) * st[e](s, t),
                    neg_odds_to_prob(negodds_rho(g, t)));
              register_gradient(g, e, s, cnts, grad, gt[e], st[e]);
            }
#pragma omp critical
    {
      gradient = gradient + grad;
      score += score_;
    }
  }

  gradient.update_contributions();

  if (parameters.targeted(Target::field))
    for (size_t c = 0; c < coordinate_systems.size(); ++c)
      score
          += field_gradient(coordinate_systems[c], coordinate_systems[c].field,
                            gradient.coordinate_systems[c].field);

  if (parameters.targeted(Target::gamma_prior))
    score += compute_gradient_gamma_prior(gradient);

  if (parameters.targeted(Target::rho_prior))
    score += compute_gradient_rho_prior(gradient);

  if (not parameters.ignore_priors)
    finalize_gradient(gradient);
  score += param_likel();

  return gradient;
}

double Model::compute_gradient_gamma_prior(Model &gradient) const {
  double score = 0;

  double a = gradient.parameters.hyperparameters.gamma_1;
  double b = gradient.parameters.hyperparameters.gamma_2;

  gradient.parameters.hyperparameters.gamma_1 = 0;
  gradient.parameters.hyperparameters.gamma_2 = 0;

  const double hyper_alpha = 1;
  const double hyper_beta = 1;
  const double hyper_gamma = 1;
  const double hyper_delta = 1;

  // TODO covariates gamma prior
  /*
  for (size_t g = 0; g < G; ++g)
    for (size_t t = 0; t < T; ++t) {
      gradient.parameters.hyperparameters.gamma_1
          += a * (log(b) - digamma(a) + log(gamma(g, t)));
      gradient.parameters.hyperparameters.gamma_2 += a - b * gamma(g, t);
    }
    */

  gradient.parameters.hyperparameters.gamma_1
      += hyper_alpha - 1 - a * hyper_beta;
  gradient.parameters.hyperparameters.gamma_2
      += hyper_gamma - 1 - b * hyper_delta;

  // TODO compute score

  return score;
}

double Model::compute_gradient_rho_prior(Model &gradient) const {
  double score = 0;

  double a = gradient.parameters.hyperparameters.rho_1;
  double b = gradient.parameters.hyperparameters.rho_2;

  gradient.parameters.hyperparameters.rho_1 = 0;
  gradient.parameters.hyperparameters.rho_2 = 0;

  const double hyper_alpha = 1;
  const double hyper_beta = 1;
  const double hyper_gamma = 1;
  const double hyper_delta = 1;

  for (size_t g = 0; g < G; ++g)
    for (size_t t = 0; t < T; ++t) {
      gradient.parameters.hyperparameters.rho_1 += log(negodds_rho(g, t))
                                                   - log(1 + negodds_rho(g, t))
                                                   + digamma_diff(a, b);
      gradient.parameters.hyperparameters.rho_2
          += -log(1 + negodds_rho(g, t)) + digamma_diff(b, a);
    }

  gradient.parameters.hyperparameters.rho_1 += hyper_alpha - 1 - a * hyper_beta;
  gradient.parameters.hyperparameters.rho_2
      += hyper_gamma - 1 - b * hyper_delta;

  // TODO compute score

  return score;
}

void Model::register_gradient(size_t g, size_t e, size_t s, const Vector &cnts,
                              Model &gradient, const Matrix &gt,
                              const Matrix &st) const {
  for (size_t t = 0; t < T; ++t)
    gradient.experiments[e].contributions_gene_type(g, t) += cnts[t];
  for (size_t t = 0; t < T; ++t)
    gradient.experiments[e].contributions_spot_type(s, t) += cnts[t];

  for (size_t t = 0; t < T; ++t) {
    const double no = negodds_rho(g, t);
    const double p = neg_odds_to_prob(no);
    const double log_one_minus_p = odds_to_log_prob(no);
    const double r = gt(g, t) * st(s, t);
    const double k = cnts[t];
    const double term = r * (log_one_minus_p + digamma_diff(r, k));

    for (auto &y : gradient.experiments[e].covariates_scalar)
      gradient.covariates_scalar[y].second += term;
    for (auto &y : gradient.experiments[e].covariates_gene)
      gradient.covariates_gene[y].second(g) += term;
    for (auto &y : gradient.experiments[e].covariates_type)
      gradient.covariates_type[y].second(t) += term;
    for (auto &y : gradient.experiments[e].covariates_gene_type)
      gradient.covariates_gene_type[y].second(g, t) += term;
    gradient.experiments[e].theta(s, t) += term;
    gradient.experiments[e].spot(s) += term;

    gradient.negodds_rho(g, t) += p * (r + k) - k;
  }
}

void Model::finalize_gradient(Model &gradient) const {
  LOG(verbose) << "Finalizing gradient";

  {
    // TODO check covariates prior contribution to gradient
    const double a = parameters.hyperparameters.gamma_1;
    const double b = parameters.hyperparameters.gamma_2;
    if (parameters.targeted(Target::covariates_scalar))
      for (size_t i = 0; i < gradient.covariates_scalar.size(); ++i)
        gradient.covariates_scalar[i].second
            += (a - 1) - covariates_scalar[i].second * b;
    if (parameters.targeted(Target::covariates_gene))
      for (size_t i = 0; i < gradient.covariates_gene.size(); ++i)
        gradient.covariates_gene[i].second.array()
            += (a - 1) - covariates_gene[i].second.array() * b;
    if (parameters.targeted(Target::covariates_type))
      for (size_t i = 0; i < gradient.covariates_type.size(); ++i)
        gradient.covariates_type[i].second.array()
            += (a - 1) - covariates_type[i].second.array() * b;
    if (parameters.targeted(Target::covariates_gene_type))
      for (size_t i = 0; i < gradient.covariates_gene_type.size(); ++i)
        gradient.covariates_gene_type[i].second.array()
            += (a - 1) - covariates_gene_type[i].second.array() * b;
  }

  if (parameters.targeted(Target::rho)) {
    const double a = parameters.hyperparameters.rho_1;
    const double b = parameters.hyperparameters.rho_2;
#pragma omp parallel for if (DO_PARALLEL)
    for (size_t g = 0; g < G; ++g)
      for (size_t t = 0; t < T; ++t) {
        const double no = negodds_rho(g, t);
        const double p = neg_odds_to_prob(no);
        gradient.negodds_rho(g, t) += (a + b - 2) * p - a + 1;
      }
  }

  if (parameters.targeted(Target::theta))
    for (auto &coord_sys : coordinate_systems)
      for (auto e : coord_sys.members)
#pragma omp parallel for if (DO_PARALLEL)
        for (size_t s = 0; s < experiments[e].S; ++s)
          for (size_t t = 0; t < T; ++t) {
            const double a = experiments[e].field(s, t) * mix_prior.r(t);
            const double b = mix_prior.p(t);
            gradient.experiments[e].theta(s, t)
                += (a - 1) - experiments[e].theta(s, t) * b;
          }

  if (parameters.targeted(Target::spot)) {
    const double a = parameters.hyperparameters.spot_a;
    const double b = parameters.hyperparameters.spot_b;
    for (auto &coord_sys : coordinate_systems)
      for (auto e : coord_sys.members)
#pragma omp parallel for if (DO_PARALLEL)
        for (size_t s = 0; s < experiments[e].S; ++s)
          gradient.experiments[e].spot(s)
              += (a - 1) - experiments[e].spot(s) * b;
  }

  if (parameters.targeted(Target::theta_prior)) {
    gradient.mix_prior.r.setZero();
    for (auto &coord_sys : coordinate_systems)
      for (auto e : coord_sys.members)
        for (size_t t = 0; t < T; ++t) {
          const double r = mix_prior.r(t);
          const double no = mix_prior.p(t);
          const double p = neg_odds_to_prob(no);
          const double log_no = log(no);

          double x = 0;
          double y = 0;
#pragma omp parallel for if (DO_PARALLEL)
          for (size_t s = 0; s < experiments[e].S; ++s) {
            const double prod = r * experiments[e].field(s, t);
            x += (log(experiments[e].theta(s, t)) + log_no - digamma(prod))
                 * prod;
            y += (1 - p) * (p * experiments[e].theta(s, t) - prod);
          }
          gradient.mix_prior.r(t) += x;
          gradient.mix_prior.p(t) += y;
        }

    {
      const double a = parameters.hyperparameters.theta_r_1;
      const double b = parameters.hyperparameters.theta_r_2;
      for (size_t t = 0; t < T; ++t)
        gradient.mix_prior.r(t) += a - 1 - b * mix_prior.r(t);
    }

    {
      const double a = parameters.hyperparameters.theta_p_1;
      const double b = parameters.hyperparameters.theta_p_2;
      for (size_t t = 0; t < T; ++t) {
        const double no = mix_prior.p(t);
        const double p = neg_odds_to_prob(no);
        gradient.mix_prior.p(t) += (a + b - 2) * p - a + 1;
      }
    }
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
  */

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

  if (parameters.targeted(Target::spot)) {
    const double a = parameters.hyperparameters.spot_a;
    const double b = parameters.hyperparameters.spot_b;
    for (auto &coord_sys : coordinate_systems)
      for (auto e : coord_sys.members)
#pragma omp parallel for if (DO_PARALLEL)
        for (size_t s = 0; s < experiments[e].S; ++s)
          score += log_gamma_rate(experiments[e].spot(s), a, b);
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

  if (parameters.targeted(Target::theta_prior)) {
    {
      const double a = parameters.hyperparameters.theta_r_1;
      const double b = parameters.hyperparameters.theta_r_2;
      for (size_t t = 0; t < T; ++t)
        score += log_gamma_rate(mix_prior.r(t), a, b);
    }
    {
      const double a = parameters.hyperparameters.theta_p_1;
      const double b = parameters.hyperparameters.theta_p_2;
      for (size_t t = 0; t < T; ++t)
        score += log_beta_neg_odds(mix_prior.p(t), a, b);
    }
  }

  return score;
}

void Model::gradient_update() {
  LOG(verbose) << "Performing gradient update iteration";

  size_t iter_cnt = 0;

  auto fnc = [&](const Vector &x, Vector &grad) {
    if (((iter_cnt++) % parameters.report_interval) == 0) {
      const size_t iteration_num_digits
          = 1 + floor(log(parameters.grad_iterations) / log(10));
      store("iter" + to_string_embedded(iter_cnt - 1, iteration_num_digits)
            + "/");
    }

    from_log_vector(begin(x));
    enforce_positive_parameters(parameters.min_value);
    double score = 0;
    Model model_grad = compute_gradient(score);
    for (auto &y : model_grad.covariates_scalar)
      LOG(debug) << "scalar cov grad = " << y.second;
    for (auto &y : model_grad.covariates_gene)
      LOG(debug) << "gene cov grad " << Stats::summary(y.second);
    for (auto &y : model_grad.covariates_type)
      LOG(debug) << "type cov grad " << Stats::summary(y.second);
    for (auto &y : model_grad.covariates_gene_type)
      LOG(debug) << "gene_type cov grad " << Stats::summary(y.second);
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

      LOG(verbose) << "LBFGS performed " << niter << " iterations";
    } break;
  }
  LOG(verbose) << "Final f(x) = " << fx;

  from_log_vector(begin(x));
}

double Model::field_gradient(const CoordinateSystem &coord_sys,
                             const Matrix &field, Matrix &grad) const {
  LOG(debug) << "Computing field gradient";
  LOG(debug) << "field dim " << field.rows() << "x" << field.cols();
  double score = 0;

  Matrix fitness(coord_sys.S, T);
  size_t cumul = 0;
  for (auto member : coord_sys.members) {
    const size_t current_S = experiments[member].S;
    // no need to calculate fitness contribution to score
    // it is calculated in finalize_gradient()
    fitness.middleRows(cumul, current_S)
        = experiments[member].field_fitness_posterior_gradient();
    cumul += current_S;
  }

  LOG(debug) << "Fitness contribution to score: " << score;

  grad = Matrix::Zero(coord_sys.N, T);

  grad.topRows(coord_sys.S) = fitness;

  if (parameters.field_lambda_dirichlet != 0) {
    Matrix grad_dirichlet = Matrix::Zero(coord_sys.N, T);
    for (size_t t = 0; t < T; ++t) {
      grad_dirichlet.col(t)
          = coord_sys.mesh.grad_dirichlet_energy(Vector(field.col(t)));

      double s = coord_sys.mesh.sum_dirichlet_energy(Vector(field.col(t)));
      score -= s * parameters.field_lambda_dirichlet;
      LOG(debug) << "Smoothness contribution to score of factor " << t << ": "
                 << s;
    }
    grad -= grad_dirichlet * parameters.field_lambda_dirichlet;
  }

  if (parameters.field_lambda_laplace != 0) {
    Matrix grad_laplace = Matrix::Zero(coord_sys.N, T);
    for (size_t t = 0; t < T; ++t) {
      grad_laplace.col(t)
          = coord_sys.mesh.grad_sq_laplace_operator(Vector(field.col(t)));

      double s = coord_sys.mesh.sum_sq_laplace_operator(Vector(field.col(t)));
      score -= s * parameters.field_lambda_laplace;
      LOG(debug) << "Curvature contribution to score of factor " << t << ": "
                 << s;
    }
    grad -= grad_laplace * parameters.field_lambda_laplace;
  }

  LOG(debug) << "Field score: " << score;

  return score;
}

void Model::enforce_positive_parameters(double min_value) {
  {
    auto v = unpack_values(covariates_scalar);
    enforce_positive_and_warn("covariates_scalar", v, min_value,
                              parameters.warn_lower_limit);
    pack_values(v, covariates_scalar);
  }
  for (size_t i = 0; i < covariates_gene.size(); ++i)
    enforce_positive_and_warn("covariates_gene_" + to_string_embedded(i, 3),
                              covariates_gene[i].second, min_value,
                              parameters.warn_lower_limit);
  for (size_t i = 0; i < covariates_type.size(); ++i)
    enforce_positive_and_warn("covariates_type_" + to_string_embedded(i, 3),
                              covariates_type[i].second, min_value,
                              parameters.warn_lower_limit);
  for (size_t i = 0; i < covariates_gene_type.size(); ++i)
    enforce_positive_and_warn(
        "covariates_gene_type_" + to_string_embedded(i, 3),
        covariates_gene_type[i].second, min_value, parameters.warn_lower_limit);
  enforce_positive_and_warn("negodds_rho", negodds_rho, min_value,
                            parameters.warn_lower_limit);
  enforce_positive_and_warn("mix_prior_r", mix_prior.r, min_value,
                            parameters.warn_lower_limit);
  enforce_positive_and_warn("mix_prior_p", mix_prior.p, min_value,
                            parameters.warn_lower_limit);
  for (auto &coord_sys : coordinate_systems)
    enforce_positive_and_warn("field", coord_sys.field, min_value,
                              parameters.warn_lower_limit);
  for (auto &experiment : experiments)
    experiment.enforce_positive_parameters();
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

void Model::update_experiment_fields() {
  LOG(verbose) << "Updating experiment fields";
  for (auto &coord_sys : coordinate_systems) {
    size_t cumul = 0;
    for (auto member : coord_sys.members) {
      experiments[member].field
          = coord_sys.field.middleRows(cumul, experiments[member].S);
      cumul += experiments[member].S;
    }
  }
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

void Model::initialize_coordinate_systems(double v) {
  size_t coord_sys_idx = 0;
  for (auto &coord_sys : coordinate_systems) {
    size_t num_additional = parameters.mesh_additional;
    coord_sys.S = 0;
    for (auto &member : coord_sys.members)
      coord_sys.S += experiments[member].S;
    if (coord_sys.S == 0)
      num_additional = 0;
    coord_sys.N = coord_sys.S + num_additional;
    coord_sys.T = T;
    coord_sys.field = Matrix(coord_sys.N, T);
    coord_sys.field.fill(v);

    using Point = Vector;
    size_t dim = experiments[coord_sys.members[0]].coords.cols();
    vector<Point> pts;
    {
      Point pt(dim);
      for (auto &member : coord_sys.members)
        for (size_t s = 0; s < experiments[member].S; ++s) {
          for (size_t i = 0; i < dim; ++i)
            pt[i] = experiments[member].coords(s, i);
          pts.push_back(pt);
        }

      if (num_additional > 0) {
        Point mi = pts[0];
        Point ma = pts[0];
        for (auto &p : pts)
          for (size_t d = 0; d < dim; ++d) {
            if (p[d] < mi[d])
              mi[d] = p[d];
            if (p[d] > ma[d])
              ma[d] = p[d];
          }
        if (parameters.mesh_hull_distance <= 0) {
          Point mid = (ma + mi) / 2;
          Point half_diff = (ma - mi) / 2;
          mi = mid - half_diff * parameters.mesh_hull_enlarge;
          ma = mid + half_diff * parameters.mesh_hull_enlarge;
        } else {
          mi.array() -= parameters.mesh_hull_distance;
          ma.array() += parameters.mesh_hull_distance;
        }
        for (size_t i = 0; i < num_additional; ++i) {
          // TODO improve this
          // e.g. enlarge area, use convex hull instead of bounding box, etc
          bool ok = false;
          while (not ok) {
            for (size_t d = 0; d < dim; ++d)
              pt[d] = mi[d]
                      + (ma[d] - mi[d])
                            * RandomDistribution::Uniform(EntropySource::rng);
            for (size_t s = 0; s < coord_sys.S; ++s)
              if ((pt - pts[s]).norm() < parameters.mesh_hull_distance) {
                ok = true;
                break;
              }
          }

          pts.push_back(pt);
        }
      }
    }
    coord_sys.mesh
        = Mesh(dim, pts,
               parameters.output_directory + "coordsys"
                   + to_string_embedded(coord_sys_idx, EXPERIMENT_NUM_DIGITS));
    coord_sys_idx++;
    S += coord_sys.S;
  }
}

void Model::add_experiment(const Counts &counts, size_t coord_sys) {
  experiments.push_back({this, counts, T, parameters});
  E++;
  while (coordinate_systems.size() <= coord_sys)
    coordinate_systems.push_back({});
  coordinate_systems[coord_sys].members.push_back(E - 1);
}

ostream &operator<<(ostream &os, const Model &model) {
  os << "Spatial Transcriptome Deconvolution "
     << "G = " << model.G << " "
     << "T = " << model.T << " "
     << "E = " << model.E << endl;

  /*
  if (verbosity >= Verbosity::debug) {
    print_matrix_head(os, model.features.matrix, "Φ");
    os << model.phi;
  }
  */
  for (auto &experiment : model.experiments)
    os << experiment;

  return os;
}

string Model::CovariateInformation::to_string(
    const Covariates &covariates) const {
  string s;
  for (size_t i = 0; i < idxs.size(); ++i) {
    if (i > 0)
      s += ",";
    if (covariates[idxs[i]].label == DesignNS::unit_label)
      s += "intercept";
    else
      s += covariates[idxs[i]].label + "="
           + covariates[idxs[i]].values[vals[i]];
  }
  if (idxs.size() == 0)
    s = "global";
  return s;
}

Model operator+(const Model &a, const Model &b) {
  Model model = a;

  model.contributions_gene_type += b.contributions_gene_type;
  model.contributions_gene += b.contributions_gene;
  for (size_t i = 0; i < a.covariates_scalar.size(); ++i)
    model.covariates_scalar[i].second += b.covariates_scalar[i].second;
  for (size_t i = 0; i < a.covariates_gene.size(); ++i)
    model.covariates_gene[i].second.array()
        += b.covariates_gene[i].second.array();
  for (size_t i = 0; i < a.covariates_type.size(); ++i)
    model.covariates_type[i].second.array()
        += b.covariates_type[i].second.array();
  for (size_t i = 0; i < a.covariates_gene_type.size(); ++i)
    model.covariates_gene_type[i].second.array()
        += b.covariates_gene_type[i].second.array();
  model.negodds_rho += b.negodds_rho;
  for (size_t e = 0; e < model.E; ++e)
    model.experiments[e] = model.experiments[e] + b.experiments[e];

  return model;
}
}
