#ifndef MODEL_HPP
#define MODEL_HPP

#include <map>
#include "Experiment.hpp"

namespace PoissonFactorization {

const double local_phi_scaling_factor = 50;
const int EXPERIMENT_NUM_DIGITS = 4;

template <typename Type>
struct Model {
  using features_t = typename Type::features_t;
  using weights_t = typename Type::weights_t;
  using experiment_t = Experiment<Type>;

  // TODO consider const
  /** number of genes */
  size_t G;
  /** number of factors */
  size_t T;
  /** number of experiments */
  size_t E;

  size_t iteration;

  std::vector<experiment_t> experiments;

  Parameters parameters;

  /** hidden contributions to the count data due to the different factors */
  Matrix contributions_gene_type;
  Vector contributions_gene;
  Matrix prev_g_r, prev_g_p;
  IMatrix prev_sign_r, prev_sign_p;

  /** factor loading matrix */
  features_t features;
  struct CoordinateSystem {
    // std::vector<Matrix> coords;
    std::vector<size_t> members;
  };
  std::vector<CoordinateSystem> coordinate_systems;
  std::map<std::pair<size_t, size_t>, Matrix> kernels;

  template <typename Fnc1, typename Fnc2>
  Vector get_low_high(size_t coord_sys_idx, Float init, Fnc1 fnc1,
                      Fnc2 fnc2) const;
  Vector get_low(size_t coord_sys_idx) const;
  Vector get_high(size_t coord_sys_idx) const;

  void predict_field(std::ofstream &ofs, size_t coord_sys_idx) const;

  typename weights_t::prior_type mix_prior;

  Model(const std::vector<Counts> &data, size_t T,
        const Parameters &parameters, bool same_coord_sys);

  void enforce_positive_parameters();

  void store(const std::string &prefix, bool reorder = true) const;
  void restore(const std::string &prefix);
  void perform_pairwise_dge(const std::string &prefix) const;
  void perform_local_dge(const std::string &prefix) const;

  /** sample each of the variables from their conditional posterior */
  void gibbs_sample(bool report_likelihood);

  void sample_contributions(bool update_phi_prior);

  void sample_global_theta_priors();
  void sample_fields();

  double log_likelihood() const;
  double log_likelihood_poisson_counts() const;

  inline Float &phi(size_t g, size_t t) { return features.matrix(g, t); };
  inline Float phi(size_t g, size_t t) const { return features.matrix(g, t); };

  // computes a matrix M(g,t)
  // with M(g,t) = sum_e local_baseline_phi(e,g) local_phi(e,g,t) sum_s theta(e,s,t) sigma(e,s)
  Matrix explained_gene_type() const;
  // computes a matrix M(g,t)
  // with M(g,t) = phi(g,t) sum_e local_baseline_phi(e,g) local_phi(e,g,t) sum_s theta(e,s,t) sigma(e,s)
  Matrix expected_gene_type() const;

  void update_contributions();
  void update_kernels();
  void identity_kernels();
  void add_experiment(const Counts &data, size_t coord_sys);
};

template <typename Type>
std::ostream &operator<<(std::ostream &os, const Model<Type> &pfa);

template <typename Type>
Model<Type>::Model(const std::vector<Counts> &c, size_t T_,
                   const Parameters &parameters_, bool same_coord_sys)
    : G(max_row_number(c)),
      T(T_),
      E(0),
      iteration(0),
      experiments(),
      parameters(parameters_),
      contributions_gene_type(G, T, arma::fill::zeros),
      contributions_gene(G, arma::fill::zeros),
      prev_g_r(G, T),
      prev_g_p(G, T),
      prev_sign_r(G, T, arma::fill::zeros),
      prev_sign_p(G, T, arma::fill::zeros),
      features(G, T, parameters),
      mix_prior(sum_rows(c), T, parameters) {
  LOG(debug) << "Model G = " << G << " T = " << T << " E = " << E;
  prev_g_r.fill(parameters.sgd_step_size);
  prev_g_p.fill(parameters.sgd_step_size);
  size_t coord_sys = 0;
  for (auto &counts : c)
    add_experiment(counts, same_coord_sys ? 0 : coord_sys++);
  update_contributions();

  features.matrix.fill(1);

  // TODO move this code into the classes for prior and features
  // if (not parameters.targeted(Target::phi_prior_local))
  // features.prior.set_unit();

  // if (not parameters.targeted(Target::phi_local))
  //   features.matrix.ones();

  enforce_positive_parameters();

  if (parameters.targeted(Target::field)) {
    if (parameters.identity_kernels)
      identity_kernels();
    else
      update_kernels();
  }
}

template <typename Type>
void Model<Type>::identity_kernels() {
  LOG(debug) << "Updating kernels: using identity kernels";
  for (auto &coordinate_system : coordinate_systems)
    for (auto e1 : coordinate_system.members)
      for (auto e2 : coordinate_system.members) {
        Matrix m(experiments[e1].S, experiments[e2].S, arma::fill::zeros);
        if (e1 == e2)
          m.eye();
        kernels[{e1, e2}] = m;
      }
}

template <typename Type>
void Model<Type>::update_kernels() {
  LOG(debug) << "Updating kernels";
  for (auto &coordinate_system : coordinate_systems)
    for (auto e1 : coordinate_system.members) {
      for (auto e2 : coordinate_system.members)
        kernels[{e1, e2}]
            = apply_kernel(compute_sq_distances(experiments[e1].coords,
                                                experiments[e2].coords),
                           parameters.hyperparameters.sigma);
    }

  // row normalize
  // TODO check should we do column normalization?
  for (auto &coordinate_system : coordinate_systems)
    if (true)
      for (auto e1 : coordinate_system.members) {
        Vector z(experiments[e1].S, arma::fill::zeros);
        for (auto e2 : coordinate_system.members)
          z += rowSums<Vector>(kernels[{e1, e2}]);
        for (auto e2 : coordinate_system.members)
          kernels[{e1, e2}].each_col() /= z;
      }
    else
      for (auto e2 : coordinate_system.members) {
        Vector z(experiments[e2].S, arma::fill::zeros);
        for (auto e1 : coordinate_system.members)
          z += colSums<Vector>(kernels[{e1, e2}]);
        for (auto e1 : coordinate_system.members)
          kernels[{e1, e2}].each_row() /= z.t();
      }
  for (auto &kernel : kernels)
    LOG(debug) << "Kernel " << kernel.first.first << " " << kernel.first.second
               << std::endl
               << kernel.second;
}

template <typename V>
std::vector<size_t> get_order(const V &v) {
  size_t N = v.size();
  std::vector<size_t> order(N);
  std::iota(begin(order), end(order), 0);
  std::sort(begin(order), end(order), [&v] (size_t a, size_t b) { return v[a] > v[b]; });
  return order;
}

template <typename Type>
void Model<Type>::store(const std::string &prefix, bool reorder) const {
  auto factor_names = form_factor_names(T);
  auto &gene_names = experiments.begin()->data.row_names;

  auto exp_gene_type = expected_gene_type();
  std::vector<size_t> order;
  if (reorder) {
    auto cs = colSums<Vector>(exp_gene_type);
    order = get_order(cs);
  }

#pragma omp parallel sections if (DO_PARALLEL)
  {
#pragma omp section
    write_matrix(exp_gene_type, prefix + "expected-features" + FILENAME_ENDING,
                 parameters.compression_mode, gene_names, factor_names, order);
#pragma omp section
    features.store(prefix, gene_names, factor_names, order);
#pragma omp section
    write_matrix(contributions_gene_type,
                 prefix + "contributions_gene_type" + FILENAME_ENDING,
                 parameters.compression_mode, gene_names, factor_names, order);
#pragma omp section
    write_vector(contributions_gene,
                 prefix + "contributions_gene" + FILENAME_ENDING,
                 parameters.compression_mode, gene_names);
  }
  for (size_t e = 0; e < E; ++e) {
    std::string exp_prefix = prefix + "experiment"
                             + to_string_embedded(e, EXPERIMENT_NUM_DIGITS)
                             + "-";
    experiments[e].store(exp_prefix, features, order);
  }
}

template <typename Type>
void Model<Type>::restore(const std::string &prefix) {
  contributions_gene_type = parse_file<Matrix>(prefix + "contributions_gene_type" + FILENAME_ENDING, read_matrix, "\t");
  contributions_gene = parse_file<Vector>(prefix + "contributions_gene" + FILENAME_ENDING, read_vector<Vector>, "\t");
  features.restore(prefix);
  for (size_t e = 0; e < E; ++e) {
    std::string exp_prefix = prefix + "experiment"
                             + to_string_embedded(e, EXPERIMENT_NUM_DIGITS)
                             + "-";
    experiments[e].restore(exp_prefix);
  }
}

template <typename Type>
void Model<Type>::perform_pairwise_dge(const std::string &prefix) const {
  for (size_t e = 0; e < E; ++e) {
    std::string exp_prefix
        = prefix + "experiment" + to_string_embedded(e, EXPERIMENT_NUM_DIGITS) + "-";
    experiments[e].perform_pairwise_dge(exp_prefix, features);
  }
}

template <typename Type>
void Model<Type>::perform_local_dge(const std::string &prefix) const {
  for (size_t e = 0; e < E; ++e) {
    std::string exp_prefix
        = prefix + "experiment" + to_string_embedded(e, EXPERIMENT_NUM_DIGITS) + "-";
    experiments[e].perform_local_dge(exp_prefix, features);
  }
}

template <typename Type>
void Model<Type>::gibbs_sample(bool report_likelihood) {
  LOG(verbose) << "perform Gibbs step for " << parameters.targets;
  min_max("r(phi)", features.prior.r);
  min_max("p(phi)", features.prior.p);
  if (parameters.targeted(Target::contributions))
    sample_contributions(true);

  min_max("r(phi)", features.prior.r);
  min_max("p(phi)", features.prior.p);
  enforce_positive_parameters();
  sample_global_theta_priors();
  enforce_positive_parameters();
  return;

  if (report_likelihood) {
      if (verbosity >= Verbosity::verbose)
        LOG(info) << "Log-likelihood = " << log_likelihood();
      else
        LOG(info) << "Observed Log-likelihood = " << log_likelihood_poisson_counts();
    }

  // TODO add CLI switch
  auto order = random_order(4);
  for (auto &o : order)
    switch (o) {
      case 0:
        if (parameters.targeted(Target::theta_prior)
            and not parameters.theta_local_priors)
          sample_global_theta_priors();
        break;

      case 1:
        // if (parameters.targeted(Target::phi_prior))
        //   features.prior.sample(*this);
        break;

      case 2:
        if (parameters.targeted(Target::phi))
          features.sample(*this);
        break;

      case 3:
        if (parameters.targeted(Target::field))
          sample_fields();

        if (true)
          for (auto &experiment : experiments)
            experiment.gibbs_sample(features);
        break;

      default:
        break;
    }
  LOG(info) << "column sums of r: " << colSums<Vector>(features.prior.r).t();
  LOG(info) << "column sums of p: " << colSums<Vector>(features.prior.p).t();
  iteration++;
}

template <typename F>
void generate_alternative_prior(F &features, double phi_prior_gen_sd) {
  std::normal_distribution<double> rnorm(0, phi_prior_gen_sd);
  features.alt_prior = features.prior;
  for (auto &x : features.alt_prior.r)
    x *= exp(rnorm(EntropySource::rng));
  for (auto &x : features.alt_prior.p)
    x *= exp(rnorm(EntropySource::rng));
}

template <typename Type>
void Model<Type>::sample_contributions(bool update_phi_prior) {
  Matrix g_r(G, T, arma::fill::zeros);
  Matrix g_p(G, T, arma::fill::zeros);
  generate_alternative_prior(features, parameters.phi_prior_gen_sd);
  for (auto &experiment : experiments)
    experiment.sample_contributions(features, g_r, g_p);
  update_contributions();

  if (true) {
    if (true)
      for (size_t g = 0; g < G; ++g)
        for (size_t t = 0; t < T; ++t)
          g_r(g, t)
              += parameters.hyperparameters.phi_r_1 - 1
                 - parameters.hyperparameters.phi_r_2 * features.prior.r(g, t);

    if (true)
      for (size_t g = 0; g < G; ++g)
        for (size_t t = 0; t < T; ++t)
          g_p(g, t) += parameters.hyperparameters.phi_p_2 - 1
                       - (parameters.hyperparameters.phi_p_1 - 1)
                             * features.prior.p(g, t);
  }

  // g_p = -g_p;

  rprop_update(g_r, prev_sign_r, prev_g_r, features.prior.r);
  rprop_update(g_p, prev_sign_p, prev_g_p, features.prior.p);

  // features.prior.r.fill(100.0);
  // features.prior.p.fill(1.0);
  // features.prior.p.fill(0.1 / 0.9);

  min_max("rates r", prev_g_r);
  min_max("rates p", prev_g_p);

  enforce_positive_parameters();
}

template <typename Type>
void Model<Type>::enforce_positive_parameters() {
  features.enforce_positive_parameters();
  for(auto &experiment: experiments)
    experiment.enforce_positive_parameters();
}

template <Partial::Kind feat_kind>
void do_sample_fields(
    Model<ModelType<feat_kind, Partial::Kind::HierGamma>> &model) {
  LOG(verbose) << "Sampling fields";
  std::vector<Matrix> observed;
  std::vector<Matrix> explained;
  for (auto &experiment : model.experiments) {
    observed.push_back(Matrix(experiment.S, experiment.T, arma::fill::zeros));
    explained.push_back(Matrix(experiment.S, experiment.T, arma::fill::zeros));
  }

  for (auto &coordinate_system : model.coordinate_systems)
    for (auto e2 : coordinate_system.members) {
      const auto intensities
          = model.experiments[e2].marginalize_genes(model.features);
#pragma omp parallel for if (DO_PARALLEL)
      for (size_t t = 0; t < model.T; ++t)
        for (auto e1 : coordinate_system.members) {
          const auto &kernel = model.kernels.find({e2, e1})->second;
          for (size_t s2 = 0; s2 < model.experiments[e2].S; ++s2) {
            for (size_t s1 = 0; s1 < model.experiments[e1].S; ++s1) {
              const Float w = kernel(s2, s1);
              observed[e1](s1, t)
                  += w
                     * (model.experiments[e2].weights.prior.r(t)
                        + model.experiments[e2].contributions_spot_type(s2, t));
              explained[e1](s1, t)
                  += w
                     * (model.experiments[e2].weights.prior.p(t)
                        // TODO play with switching
                        + intensities[t] * model.experiments[e2].spot[s2]
                              // +
                              // TODO hack: remove theta_explained_spot_type
                              // model.experiments[e2].theta_explained_spot_type(s2, t)
                              * (e1 == e2 and s1 == s2
                                     ? 1
                                     : model.experiments[e2].field(s2, t)));
            }
          }
        }
    }

  for (size_t e = 0; e < model.E; ++e) {
    LOG(verbose) << "Sampling field for experiment " << e;
    Partial::perform_sampling(observed[e], explained[e],
                              model.experiments[e].field,
                              model.parameters.over_relax);
  }
}

template <Partial::Kind feat_kind>
void do_sample_fields(
    Model<ModelType<feat_kind, Partial::Kind::Dirichlet>> &model
    __attribute__((unused))) {}

template <typename Type>
void Model<Type>::sample_fields() {
  do_sample_fields(*this);
}

template <typename Type>
void Model<Type>::sample_global_theta_priors() {
  // TODO refactor
  if (std::is_same<typename Type::weights_t::prior_type,
                   PRIOR::THETA::Dirichlet>::value)
    return;
  Matrix observed(0, T);
  for (auto &experiment : experiments)
    observed = arma::join_vert(observed, experiment.weights.matrix);

  if (parameters.normalize_spot_stats)
    observed.each_col() /= rowSums<Vector>(observed);

  mix_prior.sample(observed);

  for (auto &experiment : experiments)
    experiment.weights.prior = mix_prior;

  min_max("weights r", mix_prior.r);
  min_max("weights p", mix_prior.p);
}

template <typename Type>
double Model<Type>::log_likelihood_poisson_counts() const {
  double l = 0;
  for (auto &experiment : experiments)
    l += experiment.log_likelihood_poisson_counts();
  return l;
}

template <typename Type>
double Model<Type>::log_likelihood() const {
  double l = 0; // TODO remove unused features.log_likelihood();
  LOG(verbose) << "Global feature log likelihood: " << l;
  for (auto &experiment : experiments)
    l += experiment.log_likelihood();
  return l;
}

// computes a matrix M(g,t)
// with M(g,t) = sum_e local_baseline_phi(e,g) local_phi(e,g,t) sum_s theta(e,s,t) sigma(e,s)
template <typename Type>
Matrix Model<Type>::explained_gene_type() const {
  Matrix explained(G, T, arma::fill::zeros);
  for (auto &experiment : experiments) {
    Vector theta_t = experiment.marginalize_spots();
    for (size_t t = 0; t < T; ++t)
#pragma omp parallel for if (DO_PARALLEL)
      for (size_t g = 0; g < G; ++g)
        explained(g, t)
            += experiment.baseline_phi(g) * experiment.phi(g, t) * theta_t(t);
  }
  return explained;
}

// computes a matrix M(g,t)
// with M(g,t) = phi(g,t) sum_e local_baseline_phi(e,g) local_phi(e,g,t) sum_s
// theta(e,s,t) sigma(e,s)
template <typename Type>
Matrix Model<Type>::expected_gene_type() const {
  Matrix m(G, T, arma::fill::zeros);
  for (auto &experiment : experiments)
    m += experiment.expected_gene_type(features);
  return m;
}

template <typename V>
bool generate_next(V &v, const V &low, const V &high, double step) {
  auto l_iter = low.begin();
  auto h_iter = high.begin();
  for (auto &x : v) {
    if ((x += step) > *h_iter)
      x = *l_iter;
    else
      return true;
    l_iter++;
    h_iter++;
  }
  return false;
}

template <typename Type>
template <typename Fnc1, typename Fnc2>
Vector Model<Type>::get_low_high(size_t coord_sys_idx, Float init, Fnc1 fnc1,
                                 Fnc2 fnc2) const {
  if (coordinate_systems.size() <= coord_sys_idx
      or coordinate_systems[coord_sys_idx].members.empty())
    return {0};
  size_t D = 0;
  for (auto &exp_idx : coordinate_systems[coord_sys_idx].members)
    D = std::max<size_t>(D, experiments[exp_idx].coords.n_cols);
  Vector v(D, arma::fill::ones);
  v *= init;
  for (size_t d = 0; d < D; ++d)
    for (auto &exp_idx : coordinate_systems[coord_sys_idx].members)
      v[d] = fnc1(v[d], fnc2(experiments[exp_idx].coords.col(d)));
  return v;
}

template <typename Type>
Vector Model<Type>::get_low(size_t coord_sys_idx) const {
  return get_low_high(coord_sys_idx, std::numeric_limits<Float>::infinity(),
                      [](double a, double b) { return std::min(a, b); },
                      [](const Vector &x) { return arma::min(x); });
}

template <typename Type>
Vector Model<Type>::get_high(size_t coord_sys_idx) const {
  return get_low_high(coord_sys_idx, -std::numeric_limits<Float>::infinity(),
                      [](double a, double b) { return std::max(a, b); },
                      [](const Vector &x) { return arma::max(x); });
}

template <typename Type>
void Model<Type>::predict_field(std::ofstream &ofs,
                                size_t coord_sys_idx) const {
  const double sigma = parameters.hyperparameters.sigma;
  // Matrix feature_marginal(E, T, arma::fill::zeros); TODO
  double N = 8e8;
  Vector low = get_low(coord_sys_idx);
  Vector high = get_high(coord_sys_idx);
  double alpha = 0.05;
  low = low - (high - low) * alpha;
  high = high + (high - low) * alpha;
  // one over N of the total volume
  double step = arma::prod((high - low) % (high - low)) / N;
  // n-th root of step
  step = exp(log(step) / low.n_elem);
  Vector coord = low;
  do {
    bool first = true;
    for (auto &c : coord) {
      if (first)
        first = false;
      else
        ofs << ",";
      ofs << c;
    }
    Vector observed(T, arma::fill::ones);
    Vector explained(T, arma::fill::ones);
    for (auto exp_idx : coordinate_systems[coord_sys_idx].members) {
      for (size_t s = 0; s < experiments[exp_idx].S; ++s) {
        const Vector diff = coord - experiments[exp_idx].coords.row(s).t();
        const double d = arma::accu(diff % diff);
        const double w
            = 1 / sqrt(2 * M_PI) / sigma * exp(-d / 2 / sigma / sigma);
        for (size_t t = 0; t < T; ++t) {
          observed[t] += w * experiments[exp_idx].contributions_spot_type(s, t);
          explained[t] += w * 1  // TODO feature_marginal(exp_idx, t)
                          // * experiments[exp_idx].theta(s, t)
                          // / experiments[exp_idx].field(s, t)
                          * experiments[exp_idx].spot(s);
        }
      }
    }
    double z = 0;
    for (size_t t = 0; t < T; ++t)
      z += observed[t] / explained[t];
    for (size_t t = 0; t < T; ++t) {
      double predicted = observed[t] / explained[t] / z;
      ofs << "," << predicted;
    }
    ofs << std::endl;
  } while (generate_next(coord, low, high, step));
}

template <typename Type>
void Model<Type>::update_contributions() {
  contributions_gene_type.zeros();
  contributions_gene.zeros();
  for (auto &experiment : experiments)
#pragma omp parallel for if (DO_PARALLEL)
    for (size_t g = 0; g < G; ++g) {
      contributions_gene(g) += experiment.contributions_gene(g);
      for (size_t t = 0; t < T; ++t)
        contributions_gene_type(g, t)
            += experiment.contributions_gene_type(g, t);
    }
}

template <typename Type>
void Model<Type>::add_experiment(const Counts &counts, size_t coord_sys) {
  Parameters experiment_parameters = parameters;
  parameters.hyperparameters.phi_p_1 *= local_phi_scaling_factor;
  parameters.hyperparameters.phi_r_1 *= local_phi_scaling_factor;
  parameters.hyperparameters.phi_p_2 *= local_phi_scaling_factor;
  parameters.hyperparameters.phi_r_2 *= local_phi_scaling_factor;
  experiments.push_back({counts, T, experiment_parameters});
  E++;
  // TODO check redundancy with Experiment constructor
  experiments.rbegin()->features.matrix.ones();
  experiments.rbegin()->features.prior.set_unit(local_phi_scaling_factor);
  while(coordinate_systems.size() <= coord_sys)
    coordinate_systems.push_back({});
  coordinate_systems[coord_sys].members.push_back(E-1);
}

template <typename Type>
std::ostream &operator<<(std::ostream &os, const Model<Type> &model) {
  os << "Poisson Factorization "
     << "G = " << model.G << " "
     << "T = " << model.T << " "
     << "E = " << model.E << std::endl;

  if (verbosity >= Verbosity::debug) {
    print_matrix_head(os, model.features.matrix, "Φ");
    os << model.features.prior;
  }
  for (auto &experiment : model.experiments)
    os << experiment;

  return os;
}

template <typename Type>
Model<Type> operator*(const Model<Type> &a, const Model<Type> &b) {
  Model<Type> model = a;

  model.contributions_gene_type %= b.contributions_gene_type;
  model.contributions_gene %= b.contributions_gene;
  model.features.matrix %= b.features.matrix;
  for (size_t e = 0; e < model.E; ++e)
    model.experiments[e] = model.experiments[e] * b.experiments[e];

  return model;
}

template <typename Type>
Model<Type> operator+(const Model<Type> &a, const Model<Type> &b) {
  Model<Type> model = a;

  model.contributions_gene_type += b.contributions_gene_type;
  model.contributions_gene += b.contributions_gene;
  model.features.matrix += b.features.matrix;
  for (size_t e = 0; e < model.E; ++e)
    model.experiments[e] = model.experiments[e] + b.experiments[e];

  return model;
}

template <typename Type>
Model<Type> operator-(const Model<Type> &a, const Model<Type> &b) {
  Model<Type> model = a;

  model.contributions_gene_type -= b.contributions_gene_type;
  model.contributions_gene -= b.contributions_gene;
  model.features.matrix -= b.features.matrix;
  for (size_t e = 0; e < model.E; ++e)
    model.experiments[e] = model.experiments[e] - b.experiments[e];

  return model;
}

template <typename Type>
Model<Type> operator*(const Model<Type> &a, double x) {
  Model<Type> model = a;

  model.contributions_gene_type *= x;
  model.contributions_gene *= x;
  model.features.matrix *= x;
  for (auto &experiment : model.experiments)
    experiment = experiment * x;

  return model;
}

template <typename Type>
Model<Type> operator/(const Model<Type> &a, double x) {
  Model<Type> model = a;

  model.contributions_gene_type /= x;
  model.contributions_gene /= x;
  model.features.matrix /= x;
  for (auto &experiment : model.experiments)
    experiment = experiment / x;

  return model;
}
}

#endif
