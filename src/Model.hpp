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

  std::vector<experiment_t> experiments;

  Parameters parameters;

  /** hidden contributions to the count data due to the different factors */
  Matrix contributions_gene_type;
  Vector contributions_gene;

  /** factor loading matrix */
  features_t features;
  // TODO consider weights_t global_weights;
  struct CoordinateSystem {
    // std::vector<Matrix> coords;
    std::vector<size_t> members;
  };
  std::vector<CoordinateSystem> coordinate_systems;
  std::map<std::pair<size_t, size_t>, Matrix> kernels;

  typename weights_t::prior_type mix_prior;

  Model(const std::vector<Counts> &data, const size_t T,
        const Parameters &parameters, bool same_coord_sys);
  // TODO implement loading of Model
  // Model(const Counts &counts, const Paths &paths, const Parameters
  // &parameters);

  void store(const std::string &prefix, bool reorder = true) const;
  void perform_pairwise_dge(const std::string &prefix) const;
  void perform_local_dge(const std::string &prefix) const;

  /** sample each of the variables from their conditional posterior */
  void gibbs_sample(const std::vector<size_t> &which_experiments);

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
  void add_experiment(const Counts &data, size_t coord_sys);
};

template <typename Type>
std::ostream &operator<<(std::ostream &os, const Model<Type> &pfa);

size_t sum_rows(const std::vector<Counts> &c) {
  size_t n = 0;
  for (auto &x : c)
    n += x.counts.n_rows;
  return n;
}

size_t max_row_number(const std::vector<Counts> &c) {
  size_t x = 0;
  for(auto &m: c)
    x = std::max<size_t>(x, m.counts.n_rows);
  return x;
}

template <typename Type>
Model<Type>::Model(const std::vector<Counts> &c, const size_t T_,
                   const Parameters &parameters_, bool same_coord_sys)
    : G(max_row_number(c)),
      T(T_),
      E(0),
      experiments(),
      parameters(parameters_),
      contributions_gene_type(G, T, arma::fill::zeros),
      contributions_gene(G, arma::fill::zeros),
      features(G, T, parameters),
      mix_prior(sum_rows(c), T, parameters) {
  LOG(debug) << "Model G = " << G << " T = " << T << " E = " << E;
  size_t coord_sys = 0;
  for (auto &counts : c)
    add_experiment(counts, same_coord_sys ? 0 : coord_sys++);
  update_contributions();

  // TODO move this code into the classes for prior and features
  if (not parameters.targeted(Target::phi_prior_local)) {
    features.prior.r.fill(1);
    features.prior.p.fill(1);
  }

  if (not parameters.targeted(Target::phi_local))
    features.matrix.fill(1);

  if (parameters.targeted(Target::field))
    update_kernels();
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

  LOG(debug) << "Updating kernels 2";
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
  LOG(debug) << "Updating kernels 3";
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
  features.store(prefix, gene_names, factor_names, order);
  write_matrix(exp_gene_type, prefix + "expected-features" + FILENAME_ENDING, gene_names, factor_names, order);
  write_matrix(contributions_gene_type, prefix + "contributions_gene_type" + FILENAME_ENDING, gene_names, factor_names, order);
  write_vector(contributions_gene, prefix + "contributions_gene" + FILENAME_ENDING, gene_names);
  for (size_t e = 0; e < E; ++e) {
    std::string exp_prefix = prefix + "experiment"
                             + to_string_embedded(e, EXPERIMENT_NUM_DIGITS)
                             + "-";
    experiments[e].store(exp_prefix, features, order);
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
void Model<Type>::gibbs_sample(const std::vector<size_t> &which_experiments) {
  if (parameters.targeted(Target::contributions)) {
    for (auto &exp_idx : which_experiments)
      experiments[exp_idx].sample_contributions(features.matrix);
    update_contributions();
  }

  if (parameters.targeted(Target::theta_prior)
      and not parameters.theta_local_priors)
    sample_global_theta_priors();

  // for (auto &experiment : experiments)
  //   experiment.gibbs_sample(features.matrix);
  // update_contributions();

  if (parameters.targeted(Target::phi_prior))
    features.prior.sample_mh(*this);

  if (parameters.targeted(Target::phi))
    features.sample(*this);

  if (parameters.targeted(Target::field))
    sample_fields();

  for (auto &experiment : experiments)
    experiment.gibbs_sample(features.matrix);
  // TODO consider as alternative
  // for (auto &exp_idx : which_experiments)
  //   experiments[exp_idx].gibbs_sample(features.matrix);
}

template <typename Type>
void Model<Type>::sample_fields() {
  LOG(verbose) << "Sampling fields";
  std::vector<Matrix> observed;
  std::vector<Matrix> explained;
  for (auto &experiment : experiments) {
    observed.push_back(Matrix(experiment.S, experiment.T, arma::fill::zeros));
    explained.push_back(Matrix(experiment.S, experiment.T, arma::fill::zeros));
  }

  for (auto &coordinate_system : coordinate_systems)
    for (auto e2 : coordinate_system.members) {
      const auto intensities
          = experiments[e2].marginalize_genes(features.matrix);
#pragma omp parallel for if (DO_PARALLEL)
      for (size_t t = 0; t < T; ++t)
        for (auto e1 : coordinate_system.members) {
          const auto &kernel = kernels.find({e1, e2})->second;
          for (size_t s2 = 0; s2 < experiments[e2].S; ++s2) {
            for (size_t s1 = 0; s1 < experiments[e1].S; ++s1) {
              const Float w = kernel(s1, s2);
              observed[e1](s1, t)
                  += w * experiments[e2].contributions_spot_type(s2, t);
              explained[e1](s1, t)
                  += w * intensities[t] * experiments[e2].spot[s2]
                     * (e1 == e2 and s1 == s2 ? 1
                                              : experiments[e2].field(s2, t));
            }
          }
        }
    }

  for (size_t e = 0; e < E; ++e) {
    LOG(verbose) << "Sampling field for experiment " << e;
    observed[e].each_row() += experiments[e].weights.prior.r.t();
    explained[e].each_row() += experiments[e].weights.prior.p.t();
    Partial::perform_sampling(observed[e], explained[e], experiments[e].field,
                              parameters.over_relax);
  }
}

template <typename Type>
void Model<Type>::sample_global_theta_priors() {
  size_t S = 0;
  for (auto &experiment : experiments)
    S += experiment.S;
  Matrix feature_matrix(G, T, arma::fill::zeros);
  for (auto &experiment : experiments) {
    Matrix current_feature_matrix
        = features.matrix % experiment.features.matrix;
#pragma omp parallel for if (DO_PARALLEL)
    for (size_t g = 0; g < G; ++g)
      for (size_t t = 0; t < T; ++t)
        current_feature_matrix(g, t) *= experiment.baseline_phi(g);
    feature_matrix += current_feature_matrix;
  }

  Matrix contr_spot_type(0, T);
  for (auto &experiment : experiments)
    contr_spot_type
        = arma::join_vert(contr_spot_type, experiment.contributions_spot_type);

  Vector spot(S);
  double cumul_s = 0;
  for (auto &experiment : experiments) {
#pragma omp parallel for if (DO_PARALLEL)
    for (size_t s = 0; s < experiment.S; ++s)
      spot(s + cumul_s) = experiment.spot(s);
    cumul_s += experiment.S;
  }

  mix_prior.sample(feature_matrix, contr_spot_type, spot);

  for (auto &experiment : experiments)
    experiment.weights.prior = mix_prior;
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
  double l = features.log_likelihood();
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
    m += experiment.expected_gene_type(features.matrix);
  return m;
}

template <typename Type>
void Model<Type>::update_contributions() {
  contributions_gene_type.fill(0);
  contributions_gene.fill(0);
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
  experiments.rbegin()->features.matrix.fill(1);
  experiments.rbegin()->features.prior.r.fill(local_phi_scaling_factor);
  experiments.rbegin()->features.prior.p.fill(local_phi_scaling_factor);
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
