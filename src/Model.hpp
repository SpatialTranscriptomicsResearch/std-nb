#ifndef MODEL_HPP
#define MODEL_HPP

#include "Experiment.hpp"

namespace PoissonFactorization {

const double local_phi_scaling_factor = 50;

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

  typename weights_t::prior_type mix_prior;

  Model(const std::vector<Counts> &data, const size_t T,
        const Parameters &parameters);
  // TODO implement loading of Model
  // Model(const Counts &counts, const Paths &paths, const Parameters
  // &parameters);

  void store(const std::string &prefix) const;
  void perform_pairwise_dge(const std::string &prefix) const;
  void perform_local_dge(const std::string &prefix) const;

  /** sample each of the variables from their conditional posterior */
  void gibbs_sample(const std::vector<size_t> &which_experiments);

  void sample_global_theta_priors();

  double log_likelihood() const;
  double log_likelihood_poisson_counts() const;

  inline Float &phi(size_t g, size_t t) { return features.matrix(g, t); };
  inline Float phi(size_t g, size_t t) const { return features.matrix(g, t); };

  // computes a matrix M(g,t)
  // with M(g,t) = sum_e local_baseline_phi(e,g) local_phi(e,g,t) sum_s theta(e,s,t) sigma(e,s)
  Matrix explained_gene_type() const;

  void update_contributions();
  void add_experiment(const Counts &data);
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
                const Parameters &parameters_)
    : G(max_row_number(c)),
      T(T_),
      E(c.size()),
      experiments(),
      parameters(parameters_),
      contributions_gene_type(G, T, arma::fill::zeros),
      contributions_gene(G, arma::fill::zeros),
      features(G, T, parameters),
      mix_prior(sum_rows(c), T, parameters) {
  LOG(debug) << "Model G = " << G << " T = " << T << " E = " << E;
  for (auto &counts : c)
    add_experiment(counts);
  update_contributions();

  // TODO move this code into the classes for prior and features
  if (not parameters.targeted(Target::phi_prior_local)) {
    features.prior.r.fill(1);
    features.prior.p.fill(1);
  }

  if (not parameters.targeted(Target::phi_local))
    features.matrix.fill(1);
}

template <typename Type>
void Model<Type>::store(const std::string &prefix) const {
  auto factor_names = form_factor_names(T);
  auto &gene_names = experiments.begin()->data.row_names;
  features.store(prefix, gene_names, factor_names);
  write_matrix(contributions_gene_type, prefix + "contributions_gene_type.txt",
               gene_names, factor_names);
  write_vector(contributions_gene, prefix + "contributions_gene.txt",
               gene_names);
  for (size_t e = 0; e < E; ++e) {
    std::string exp_prefix
        = prefix + "experiment" + to_string_embedded(e, 3) + "-";
    experiments[e].store(exp_prefix, features);
  }
}

template <typename Type>
void Model<Type>::perform_pairwise_dge(const std::string &prefix) const {
  for (size_t e = 0; e < E; ++e) {
    std::string exp_prefix
        = prefix + "experiment" + to_string_embedded(e, 3) + "-";
    experiments[e].perform_pairwise_dge(exp_prefix, features);
  }
}

template <typename Type>
void Model<Type>::perform_local_dge(const std::string &prefix) const {
  for (size_t e = 0; e < E; ++e) {
    std::string exp_prefix
        = prefix + "experiment" + to_string_embedded(e, 3) + "-";
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

  for (auto &experiment : experiments)
    experiment.gibbs_sample(features.matrix);
  // TODO consider as alternative
  // for (auto &exp_idx : which_experiments)
  //   experiments[exp_idx].gibbs_sample(features.matrix);
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
void Model<Type>::add_experiment(const Counts &counts) {
  Parameters experiment_parameters = parameters;
  parameters.hyperparameters.phi_p_1 *= local_phi_scaling_factor;
  parameters.hyperparameters.phi_r_1 *= local_phi_scaling_factor;
  parameters.hyperparameters.phi_p_2 *= local_phi_scaling_factor;
  parameters.hyperparameters.phi_r_2 *= local_phi_scaling_factor;
  experiments.push_back({counts, T, experiment_parameters});
  experiments.rbegin()->features.matrix.fill(1);
  experiments.rbegin()->features.prior.r.fill(local_phi_scaling_factor);
  experiments.rbegin()->features.prior.p.fill(local_phi_scaling_factor);
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
