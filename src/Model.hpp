#ifndef MODEL_HPP
#define MODEL_HPP

#include <random>
#include "PartialModel.hpp"
#include "compression.hpp"
#include "counts.hpp"
#include "entropy.hpp"
#include "io.hpp"
#include "Experiment.hpp"
#include "log.hpp"
#include "metropolis_hastings.hpp"
#include "odds.hpp"
#include "parallel.hpp"
#include "parameters.hpp"
#include "pdist.hpp"
#include "priors.hpp"
#include "sampling.hpp"
#include "stats.hpp"
#include "target.hpp"
#include "timer.hpp"
#include "verbosity.hpp"

namespace PoissonFactorization {

const double local_phi_scaling_factor = 50;

template <Partial::Kind feat_kind = Partial::Kind::Gamma,
          Partial::Kind mix_kind = Partial::Kind::HierGamma>
struct Model {
  using features_t = Partial::Model<Partial::Variable::Feature, feat_kind>;
  using weights_t = Partial::Model<Partial::Variable::Mix, mix_kind>;
  using experiment_t = Experiment<feat_kind, mix_kind>;

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
  IMatrix contributions_gene_type;
  IVector contributions_gene;

  /** factor loading matrix */
  features_t features;
  // TODO consider weights_t global_weights;

  typename weights_t::prior_type mix_prior;

  Model(const std::vector<Counts> &data, const size_t T,
        const Parameters &parameters);
  // TODO implement loading of Experiment
  // Model(const Counts &counts, const Paths &paths, const Parameters
  // &parameters);

  void store(const std::string &prefix) const;

  /** sample each of the variables from their conditional posterior */
  void gibbs_sample();

  double log_likelihood() const;

  inline Float &phi(size_t g, size_t t) { return features.matrix(g, t); };
  inline Float phi(size_t g, size_t t) const { return features.matrix(g, t); };

  // computes a matrix M(g,t)
  // with M(g,t) = sum_e local_baseline_phi(e,g) local_phi(e,g,t) sum_s theta(e,s,t) sigma(e,s)
  Matrix explained_gene_type() const;

  void update_contributions();
  void add_experiment(const Counts &data);
};

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
std::ostream &operator<<(std::ostream &os,
                         const Model<feat_kind, mix_kind> &pfa);

size_t sum_rows(const std::vector<Counts> &c) {
  size_t n = 0;
  for(auto &x: c)
    n += x.counts.n_rows;
  return n;
}

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
Model<feat_kind, mix_kind>::Model(const std::vector<Counts> &c, const size_t T_,
                                  const Parameters &parameters_)
    : G(c.begin()->counts.n_rows),  // TODO FIXME c could be empty
      T(T_),
      E(c.size()),
      experiments(),
      parameters(parameters_),
      contributions_gene_type(G, T, arma::fill::zeros),
      contributions_gene(G, arma::fill::zeros),
      features(G, T, parameters),
      mix_prior(sum_rows(c), T, parameters) {
  LOG(verbose) << "G = " << G << " T = " << T << " E = " << E;
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

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
void Model<feat_kind, mix_kind>::store(const std::string &prefix) const {
  std::vector<std::string> factor_names;
  for (size_t t = 1; t <= T; ++t)
    factor_names.push_back("Factor " + std::to_string(t));
  auto &gene_names = experiments.begin()->data.row_names;
  features.store(prefix, gene_names, factor_names);
  write_matrix(contributions_gene_type, prefix + "contributions_gene_type.txt", gene_names, factor_names);
  write_vector(contributions_gene, prefix + "contributions_gene.txt", gene_names);
  for (size_t e = 0; e < E; ++e) {
    std::string exp_prefix = prefix + "experiment" + to_string_embedded(e, 3) + "-";
    experiments[e].store(exp_prefix, features);
  }
}

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
void Model<feat_kind, mix_kind>::gibbs_sample() {
  if (parameters.targeted(Target::contributions)) {
    for (auto &experiment : experiments)
      experiment.sample_contributions(features.matrix);
    update_contributions();
  }

  if (parameters.targeted(Target::theta_prior) and parameters.theta_global) {
    size_t S = 0;
    for (auto &experiment : experiments)
      S += experiment.S;
    Matrix feature_matrix(G, T, arma::fill::zeros);
    for (auto &experiment : experiments) {
      Matrix current_feature_matrix
          = features.matrix % experiment.features.matrix;
      for (size_t g = 0; g < G; ++g)
        for (size_t t = 0; t < T; ++t)
          current_feature_matrix(g, t) *= experiment.baseline_phi(g);
      feature_matrix += current_feature_matrix;
    }

    IMatrix contr_spot_type(0, T);
    for (auto &experiment : experiments)
      contr_spot_type = arma::join_vert(contr_spot_type,
                                        experiment.contributions_spot_type);

    Vector spot(S);
    double cumul_s = 0;
    for (auto &experiment : experiments) {
      for (size_t s = 0; s < experiment.S; ++s)
        spot(s + cumul_s) = experiment.spot(s);
      cumul_s += experiment.S;
    }

    mix_prior.sample(feature_matrix, contr_spot_type, spot);

    for (auto &experiment : experiments) {
      experiment.weights.prior.r = mix_prior.r;
      experiment.weights.prior.p = mix_prior.p;
    }
  }

  // for (auto &experiment : experiments)
  //   experiment.gibbs_sample(features.matrix);
  // update_contributions();

  if (parameters.targeted(Target::phi_prior))
    features.prior.sample(*this);

  if (parameters.targeted(Target::phi))
    features.sample(*this);

  for (auto &experiment : experiments)
    experiment.gibbs_sample(features.matrix);
}

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
double Model<feat_kind, mix_kind>::log_likelihood() const {
  double l = features.log_likelihood(contributions_gene_type);
  for(auto &experiment: experiments)
    l += experiment.log_likelihood();
  return l;
}

// computes a matrix M(g,t)
// with M(g,t) = sum_e local_baseline_phi(e,g) local_phi(e,g,t) sum_s theta(e,s,t) sigma(e,s)
template <Partial::Kind feat_kind, Partial::Kind mix_kind>
Matrix Model<feat_kind, mix_kind>::explained_gene_type() const {
  Matrix explained(G, T, arma::fill::zeros);
  for (auto &experiment : experiments) {
    Vector theta_t = experiment.marginalize_spots();
    for (size_t t = 0; t < T; ++t)
#pragma omp parallel for if (DO_PARALLEL)
      for (size_t g = 0; g < G; ++g)
        explained(g, t) += experiment.baseline_phi(g) * experiment.phi(g, t) * theta_t(t);
  }
  return explained;
}

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
void Model<feat_kind, mix_kind>::update_contributions() {
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

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
void Model<feat_kind, mix_kind>::add_experiment(const Counts &counts) {
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

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
std::ostream &operator<<(std::ostream &os,
                         const Model<feat_kind, mix_kind> &model) {
  os << "Poisson Factorization "
     << "G = " << model.G << " "
     << "T = " << model.T << " "
     << "E = " << model.E << std::endl;

  if (verbosity >= Verbosity::verbose) {
    print_matrix_head(os, model.features.matrix, "Φ");
    os << model.features.prior;
  }
  for (auto &experiment : model.experiments)
    os << experiment;

  return os;
}

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
Model<feat_kind, mix_kind> operator*(const Model<feat_kind, mix_kind> &a,
                                     const Model<feat_kind, mix_kind> &b) {
  Model<feat_kind, mix_kind> model = a;

  model.contributions_gene_type %= b.contributions_gene_type;
  model.contributions_gene %= b.contributions_gene;
  model.features.matrix %= b.features.matrix;
  for (size_t e = 0; e < model.E; ++e)
    model.experiments[e] = model.experiments[e] * b.experiments[e];

  return model;
}

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
Model<feat_kind, mix_kind> operator+(const Model<feat_kind, mix_kind> &a,
                                     const Model<feat_kind, mix_kind> &b) {
  Model<feat_kind, mix_kind> model = a;

  model.contributions_gene_type += b.contributions_gene_type;
  model.contributions_gene += b.contributions_gene;
  model.features.matrix += b.features.matrix;
  for (size_t e = 0; e < model.E; ++e)
    model.experiments[e] = model.experiments[e] + b.experiments[e];

  return model;
}

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
Model<feat_kind, mix_kind> operator-(const Model<feat_kind, mix_kind> &a,
                                     const Model<feat_kind, mix_kind> &b) {
  Model<feat_kind, mix_kind> model = a;

  model.contributions_gene_type -= b.contributions_gene_type;
  model.contributions_gene -= b.contributions_gene;
  model.features.matrix -= b.features.matrix;
  for (size_t e = 0; e < model.E; ++e)
    model.experiments[e] = model.experiments[e] - b.experiments[e];

  return model;
}

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
Model<feat_kind, mix_kind> operator*(const Model<feat_kind, mix_kind> &a,
                                     double x) {
  Model<feat_kind, mix_kind> model = a;

  model.contributions_gene_type *= x;
  model.contributions_gene *= x;
  model.features.matrix *= x;
  for (auto &experiment : model.experiments)
    experiment = experiment * x;

  return model;
}

template <Partial::Kind feat_kind, Partial::Kind mix_kind>
Model<feat_kind, mix_kind> operator/(const Model<feat_kind, mix_kind> &a,
                                     double x) {
  Model<feat_kind, mix_kind> model = a;

  model.contributions_gene_type /= x; // TODO note that this is inaccurate due to integer division
  model.contributions_gene /= x; // TODO note that this is inaccurate due to integer division
  model.features.matrix /= x;
  for (auto &experiment : model.experiments)
    experiment = experiment / x;

  return model;
}
}

#endif
