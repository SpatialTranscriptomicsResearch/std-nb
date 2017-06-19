#ifndef MODEL_HPP
#define MODEL_HPP

#include "Experiment.hpp"
#include "Mesh.hpp"
#include "priors.hpp"

namespace STD {

const int EXPERIMENT_NUM_DIGITS = 4;
const bool abort_on_fatal_errors = false;

struct Model {
  // TODO consider const
  /** number of genes */
  size_t G;
  /** number of factors */
  size_t T;
  /** number of experiments */
  size_t E;
  /** number of spots */
  size_t S;

  std::vector<Experiment> experiments;

  Parameters parameters;

  /** factor loading matrix */
  Matrix gamma;
  Matrix negodds_rho;
  struct CoordinateSystem {
    CoordinateSystem() : S(0), N(0), T(0){};
    size_t S, N, T;
    // std::vector<Matrix> coords;
    std::vector<size_t> members;
    Mesh mesh;
    Matrix field;
  };
  std::vector<CoordinateSystem> coordinate_systems;

  using prior_type = PRIOR::THETA::Gamma;
  prior_type mix_prior;

  /** hidden contributions to the count data due to the different factors */
  Matrix contributions_gene_type;
  Vector contributions_gene;

  Model(const std::vector<Counts> &data, size_t T, const Parameters &parameters,
        bool same_coord_sys);

  void set_zero();
  Model compute_gradient(double &score) const;
  double compute_hyperparameter_gradient(Model &gradient) const;
  void register_gradient(size_t g, size_t e, size_t s, const Vector &cnts,
                         Model &gradient) const;
  void finalize_gradient(Model &gradient) const;
  double param_likel() const;
  Vector vectorize() const;
  template <typename Iter>
  void from_log_vector(Iter iter) {
    if (parameters.targeted(Target::gamma_prior)) {
      LOG(debug) << "Getting gamma prior from vector";
      parameters.hyperparameters.gamma_1 = exp(*iter++);
      parameters.hyperparameters.gamma_2 = exp(*iter++);
    }

    if (parameters.targeted(Target::gamma)) {
      LOG(debug) << "Getting gamma from vector";
      for (auto &x : gamma)
        x = exp(*iter++);
    }

    if (parameters.targeted(Target::rho)) {
      LOG(debug) << "Getting negodds_rho from vector";
      for (auto &x : negodds_rho)
        x = exp(*iter++);
    }

    if (parameters.targeted(Target::field)) {
      LOG(debug) << "Getting global field from vector";
      for (auto &coord_sys : coordinate_systems)
        for (auto &x : coord_sys.field)
          x = exp(*iter++);
    }

    if (parameters.targeted(Target::theta_prior)) {
      LOG(debug) << "Getting theta prior r and p from vector";
      for (auto &x : mix_prior.r)
        x = exp(*iter++);
      for (auto &x : mix_prior.p)
        x = exp(*iter++);
    }

    for (auto &experiment : experiments)
      experiment.from_log_vector(iter);

    update_experiment_fields();
  };

  void gradient_update();
  size_t size() const;

  void enforce_positive_parameters();

  void store(const std::string &prefix, bool reorder = true) const;
  void restore(const std::string &prefix);

  double log_likelihood(const std::string &prefix) const;

  // computes a matrix M(g,t)
  //   gamma(g,t) sum_e beta(e,g) lambda(e,g,t) sum_s theta(e,s,t) sigma(e,s)
  Matrix expected_gene_type() const;

  void update_experiment_fields();
  void update_contributions();
  Matrix field_fitness_posterior_gradient() const;
  double field_gradient(const CoordinateSystem &coord_sys, const Matrix &field,
                        Matrix &grad) const;
  void initialize_coordinate_systems(double v);
  void add_experiment(const Counts &data, size_t coord_sys);
};

std::ostream &operator<<(std::ostream &os, const Model &pfa);

Model operator*(const Model &a, const Model &b);
Model operator+(const Model &a, const Model &b);
Model operator-(const Model &a, const Model &b);
Model operator*(const Model &a, double x);
Model operator/(const Model &a, double x);
}

#endif
