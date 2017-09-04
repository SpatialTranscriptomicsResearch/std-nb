#ifndef MODEL_HPP
#define MODEL_HPP

#include <map>

#include "Experiment.hpp"
#include "Mesh.hpp"
#include "design.hpp"
#include "formula.hpp"
#include "modelspec.hpp"
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

  Formula rate_formula, variance_formula;
  Design design;
  ModelSpec model_spec;
  std::vector<Experiment> experiments;

  std::map<CoefficientId, size_t> coeff2idx;

  Parameters parameters;

  using Coefficients = std::vector<Coefficient>;
  Coefficients coeffs;

  struct CoordinateSystem {
    CoordinateSystem() : S(0), N(0), T(0){};
    size_t S, N, T;
    // std::vector<Matrix> coords;
    std::vector<size_t> members;
    Mesh mesh;
    Matrix field;
  };
  std::vector<CoordinateSystem> coordinate_systems;

  /** hidden contributions to the count data due to the different factors */
  Matrix contributions_gene_type;
  Vector contributions_gene;

  Model(const std::vector<Counts>& data, size_t T, const Design& design,
        const ModelSpec& model_spec, const Parameters& parameters,
        bool same_coord_sys);
  void remove_redundant_terms();
  void remove_redundant_terms(Coefficient::Variable variable,
                              Coefficient::Kind kind);

  std::vector<size_t> get_covariate_idxs(
      const std::set<std::string>& covariates);
  Coefficient::Kind get_kind(const std::set<std::string>& covariates);
  void add_covariate_terms(
      const std::unordered_map<std::string, RandomVariable>& variable_map,
      Coefficient::Variable variable_type, const std::string& var);
  size_t register_coefficient(
      const std::unordered_map<std::string, RandomVariable>& variable_map,
      Coefficient::Variable variable_type,
      std::string id, size_t experiment);

  void setZero();
  Model compute_gradient(double &score) const;
  void register_gradient(size_t g, size_t e, size_t s, const Vector &cnts,
                         Model &gradient, const Matrix &rate_gt,
                         const Matrix &rate_st, const Matrix &odds_gt,
                         const Matrix &odds_st) const;
  void coeff_debug_dump(const std::string &tag) const;
  double param_likel() const;

  Vector vectorize() const;
  void from_vector(const Vector &v);
  void gradient_update();
  size_t size() const;
  size_t number_parameters() const;

  Vector make_mask() const;
  void apply_mask(Vector &x, Vector &rates, Vector &prev_sign) const;

  void enforce_positive_parameters(double min_value = 1e-200);

  void store(const std::string &prefix, bool mean_and_var = false,
             bool reorder = true) const;
  void restore(const std::string &prefix);

  // TODO covariates reactivate likelihood
  // double log_likelihood(const std::string &prefix) const;

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

Model operator+(const Model &a, const Model &b);
}

#endif
