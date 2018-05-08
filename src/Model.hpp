#ifndef MODEL_HPP
#define MODEL_HPP

#include <map>

#include "Experiment.hpp"
#include "Mesh.hpp"
#include "design.hpp"
#include "modelspec.hpp"
#include "spec_parser/expression/generate.hpp"

namespace STD {

const int EXPERIMENT_NUM_DIGITS = 4;

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

  ModelSpec model_spec;
  Design::Design design;
  std::string module_name;
  using FuncType = std::function<double(const double *)>;
  FuncType rate_fnc, odds_fnc;
  std::vector<FuncType> rate_derivs, odds_derivs;
  std::vector<Experiment> experiments;

  Parameters parameters;

  using Coefficients = std::vector<CoefficientPtr>;
  Coefficients coeffs;

  std::vector<CoefficientPtr>::iterator find_coefficient(
      const Coefficient::Id &cid);

  /** hidden contributions to the count data due to the different factors */
  Matrix contributions_gene_type;
  Vector contributions_gene;

  Model(const std::vector<Counts> &data, size_t T, const Design::Design &design,
        const ModelSpec &model_spec, const Parameters &parameters);
  Model clone() const;
  void remove_redundant_terms();
  void remove_redundant_terms(Coefficient::Kind kind);
  void remove_redundant_terms_sub(
      const std::map<CoefficientPtr, std::set<size_t>> &cov_groups);

  CoefficientPtr register_coefficient(
      const std::unordered_map<std::string, ModelSpec::Variable> &variable_map,
      std::string id, size_t experiment);
  void add_covariates();

  void add_gp_proxies();
  void add_prior_coefficients();

  void setZero();
  std::pair<Matrix, Matrix> compute_mean_and_var(size_t e) const;
  Model compute_gradient(double &score, bool compute_likelihood) const;
  void register_gradient(size_t g, size_t e, size_t s, size_t t,
                         const Vector &cnts, Model &gradient,
                         const Vector &rate, const Vector &odds,
                         const std::vector<double> &rate_coeffs,
                         const std::vector<double> &odds_coeffs) const;

  void register_gradient_total(
      size_t g, size_t e, size_t s, double total_rate, double total_odds,
      Model &gradient, const Vector &rate, const Vector &odds,
      const std::vector<std::vector<double>> &rate_coeffs,
      const std::vector<std::vector<double>> &odds_coeffs, RNG &rng) const;

  void register_gradient_zero_count(
      size_t g, size_t e, size_t s, size_t t, const Vector &cnts,
      Model &gradient, const Vector &rate, const Vector &odds,
      const std::vector<double> &rate_coeffs,
      const std::vector<double> &odds_coeffs) const;

  void coeff_debug_dump(const std::string &tag) const;

  Vector vectorize() const;
  void from_vector(const Vector &v);
  void gradient_update(size_t num_iterations,
                       std::function<bool(const CoefficientPtr)> is_included);
  size_t size() const;
  size_t number_parameters() const;

  void store(const std::string &prefix, bool mean_and_var = false,
             bool reorder = true) const;
  void restore(const std::string &prefix);

  void ensure_dimensions() const;

  void update_contributions();
  void add_experiment(const Counts &data);
};

std::ostream &operator<<(std::ostream &os, const Model &pfa);

Model operator+(const Model &a, const Model &b);
}  // namespace STD

#endif
