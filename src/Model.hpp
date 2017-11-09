#ifndef MODEL_HPP
#define MODEL_HPP

#include <map>

#include "Experiment.hpp"
#include "Mesh.hpp"
#include "design.hpp"
#include "modelspec.hpp"

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

  Design design;
  std::vector<Experiment> experiments;

  Parameters parameters;

  using Coefficients = std::vector<Coefficient>;
  Coefficients coeffs;

  std::vector<Coefficient>::iterator find_coefficient(const Coefficient::Id& cid);

  /** hidden contributions to the count data due to the different factors */
  Matrix contributions_gene_type;
  Vector contributions_gene;

  Model(const std::vector<Counts>& data, size_t T, const Design& design,
        const ModelSpec& model_spec, const Parameters& parameters);
  void remove_redundant_terms();
  void remove_redundant_terms(Coefficient::Kind kind);
  void remove_redundant_terms_sub(const std::vector<std::vector<size_t>> &cov_groups);

  size_t register_coefficient(
      const std::unordered_map<std::string, spec_parser::RandomVariable>& variable_map,
      std::string id,
      size_t experiment);
  void add_covariate_terms(
    const std::unordered_map<std::string, spec_parser::RandomVariable> &variable_map,
    const std::string &variable_id, std::function<void(Experiment&, size_t)> fnc);
  void add_covariates(const ModelSpec& ms);

  void add_gp_proxies();
  void add_prior_coefficients();

  void setZero();
  Model compute_gradient(double &score) const;
  void register_gradient(size_t g, size_t e, size_t s, const Vector &cnts,
                         Model &gradient, const Matrix &rate_gt,
                         const Matrix &rate_st, const Matrix &odds_gt,
                         const Matrix &odds_st) const;
  void register_gradient_zero_count(size_t g, size_t e, size_t s,
                                    Model &gradient, const Matrix &rate_gt,
                                    const Matrix &rate_st,
                                    const Matrix &odds_gt,
                                    const Matrix &odds_st) const;

  void coeff_debug_dump(const std::string &tag) const;

  Vector vectorize() const;
  void from_vector(const Vector &v);
  void gradient_update();
  size_t size() const;
  size_t number_parameters() const;

  void enforce_positive_parameters(double min_value = 1e-200);

  void store(const std::string &prefix, bool mean_and_var = false,
             bool reorder = true) const;
  void restore(const std::string &prefix);

  // computes a matrix M(g,t)
  //   gamma(g,t) sum_e beta(e,g) lambda(e,g,t) sum_s theta(e,s,t) sigma(e,s)
  Matrix expected_gene_type() const;

  void update_contributions();
  void add_experiment(const Counts &data);
};

std::ostream &operator<<(std::ostream &os, const Model &pfa);

Model operator+(const Model &a, const Model &b);
}

#endif
