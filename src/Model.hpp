#ifndef MODEL_HPP
#define MODEL_HPP

#include <map>
#ifndef NDEBUG
// Uncomment the next line for debug output from Boost's Newton-Raphson code
// #define BOOST_MATH_INSTRUMENT
#endif
#include <LBFGS.h>
#include <boost/math/tools/roots.hpp>
#include "Experiment.hpp"
#include "Mesh.hpp"

namespace PoissonFactorization {

namespace NewtonRaphson {
const double lower = std::numeric_limits<double>::denorm_min();
const double upper = 1e5;

// Maximum possible binary digits accuracy for type T.
const int digits = std::numeric_limits<double>::digits;
// Accuracy doubles with each step, so stop when we have just over half the
// digits correct.
const int get_digits = static_cast<int>(digits * 0.6);

const size_t max_iter = 100;
};

const int EXPERIMENT_NUM_DIGITS = 4;

const bool abort_on_fatal_errors = false;

struct Model {
  using weights_t
      = Partial::Model<Partial::Variable::Mix, Partial::Kind::HierGamma>;
  using experiment_t = Experiment;

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
  Matrix phi_r;
  Matrix phi_p;
  struct CoordinateSystem {
    CoordinateSystem() : S(0), N(0), T(0){};
    size_t S, N, T;
    // std::vector<Matrix> coords;
    std::vector<size_t> members;
    Mesh mesh;
    Matrix field;
  };
  std::vector<CoordinateSystem> coordinate_systems;

  typename weights_t::prior_type mix_prior;

  Model(const std::vector<Counts> &data, size_t T, const Parameters &parameters,
        bool same_coord_sys);

  void enforce_positive_parameters();

  void store(const std::string &prefix, bool reorder = true) const;
  void restore(const std::string &prefix);

  void sample_local_r(size_t g, const std::vector<Matrix> counts_gst,
                      const Matrix &experiment_counts_gt,
                      const Matrix &experiment_theta_marginals,
                      std::mt19937 &rng);
  void sample_contributions(bool do_global_features, bool do_local_features,
                            bool do_theta, bool do_baseline);

  void sample_global_theta_priors();

  double log_likelihood(const std::string &prefix) const;

  // computes a matrix M(g,t)
  // with M(g,t) = phi(g,t) sum_e local_baseline_phi(e,g) local_phi(e,g,t) sum_s
  // theta(e,s,t) sigma(e,s)
  Matrix expected_gene_type() const;

  void update_fields();
  void update_experiment_fields();
  void update_contributions();
  Matrix field_fitness_posterior_gradient(const Matrix &f) const;

  double field_gradient(CoordinateSystem &coord_sys, const Matrix &field,
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
