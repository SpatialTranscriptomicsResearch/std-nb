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
#include "mesh.hpp"

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
  void perform_pairwise_dge(const std::string &prefix) const;
  void perform_local_dge(const std::string &prefix) const;

  void sample_local_r(size_t g, const std::vector<Matrix> counts_gst,
                      const Matrix &experiment_counts_gt,
                      const Matrix &experiment_theta_marginals,
                      std::mt19937 &rng);
  void sample_contributions(bool do_global_features, bool do_local_features,
                            bool do_theta, bool do_baseline);

  void sample_global_theta_priors();

  double log_likelihood(const std::string &prefix) const;
  double log_likelihood_conv_NB_counts() const;

  // computes a matrix M(g,t)
  // with M(g,t) = sum_e local_baseline_phi(e,g) local_phi(e,g,t) sum_s theta(e,s,t) sigma(e,s)
  Matrix explained_gene_type() const;
  // computes a matrix M(g,t)
  // with M(g,t) = phi(g,t) sum_e local_baseline_phi(e,g) local_phi(e,g,t) sum_s theta(e,s,t) sigma(e,s)
  Matrix expected_gene_type() const;

  void update_fields();
  void update_experiment_fields();
  void update_contributions();
  Matrix field_fitness_posterior_gradient(const Matrix &f) const;

  double field_gradient(CoordinateSystem &coord_sys, const Matrix &phi,
                        Matrix &grad) const;
  void initialize_coordinate_systems(double v);
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

  features.matrix.fill(1);

  // TODO move this code into the classes for prior and features
  // if (not parameters.targeted(Target::phi_prior_local))
  // features.prior.set_unit();

  // if (not parameters.targeted(Target::phi_local))
  //   features.matrix.ones();

  enforce_positive_parameters();

  initialize_coordinate_systems(1);
}

template <typename V>
std::vector<size_t> get_order(const V &v) {
  size_t N = v.size();
  std::vector<size_t> order(N);
  std::iota(begin(order), end(order), 0);
  std::sort(begin(order), end(order),
            [&v](size_t a, size_t b) { return v[a] > v[b]; });
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
#pragma omp section
    {
      auto print_field_matrix
          = [&](const std::string &path, const Vector &weights) {
              std::ofstream ofs(path);
              ofs << "coord_sys\tpoint_idx";
              for (size_t d = 0; d < coordinate_systems[0].mesh.dim; ++d)
                ofs << "\tx" << d;
              for (size_t t = 0; t < T; ++t)
                ofs << "\tFactor " << t + 1;
              ofs << std::endl;

              size_t coord_sys_idx = 0;
              for (size_t c = 0; c < coordinate_systems.size(); ++c) {
                for (size_t n = 0; n < coordinate_systems[c].N; ++n) {
                  ofs << coord_sys_idx << "\t" << n;
                  for (size_t d = 0; d < coordinate_systems[c].mesh.dim; ++d)
                    ofs << "\t" << coordinate_systems[c].mesh.points[n][d];
                  for (size_t t = 0; t < T; ++t)
                    ofs << "\t"
                        << coordinate_systems[c].field(n, order[t])
                               * weights[order[t]];
                  ofs << std::endl;
                }
                coord_sys_idx++;
              }
            };
      Vector weights(T, arma::fill::ones);
      print_field_matrix(prefix + "field" + FILENAME_ENDING, weights);
      // NOTE we ignore local features and local baseline
      Matrix mean = features.prior.r / features.prior.p;
      weights = colSums<Vector>(mean) % mix_prior.r / mix_prior.p;
      print_field_matrix(prefix + "expfield" + FILENAME_ENDING, weights);
    }
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
    std::string exp_prefix = prefix + "experiment"
                             + to_string_embedded(e, EXPERIMENT_NUM_DIGITS)
                             + "-";
    experiments[e].perform_pairwise_dge(exp_prefix, features);
  }
}

template <typename Type>
void Model<Type>::perform_local_dge(const std::string &prefix) const {
  for (size_t e = 0; e < E; ++e) {
    std::string exp_prefix = prefix + "experiment"
                             + to_string_embedded(e, EXPERIMENT_NUM_DIGITS)
                             + "-";
    experiments[e].perform_local_dge(exp_prefix, features);
  }
}

template <typename Fnc, typename Grad>
void optimize_newton_raphson(const std::string &tag, double &x, Fnc func,
                             Grad grad) {
  auto fnc = [&](double r) {
    double fn = func(r);
    double gr = grad(r);
    if (noisy)
      LOG(debug) << "local fnc/grad = " << fn << "/" << gr;
    return std::pair<double, double>(fn, gr);
  };

  if (noisy)
    LOG(debug) << "BEFORE: " << tag << "=" << x << " f=" << func(x);

  boost::uintmax_t it = NewtonRaphson::max_iter;
  double guess = x;
  x = boost::math::tools::newton_raphson_iterate(
      fnc, guess, NewtonRaphson::lower, guess * 100, NewtonRaphson::get_digits,
      it);
  if (it >= NewtonRaphson::max_iter) {
    LOG(fatal) << "Unable to locate solution for " << tag << " in "
               << NewtonRaphson::max_iter << " iterations.";
    LOG(fatal) << "prev=" << guess << " f(prev)=" << func(guess) << " cur=" << x
               << " f(cur)=" << func(x);

    if (abort_on_fatal_errors)
      exit(-1);
  }

  bool reached_upper = x >= NewtonRaphson::upper;

  if (noisy)
    LOG(debug) << "AFTER: " << tag << "=" << x << " f=" << func(x);

  if (reached_upper) {
    LOG(fatal) << "Error: reached upper limit for " << tag << "!";
    if (abort_on_fatal_errors)
      exit(-1);
  }
}

template <typename Type>
void Model<Type>::sample_local_r(size_t g, const std::vector<Matrix> counts_gst,
                                 const Matrix &experiment_counts_gt,
                                 const Matrix &experiment_theta_marginals,
                                 std::mt19937 &rng) {
  const double a = parameters.hyperparameters.phi_r_1;
  const double b = parameters.hyperparameters.phi_r_2;
  for (size_t t = 0; t < T; ++t) {
    double A = a * 50;
    double B = b * 50;

    for (size_t e = 0; e < E; ++e) {
      double theta_marginal = experiment_theta_marginals(e, t)
                              * experiments[e].baseline_feature.prior.r(g)
                              * features.prior.r(g, t);

      if (experiment_counts_gt(e, t) == 0) {
        if (noisy)
          LOG(debug) << "Gibbs sampling local r of (" << g << ", " << t
                     << ") for experiment " << e << ": "
                     << experiments[e].data.row_names[g];

        experiments[e].features.prior.r(g, t) = std::gamma_distribution<Float>(
            A,
            1.0 / (B
                   - theta_marginal * log(1 - neg_odds_to_prob(features.prior.p(
                                                  g, t)))))(rng);

        if (noisy)
          LOG(debug) << "local r= " << experiments[e].features.prior.r(g, t);
      } else {
        if (noisy)
          LOG(debug) << "t = " << t << " experiment_counts_gt(e,t) = "
                     << experiment_counts_gt(e, t);
        auto fn0 = [&](double r) {
          if (noisy)
            LOG(debug) << "g/t/e = " << g << "/" << t << "/" << e << " r=" << r;

          const double no = features.prior.p(g, t);
          double fnc = theta_marginal * (log(no) - log(1 + no));
          for (size_t s = 0; s < experiments[e].S; ++s) {
            double prod = experiments[e].baseline_feature.prior.r(g)
                          * features.prior.r(g, t) * experiments[e].theta(s, t)
                          * experiments[e].spot(s);
            fnc += prod * digamma_diff(r * prod, counts_gst[e](s, t));
          }

          if (not parameters.ignore_priors)
            fnc += (A - 1) / r - B;

          return fnc;
        };

        auto gr0 = [&](double r) {
          double grad = 0;
          for (size_t s = 0; s < experiments[e].S; ++s) {
            double prod = experiments[e].baseline_feature.prior.r(g)
                          * features.prior.r(g, t) * experiments[e].theta(s, t)
                          * experiments[e].spot(s);
            double prod_sq = prod * prod;
            grad += prod_sq * trigamma_diff(r * prod, counts_gst[e](s, t));
          }

          if (not parameters.ignore_priors)
            grad += -(A - 1) / r / r;

          return grad;
        };

        optimize_newton_raphson(
            "local r", experiments[e].features.prior.r(g, t), fn0, gr0);
      }
    }
  }
}

template <typename Type>
void Model<Type>::sample_contributions(bool do_global_features,
                                       bool do_local_features, bool do_theta,
                                       bool do_baseline) {
  // for (auto &experiment : experiments)
  //  experiment.weights.matrix.ones();
  for (auto &experiment : experiments)
    experiment.spot.ones();

  const double a = parameters.hyperparameters.phi_r_1;
  const double b = parameters.hyperparameters.phi_r_2;
  const double alpha = parameters.hyperparameters.phi_p_1;
  const double beta = parameters.hyperparameters.phi_p_2;

  Matrix experiment_theta_marginals(E, T, arma::fill::zeros);
  for (size_t e = 0; e < E; ++e)
    for (size_t s = 0; s < experiments[e].S; ++s)
      experiment_theta_marginals.row(e)
          += experiments[e].spot(s) * experiments[e].weights.matrix.row(s);

  do_local_features = do_local_features and E > 1;

  if (not do_local_features)
    for (size_t e = 0; e < E; ++e)
      experiments[e].features.prior.set_unit();

  if (do_global_features or do_local_features or do_baseline)
#pragma omp parallel if (DO_PARALLEL)
  {
    auto rng = EntropySource::rngs[omp_get_thread_num()];

#pragma omp for
    for (size_t g = 0; g < G; ++g) {
      std::vector<Matrix> counts_gst;

      Matrix experiment_counts_gt(E, T, arma::fill::zeros);
      for (size_t e = 0; e < E; ++e) {
        auto exp_counts
            = experiments[e].sample_contributions_gene(g, features, rng);
        experiment_counts_gt.row(e) = colSums<Vector>(exp_counts).t();
        counts_gst.push_back(exp_counts);
      }

      contributions_gene(g) = 0;
      for (size_t t = 0; t < T; ++t)
        contributions_gene_type(g, t) = 0;
      for (size_t e = 0; e < E; ++e)
        for (size_t t = 0; t < T; ++t)
          for (size_t s = 0; s < experiments[e].S; ++s) {
            contributions_gene(g) += counts_gst[e](s, t);
            contributions_gene_type(g, t) += counts_gst[e](s, t);
          }

      if (do_global_features and parameters.targeted(Target::global))
        for (size_t t = 0; t < T; ++t) {
          double cs = 0;
          for (size_t e = 0; e < E; ++e)
            cs += experiment_counts_gt(e, t);

          double theta_marginal = 0;
          for (size_t e = 0; e < E; ++e)
            theta_marginal += experiment_theta_marginals(e, t)
                              * experiments[e].baseline_feature.prior.r(g)
                              * experiments[e].features.prior.r(g, t);

          auto r2p = [&](double r) {
            return (alpha + contributions_gene_type(g, t) - 1)
                   / (alpha + contributions_gene_type(g, t) + beta
                      + r * theta_marginal - 2);
          };
          auto r2no = [&](double r) {
            return (beta + r * theta_marginal - 1)
                   / (alpha + contributions_gene_type(g, t) - 1);
          };

          if (cs == 0) {
            if (noisy)
              LOG(debug) << "Gibbs sampling r and p of (" << g << ", " << t
                         << "): " << experiments[0].data.row_names[g];

            features.prior.r(g, t) = std::gamma_distribution<Float>(
                a, 1.0 / (b
                          - theta_marginal
                                * log(1 - neg_odds_to_prob(
                                              features.prior.p(g, t)))))(rng);

            if (parameters.p_empty_map)
              // when this is used the r/p values lie along a curve, separated
              // from the MAP estimated ones
              features.prior.p(g, t) = r2no(features.prior.r(g, t));
            else
              // even when this is used the r/p values are somewhat separated
              // from the other values
              features.prior.p(g, t) = prob_to_neg_odds(sample_beta<Float>(
                  alpha, beta + features.prior.r(g, t) * theta_marginal, rng));

            if (noisy)
              LOG(debug) << "r/p= " << features.prior.r(g, t) << "/"
                         << features.prior.p(g, t);
          } else {
            if (noisy)
              LOG(debug) << "t = " << t << " cs = " << cs;
            auto fn0 = [&](double r) {
              const double p = r2p(r);
              if (noisy)
                LOG(debug) << "g/t = " << g << "/" << t << " r=" << r
                           << " p=" << p;

              const double no = r2no(r);
              double fnc = theta_marginal * (log(no) - log(1 + no));
              for (size_t e = 0; e < E; ++e)
                for (size_t s = 0; s < experiments[e].S; ++s) {
                  double prod = experiments[e].baseline_feature.prior.r(g)
                                * experiments[e].features.prior.r(g, t)
                                * experiments[e].theta(s, t)
                                * experiments[e].spot(s);
                  fnc += prod * digamma_diff(r * prod, counts_gst[e](s, t));
                }

              if (not parameters.ignore_priors)
                fnc += (a - 1) / r - b;

              return fnc;
            };

            auto gr0 = [&](double r) {
              double grad = 0;
              for (size_t e = 0; e < E; ++e)
                for (size_t s = 0; s < experiments[e].S; ++s) {
                  double prod = experiments[e].baseline_feature.prior.r(g)
                                * experiments[e].features.prior.r(g, t)
                                * experiments[e].theta(s, t)
                                * experiments[e].spot(s);
                  double prod_sq = prod * prod;
                  grad
                      += prod_sq * trigamma_diff(r * prod, counts_gst[e](s, t));
                }

              if (not parameters.ignore_priors)
                grad += -(a - 1) / r / r;

              return grad;
            };

            optimize_newton_raphson("global feature r", features.prior.r(g, t),
                                    fn0, gr0);
            features.prior.p(g, t) = r2no(features.prior.r(g, t));
          }
        }

      // baseline feature

      if (do_baseline and parameters.targeted(Target::baseline)) {
        double A = a * 50;
        double B = b * 50;

        for (size_t e = 0; e < E; ++e) {
          std::vector<double> theta_marginals(T);
          double theta_marginal = 0;
          for (size_t t = 0; t < T; ++t) {
            theta_marginal += theta_marginals[t]
                = experiment_theta_marginals(e, t)
                  * experiments[e].features.prior.r(g, t)
                  * features.prior.r(g, t);
          }

          auto fn0 = [&](double r) {
            if (noisy)
              LOG(debug) << "g/e = " << g << "/" << e << " r=" << r;

            double fnc = 0;
            for (size_t t = 0; t < T; ++t) {
              const double no = features.prior.p(g, t);
              fnc += theta_marginals[t] * (log(no) - log(1 + no));
              for (size_t s = 0; s < experiments[e].S; ++s) {
                double prod = experiments[e].features.prior.r(g, t)
                              * features.prior.r(g, t)
                              * experiments[e].theta(s, t)
                              * experiments[e].spot(s);
                fnc += prod * digamma_diff(r * prod, counts_gst[e](s, t));
              }
            }

            if (not parameters.ignore_priors)
              fnc += (A - 1) / r - B;

            return fnc;
          };

          auto gr0 = [&](double r) {
            double grad = 0;
            for (size_t t = 0; t < T; ++t)
              for (size_t s = 0; s < experiments[e].S; ++s) {
                double prod = experiments[e].features.prior.r(g, t)
                              * features.prior.r(g, t)
                              * experiments[e].theta(s, t)
                              * experiments[e].spot(s);
                double prod_sq = prod * prod;
                grad += prod_sq * trigamma_diff(r * prod, counts_gst[e](s, t));
              }

            if (not parameters.ignore_priors)
              grad += -(A - 1) / r / r;

            return grad;
          };

          optimize_newton_raphson("local feature baseline r",
                                  experiments[e].baseline_feature.prior.r(g),
                                  fn0, gr0);
        }
      }

      // local features

      if (do_local_features and parameters.targeted(Target::local))
        sample_local_r(g, counts_gst, experiment_counts_gt,
                       experiment_theta_marginals, rng);
    }
  }
  update_contributions();

  // theta
  if (do_theta and parameters.targeted(Target::theta))
#pragma omp parallel if (DO_PARALLEL)
  {
    auto rng = EntropySource::rngs[omp_get_thread_num()];

    for (auto &experiment : experiments) {
#pragma omp for
      for (size_t s = 0; s < experiment.S; ++s) {
        auto counts_gst
            = experiment.sample_contributions_spot(s, features, rng);
        auto cs = colSums<Vector>(counts_gst);

        for (size_t t = 0; t < T; ++t)
          if (cs[t] == 0) {
            if (noisy)
              LOG(debug) << "Gibbs sampling theta(" << s << ", " << t << ").";
            double marginal = 0;
            for (size_t g = 0; g < G; ++g)
              marginal += experiment.baseline_feature.prior.r(g)
                          * features.prior.r(g, t)
                          * experiment.features.prior.r(g, t)
                          * log(1 - neg_odds_to_prob(features.prior.p(g, t)));
            marginal *= experiment.spot(s);
            experiment.theta(s, t) = std::gamma_distribution<Float>(
                mix_prior.r(t), 1.0 / (mix_prior.p(t) - marginal))(rng);
            if (noisy)
              LOG(debug) << "theta = " << experiment.theta(s, t);
          } else {
            auto fn0 = [&](double x) {
              double fnc = 0;
              for (size_t g = 0; g < G; ++g) {
                if (noisy)
                  LOG(debug)
                      << "g/s/t = " << g << "/" << s << "/" << t
                      << " theta=" << x << " r=" << features.prior.r(g, t)
                      << " local r=" << experiment.features.prior.r(g, t)
                      << " p=" << neg_odds_to_prob(features.prior.p(g, t))
                      << " sigma=" << experiment.spot(s);

                // NOTE we don't multiply spot(s) in now, but once at the end
                double prod = experiment.baseline_feature.prior.r(g)
                              * features.prior.r(g, t)
                              * experiment.features.prior.r(g, t);

                fnc += prod * log(1 - neg_odds_to_prob(features.prior.p(g, t)));
                fnc += prod * digamma_diff(x * prod * experiment.spot(s),
                                           counts_gst(g, t));
              }

              fnc *= experiment.spot(s);

              if (not parameters.ignore_priors)
                // TODO ensure mix_prior.p is stored as negative odds
                fnc += (mix_prior.r(t) * experiment.field(s, t) - 1) / x
                       - mix_prior.p(t);

              return fnc;
            };

            auto gr0 = [&](double x) {
              double grad = 0;
              for (size_t g = 0; g < G; ++g) {
                // NOTE we don't multiply spot(s) in now, but once at the end
                double prod = experiment.baseline_feature.prior.r(g)
                              * features.prior.r(g, t)
                              * experiment.features.prior.r(g, t);
                double prod_sq = prod * prod;
                grad += prod_sq * trigamma_diff(x * prod * experiment.spot(s),
                                                counts_gst(g, t));
              }

              grad *= experiment.spot(s) * experiment.spot(s);

              if (not parameters.ignore_priors)
                // NOTE TODO this needs testing! the minus sign was missing
                grad += -(mix_prior.r(t) * experiment.field(s, t) - 1) / x / x;

              return grad;
            };

            optimize_newton_raphson("theta", experiment.theta(s, t), fn0, gr0);
          }
      }
    }
  }
  update_contributions();

  enforce_positive_parameters();

  if (parameters.targeted(Target::theta_prior))
    sample_global_theta_priors();

  enforce_positive_parameters();

  if (parameters.targeted(Target::field))
    update_fields();

  enforce_positive_parameters();
}

template <typename Type>
Matrix Model<Type>::field_fitness_posterior_gradient(const Matrix &f) const {
  Matrix grad(0, T);
  size_t cumul = 0;
  for (auto &experiment : experiments) {
    grad = arma::join_vert(grad, experiment.field_fitness_posterior_gradient(
                                     f.rows(cumul, cumul + experiment.S - 1)));
    cumul += experiment.S;
  }
  return grad;
}

template <typename Type>
void Model<Type>::update_fields() {
  LOG(verbose) << "Updating fields";

  using namespace LBFGSpp;
  LBFGSParam<double> param;
  param.epsilon = parameters.lbfgs_epsilon;
  param.max_iterations = parameters.lbfgs_iter;
  // Create solver and function object
  LBFGSSolver<double> solver(param);

  for (auto &coord_sys : coordinate_systems) {
    LOG(verbose) << "Updating coordinate system";
    using Vec = Eigen::VectorXd;
    const size_t NT = coord_sys.N * T;
    Vec x(NT);

    LOG(debug) << "initial phi: " << std::endl
               << Stats::summary(coord_sys.field);

    for (size_t i = 0; i < NT; ++i)
      // NOTE log for grad w.r.t. exp-transform
      x[i] = log(coord_sys.field[i]);

    size_t call_cnt = 0;

    auto fnc = [&](const Vec &phi_, Vec &grad_) {
      Matrix phi(coord_sys.N, T);
      for (size_t i = 0; i < NT; ++i)
        phi[i] = exp(phi_[i]);
      Matrix grad;
      double score = field_gradient(coord_sys, phi, grad);
      for (size_t i = 0; i < NT; ++i)
        // NOTE multiply w phi to compute gradient of exp-transform
        grad_[i] = grad[i] * phi[i];
      if (((call_cnt++) % parameters.lbfgs_report_interval) == 0) {
        LOG(debug) << "Field score = " << score;
        LOG(debug) << "phi: " << std::endl << Stats::summary(phi);
        LOG(debug) << "grad: " << std::endl << Stats::summary(grad);
        Matrix tmp = grad % phi;
        LOG(debug) << "grad*phi: " << std::endl << Stats::summary(tmp);
      }
      return score;
    };

    double fx;
    int niter = solver.minimize(fnc, x, fx);

    for (size_t i = 0; i < NT; ++i)
      coord_sys.field[i] = exp(x[i]);

    LOG(verbose) << "LBFGS performed " << niter << " iterations";
    LOG(verbose) << "LBFGS evaluated function and gradient " << call_cnt
                 << " times";
    LOG(verbose) << "LBFGS achieved f(x) = " << fx;
    LOG(verbose) << "LBFGS field summary: " << std::endl
                 << Stats::summary(coord_sys.field);
  }
  update_experiment_fields();
}

template <typename Type>
double Model<Type>::field_gradient(CoordinateSystem &coord_sys,
                                   const Matrix &phi, Matrix &grad) const {
  LOG(debug) << "phi dim " << phi.n_rows << "x" << phi.n_cols;
  double score = 0;

  Matrix fitness(0, T);
  size_t cumul = 0;
  for (auto member : coord_sys.members) {
    double s = -arma::accu(experiments[member].field_fitness_posterior(
        phi.rows(cumul, cumul + experiments[member].S - 1)));
    LOG(debug) << "Fitness contribution to score of sample " << member << ": "
               << s;
    score += s;
    fitness = arma::join_vert(
        fitness, experiments[member].field_fitness_posterior_gradient(
                     phi.rows(cumul, cumul + experiments[member].S - 1)));
    cumul += experiments[member].S;
  }

  LOG(debug) << "Fitness contribution to score: " << score;

  grad = Matrix(coord_sys.N, T, arma::fill::zeros);

  grad.rows(0, coord_sys.S - 1) = -fitness + grad.rows(0, coord_sys.S - 1);

  if (parameters.field_lambda_dirichlet != 0) {
    Matrix grad_dirichlet = Matrix(coord_sys.N, T, arma::fill::zeros);
    for (size_t t = 0; t < T; ++t) {
      grad_dirichlet.col(t)
          = coord_sys.mesh.grad_dirichlet_energy(Vector(phi.col(t)));

      double s = parameters.field_lambda_dirichlet
                 * coord_sys.mesh.sum_dirichlet_energy(Vector(phi.col(t)));
      score += s;
      LOG(debug) << "Smoothness contribution to score of factor " << t << ": "
                 << s;
    }
    grad = grad + grad_dirichlet * parameters.field_lambda_dirichlet;
  }

  if (parameters.field_lambda_laplace != 0) {
    Matrix grad_laplace = Matrix(coord_sys.N, T, arma::fill::zeros);
    for (size_t t = 0; t < T; ++t) {
      grad_laplace.col(t)
          = coord_sys.mesh.grad_sq_laplace_operator(Vector(phi.col(t)));

      double s = parameters.field_lambda_laplace
                 * coord_sys.mesh.sum_sq_laplace_operator(Vector(phi.col(t)));
      score += s;
      LOG(debug) << "Curvature contribution to score of factor " << t << ": "
                 << s;
    }
    grad = grad + grad_laplace * parameters.field_lambda_laplace;
  }

  LOG(debug) << "Fitness and smoothness score: " << score;

  return score;
}

template <typename Type>
void Model<Type>::enforce_positive_parameters() {
  for (auto &coord_sys : coordinate_systems)
    enforce_positive_and_warn("field", coord_sys.field);
  features.enforce_positive_parameters("global feature");
  for (auto &experiment : experiments)
    experiment.enforce_positive_parameters();
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

  Matrix field(0, T);
  for (auto &experiment : experiments)
    field = arma::join_vert(field, experiment.field);

  mix_prior.sample(observed, field);

  for (auto &experiment : experiments)
    experiment.weights.prior = mix_prior;

  min_max("weights r", mix_prior.r);
  min_max("weights p", mix_prior.p);
}

template <typename Type>
double Model<Type>::log_likelihood_conv_NB_counts() const {
  double l = 0;
  for (auto &experiment : experiments)
    l += experiment.log_likelihood_conv_NB_counts();
  return l;
}

template <typename Type>
double Model<Type>::log_likelihood(const std::string &prefix) const {
  double l = 0;
  for (size_t e = 0; e < E; ++e) {
    Matrix m = experiments[e].log_likelihood(features);

    auto &gene_names = experiments[e].data.row_names;
    auto &spot_names = experiments[e].data.col_names;

    std::string exp_prefix = prefix + "experiment"
                             + to_string_embedded(e, EXPERIMENT_NUM_DIGITS)
                             + "-";
    write_matrix(m, exp_prefix + "loglikelihood" + FILENAME_ENDING,
                 parameters.compression_mode, gene_names, spot_names);
    write_matrix(experiments[e].data.counts,
                 exp_prefix + "loglikelihood-counts" + FILENAME_ENDING,
                 parameters.compression_mode, gene_names, spot_names);
    for (auto &x : m)
      l += x;
  }
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
void Model<Type>::update_experiment_fields() {
  LOG(verbose) << "Updating experiment fields";
  for (auto &coord_sys : coordinate_systems) {
    size_t cumul = 0;
    for (auto member : coord_sys.members) {
      experiments[member].field
          = coord_sys.field.rows(cumul, cumul + experiments[member].S - 1);
      cumul += experiments[member].S;
    }
  }
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
void Model<Type>::initialize_coordinate_systems(double v) {
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
    size_t dim = experiments[coord_sys.members[0]].coords.n_cols;
    std::vector<Point> pts;
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
          mi = mi - parameters.mesh_hull_distance;
          ma = ma + parameters.mesh_hull_distance;
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
              if (norm(pt - pts[s]) < parameters.mesh_hull_distance) {
                ok = true;
                break;
              }
          }

          pts.push_back(pt);
        }
      }
    }
    coord_sys.mesh = Mesh(dim, pts);
  }
}

template <typename Type>
void Model<Type>::add_experiment(const Counts &counts, size_t coord_sys) {
  Parameters experiment_parameters = parameters;
  const double factor = parameters.local_phi_scaling_factor;
  experiment_parameters.hyperparameters.phi_p_1 *= factor;
  experiment_parameters.hyperparameters.phi_r_1 *= factor;
  experiment_parameters.hyperparameters.phi_p_2 *= factor;
  experiment_parameters.hyperparameters.phi_r_2 *= factor;
  experiments.push_back({counts, T, experiment_parameters});
  E++;
  // TODO check redundancy with Experiment constructor
  experiments.rbegin()->features.matrix.ones();
  while (coordinate_systems.size() <= coord_sys)
    coordinate_systems.push_back({});
  coordinate_systems[coord_sys].members.push_back(E - 1);
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
