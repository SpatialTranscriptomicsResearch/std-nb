#ifndef PRIORS_HPP
#define PRIORS_HPP

#include <cstddef>
#include "log.hpp"
#include "entropy.hpp"
#include "parallel.hpp"
#include "odds.hpp"
#include "sampling.hpp"
#include "parameters.hpp"
#include "types.hpp"

namespace PoissonFactorization {
namespace PRIOR {
namespace PHI {

struct Gamma {
  size_t dim1, dim2;
  /** shape parameter for the prior of the loading matrix */
  Matrix r;
  /** scale parameter for the prior of the loading matrix */
  /* Stored as negative-odds */
  Matrix p;
  Parameters parameters;

  Gamma(size_t dim1_, size_t dim2_, const Parameters &params);
  Gamma(const Gamma &other);
  /** sample p_phi and r_phi */
  /* This chooses first r then p by maximum likelihood */
  template <typename Type, typename... Args>
  void sample(const Type& experiment, const Args&... args);

  /* This is a simple Metropolis-Hastings sampling scheme */
  void sample_mh(const Matrix &theta, const IMatrix &contributions_gene_type,
                 const Vector &spot_scaling, Float experiment_scaling);

  void store(const std::string &prefix,
             const std::vector<std::string> &gene_names,
             const std::vector<std::string> &factor_names) const;

  void lift_sub_model(const Gamma &sub_model, size_t t1, size_t t2);

private:
  void initialize_r();
  void initialize_p();
};

template <typename F, typename... Args>
size_t solve_newton(double eps, F fnc, F dfnc, double &x, Args... args) {
  size_t n = 0;
  double f = fnc(x, args...);
  while (fabs(f = fnc(x, args...)) > eps) {
    double df = dfnc(x, args...);
    LOG(verbose) << "x = " << x << " f = " << f << " df = " << df;
    double ratio = f / df;
    if(ratio > x)
      x /= 2;
    else
      x -= f / df;
    n++;
  }
  return n;
}

double fnc(double r, double x);
double dfnc(double r, double x);
double fnc2(double r, double x, double gamma, double theta);
double dfnc2(double r, double x, double gamma, double theta);

template <typename Type, typename... Args>
void Gamma::sample(const Type &experiment, const Args&... args) {
  LOG(info) << "Sampling P and R of Φ using maximum likelihood and from the posterior, respectively.";

  auto expected_gene_type = experiment.expected_gene_type(args...);
  for (size_t t = 0; t < experiment.T; ++t) {
#pragma omp parallel if (DO_PARALLEL)
    {
      const size_t thread_num = omp_get_thread_num();
#pragma omp for
      for (size_t g = 0; g < dim1; ++g) {
        const Int count_sum = experiment.contributions_gene_type(g, t);
        const Float weight_sum = expected_gene_type(g, t);
        LOG(debug) << "count_sum = " << count_sum;
        LOG(debug) << "weight_sum = " << weight_sum;
        LOG(debug) << "r(" << g << ", " << t << ") = " << r(g, t);
        LOG(debug) << "p(" << g << ", " << t << ") = " << p(g, t);
        if (count_sum == 0) {
          r(g, t) = std::gamma_distribution<Float>(
              parameters.hyperparameters.phi_r_1,
              1 / parameters.hyperparameters.phi_r_2)(
              EntropySource::rngs[thread_num]);
        } else {
          // TODO should this be deactivated?
          // // set to arithmetic mean of current value and 1
          r(g, t) = (1 + r(g, t)) / 2;
          // size_t num_steps = solve_newton(1e-6, fnc, dfnc, r(g, t), count_sum);
          auto num_steps = solve_newton(1e-6, fnc2, dfnc2, r(g, t), count_sum,
                                        p(g, t), weight_sum);
          LOG(verbose) << "r'(" << g << ", " << t << ") = " << r(g, t);
          LOG(debug) << "number of steps = " << num_steps;
        }

        p(g, t) = sample_compound_gamma(
            parameters.hyperparameters.phi_p_1 + r(g, t),
            parameters.hyperparameters.phi_p_2 + count_sum, weight_sum,
            EntropySource::rngs[thread_num]);

        assert(r(g, t) >= 0);
        assert(p(g, t) >= 0);

        LOG(verbose) << "p'(" << g << ", " << t << ") = " << p(g, t);

        if (false)
          if (count_sum > 0) {
            const double pseudo_cnt = 1e-6;
            auto p_ml = r(g, t) / count_sum * weight_sum;
            auto p_ml_ps = r(g, t) / (count_sum + pseudo_cnt) * (weight_sum + pseudo_cnt);

            LOG(verbose) << "p*(" << g << ", " << t << ") = " << p_ml;
            LOG(info) << "pML " << r(g, t) << " " << p(g, t) << " " << p_ml << " " << p_ml_ps;
          }
        LOG(verbose) << std::endl;
      }
    }
  }
}

struct Dirichlet {
  size_t dim1, dim2;
  Float alpha_prior;
  Matrix alpha;

  Dirichlet(size_t dim1_, size_t dim2_, const Parameters &parameters);
  Dirichlet(const Dirichlet &other);
  /** This routine does nothing, as this sub-model doesn't have random variables
   * but only hyper-parameters */
  void sample(const Matrix &theta, const IMatrix &contributions_gene_type,
              const Vector &spot_scaling, Float experiment_scaling) const;
  void store(const std::string &prefix,
             const std::vector<std::string> &gene_names,
             const std::vector<std::string> &factor_names) const;

  void lift_sub_model(const Dirichlet &sub_model, size_t t1, size_t t2) const;
};

/** This routine doesn't print, for the same reason as sample() does nothing */
std::ostream &operator<<(std::ostream &os, const Gamma &x);
std::ostream &operator<<(std::ostream &os, const Dirichlet &x);
}

namespace THETA {

struct Gamma {
  size_t dim1, dim2;
  /** shape parameter for the prior of the mixing matrix */
  Vector r;
  /** scale parameter for the prior of the mixing matrix */
  /* Stored as negative-odds */
  Vector p;
  Parameters parameters;

  Gamma(size_t G_, size_t dim2_, const Parameters &params);
  Gamma(const Gamma &other);
  /** sample p_phi and r_phi */
  /* This is a simple Metropolis-Hastings sampling scheme */
  void sample(const Matrix &phi, const IMatrix &contributions_spot_type,
              const Vector &spot_scaling, Float experiment_scaling);

  void store(const std::string &prefix,
             const std::vector<std::string> &spot_names,
             const std::vector<std::string> &factor_names) const;

  void lift_sub_model(const Gamma &sub_model, size_t t1, size_t t2);

private:
  void initialize_r();
  void initialize_p();
};

struct Dirichlet {
  size_t dim1, dim2;
  Float alpha_prior;
  std::vector<Float> alpha;

  Dirichlet(size_t G_, size_t dim2_, const Parameters &parameters);
  Dirichlet(const Dirichlet &other);
  /** This routine does nothing, as this sub-model doesn't have random variables
   * but only hyper-parameters */
  void sample(const Matrix &phi, const IMatrix &contributions_spot_type,
              const Vector &spot_scaling, Float experiment_scaling) const;
  void store(const std::string &prefix,
             const std::vector<std::string> &spot_names,
             const std::vector<std::string> &factor_names) const;

  void lift_sub_model(const Dirichlet &sub_model, size_t t1, size_t t2) const;
};

/** This routine doesn't print, for the same reason as sampl() does nothing */
std::ostream &operator<<(std::ostream &os, const Gamma &x);
std::ostream &operator<<(std::ostream &os, const Dirichlet &x);
}
}
}

#endif
