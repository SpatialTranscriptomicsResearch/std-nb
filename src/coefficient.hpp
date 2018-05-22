#ifndef COEFFICIENT_HPP
#define COEFFICIENT_HPP

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include "covariate.hpp"
#include "gp.hpp"
#include "types.hpp"

namespace STD {
struct Experiment;
}

namespace Coefficient {

struct Coefficient;
}

using CoefficientPtr = std::shared_ptr<Coefficient::Coefficient>;

namespace Coefficient {
enum class Kind {
  scalar = 0,
  gene = 1,
  spot = 2,
  type = 4,
  gene_type = 5,
  spot_type = 6
  // TODO make gene_spot and gene_spot_type illegal
};

enum class Type {
  fixed,
  gamma,
  beta,
  beta_prime,
  // linear term
  // file for quantitative covariates;
  //   notation:
  //     myfile ~ file(/where/the/file/is)
  //     rate = rate(gene) + rate(type) + ... + myfile(gene) * rate_coeff(gene)
  // continuous distributions:
  //   exponential
  //   multi-variate normal
  // discrete distributions:
  //   Bernoulli
  //   Binomial
  //   Poisson
  normal,
  gp_points,
  gp_coord
};

struct Parameters {
  double variance = 0.1;
};

struct Id {
  std::string name;
  Kind kind;
  Type type;
  CovariateInformation info;
};

struct Coefficient : public Id {
  Coefficient(size_t G, size_t T, size_t S, const Id &id,
              const Parameters &params,
              const std::vector<CoefficientPtr> &priors_);

  Parameters parameters;
  STD::Matrix values;
  std::vector<CoefficientPtr> priors;
  std::vector<STD::Experiment *> experiments;

  bool parent_a_flexible;
  bool parent_b_flexible;
  bool gene_dependent() const;
  bool type_dependent() const;
  bool spot_dependent() const;

  size_t size() const;
  size_t number_variable() const;

  template <typename Iter>
  void from_vector(Iter &iter) {
    for (auto &x : values)
      x = *iter++;
  };

  STD::Vector vectorize() const;
  std::string to_string() const;

  virtual double compute_gradient(CoefficientPtr grad_coeff) const = 0;

  double get_raw(size_t g, size_t t, size_t s) const;  // rename to operator()
  double &get_raw(size_t g, size_t t, size_t s);       // rename to operator()
  double get_actual(size_t g, size_t t, size_t s) const;  // rename to operator()
  virtual void sample() = 0;
  void store(const std::string &path, CompressionMode,
             const std::vector<std::string> &gene_names,
             const std::vector<std::string> &spot_names,
             const std::vector<std::string> &factor_names,
             const std::vector<size_t> col_order) const;
  void restore(const std::string &path);

  template <typename Fnc>
  double visit(Fnc fnc) const {
    double score = 0;
    switch (kind) {
      case Kind::scalar:
        score = fnc(0, 0, 0);
        break;
      case Kind::gene:
#pragma omp parallel for reduction(+ : score) if (DO_PARALLEL)
        for (int g = 0; g < values.rows(); ++g)
          score += fnc(g, 0, 0);
        break;
      case Kind::type:
#pragma omp parallel for reduction(+ : score) if (DO_PARALLEL)
        for (int t = 0; t < values.rows(); ++t)
          score += fnc(0, t, 0);
        break;
      case Kind::spot:
#pragma omp parallel for reduction(+ : score) if (DO_PARALLEL)
        for (int s = 0; s < values.rows(); ++s)
          score += fnc(0, 0, s);
        break;
      case Kind::gene_type:
#pragma omp parallel for reduction(+ : score) if (DO_PARALLEL)
        for (int g = 0; g < values.rows(); ++g)
          for (int t = 0; t < values.cols(); ++t)
            score += fnc(g, t, 0);
        break;
      case Kind::spot_type:
#pragma omp parallel for reduction(+ : score) if (DO_PARALLEL)
        for (int s = 0; s < values.rows(); ++s)
          for (int t = 0; t < values.cols(); ++t)
            score += fnc(0, t, s);
        break;
    }
    return score;
  }
};

struct Fixed : public Coefficient {
  Fixed(size_t G, size_t T, size_t S, const Id &id, const Parameters &params);
  double compute_gradient(CoefficientPtr grad_coeff) const { return 0; };
  void sample();
};

struct Distributions : public Coefficient {
  Distributions(size_t G, size_t T, size_t S, const Id &id,
                const Parameters &params,
                const std::vector<CoefficientPtr> &priors);
};
struct Beta : public Distributions {
  Beta(size_t G, size_t T, size_t S, const Id &id, const Parameters &params,
       const std::vector<CoefficientPtr> &priors);
  double compute_gradient(CoefficientPtr grad_coeff) const;
  void sample();
};
struct BetaPrime : public Distributions {
  BetaPrime(size_t G, size_t T, size_t S, const Id &id,
            const Parameters &params,
            const std::vector<CoefficientPtr> &priors);
  double compute_gradient(CoefficientPtr grad_coeff) const;
  void sample();
};
struct Normal : public Distributions {
  Normal(size_t G, size_t T, size_t S, const Id &id, const Parameters &params,
         const std::vector<CoefficientPtr> &priors);
  double compute_gradient(CoefficientPtr grad_coeff) const;
  void sample();
};
struct Gamma : public Distributions {
  Gamma(size_t G, size_t T, size_t S, const Id &id, const Parameters &params,
        const std::vector<CoefficientPtr> &priors);
  double compute_gradient(CoefficientPtr grad_coeff) const;
  void sample();
};

namespace Spatial {
struct Points : public Coefficient {
  Points(size_t G, size_t T, size_t S, const Id &id, const Parameters &params,
         const std::vector<CoefficientPtr> &priors);
  double compute_gradient(CoefficientPtr grad_coeff) const { return 0; };
  void sample();
};

struct Coord : public Coefficient {
  using PointsPtr = std::shared_ptr<Points>;
  using PointsPtrs = std::vector<PointsPtr>;
  Coord(size_t G, size_t T, size_t S, const Id &id, const Parameters &params,
        const std::vector<CoefficientPtr> &priors);
  PointsPtrs points;
  double length_scale;
  std::shared_ptr<GP::GaussianProcess> gp;
  STD::Matrix form_data() const;
  STD::Matrix form_mean() const;
  STD::Vector form_priors(size_t prior_idx) const;
  STD::Vector form_svs() const;
  STD::Vector form_deltas() const;
  void subtract_mean();
  void construct_gp();
  size_t size() const;
  void add_formed_data(const STD::Matrix &m, bool subtract_prior);
  double compute_gradient(CoefficientPtr grad_coeff) const;
  void sample();
};

}  // namespace Spatial

size_t distribution_number_parameters(Type distribution);

Kind determine_kind(const std::set<std::string> &term);

std::string to_string(Kind kind);
std::string to_string(Type distribution);
std::string to_token(Kind kind);
std::string storage_type(Kind kind);
std::ostream &operator<<(std::ostream &os, const Coefficient &coeff);

inline constexpr Kind operator&(Kind a, Kind b) {
  return static_cast<Kind>(static_cast<int>(a) & static_cast<int>(b));
}

inline constexpr Kind operator|(Kind a, Kind b) {
  return static_cast<Kind>(static_cast<int>(a) | static_cast<int>(b));
}

inline constexpr Kind operator^(Kind a, Kind b) {
  return static_cast<Kind>(static_cast<int>(a) & static_cast<int>(b));
}

inline constexpr Kind operator~(Kind a) {
  return static_cast<Kind>((~static_cast<int>(a)) & ((1 << 11) - 1));
}

CoefficientPtr make_shared(size_t G, size_t T, size_t S, const Id &id,
                           const Parameters &params,
                           const std::vector<CoefficientPtr> &priors);

}  // namespace Coefficient

#endif
