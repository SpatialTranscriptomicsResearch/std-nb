#ifndef COEFFICIENT_HPP
#define COEFFICIENT_HPP

#include <memory>
#include <string>
#include <vector>
#include "covariate.hpp"
#include "gp.hpp"
#include "types.hpp"

enum class DistributionMode { log_normal, gamma_odds, gamma_odds_log_normal };

std::string to_string(DistributionMode mode);
DistributionMode distribution_from_string(const std::string &s);
std::ostream &operator<<(std::ostream &os, DistributionMode mode);
std::istream &operator>>(std::istream &is, DistributionMode &mode);

struct Coefficient {
  enum class Variable { rate, odds, prior };
  enum class Kind {
    scalar = 0,
    gene = 1,
    spot = 2,
    type = 4,
    gene_type = 5,
    spot_type = 6
    // TODO make gene_spot and gene_spot_type illegal
  };
  enum class Distribution {
    fixed,
    gamma,
    beta_prime,
    // linear term
    log_normal,
    log_gp
  };
  Coefficient(size_t G, size_t T, size_t S, Variable variable, Kind kind,
              Distribution distribution,
              std::shared_ptr<GP::GaussianProcess> gp,
              CovariateInformation info);
  Variable variable;
  Kind kind;
  Distribution distribution;
  std::shared_ptr<GP::GaussianProcess> gp;

  CovariateInformation info;
  STD::Matrix values;
  std::vector<size_t> prior_idxs;
  std::vector<size_t> experiment_idxs;

  bool gene_dependent() const;
  bool type_dependent() const;
  bool spot_dependent() const;

  size_t size() const;
  size_t number_parameters() const;
  STD::Vector setZero();

  template <typename Iter>
  void from_vector(Iter &iter) {
    for (auto &x : values)
      x = *iter++;
  };

  STD::Vector vectorize() const;
  std::string to_string() const;

  double compute_gradient(const std::vector<Coefficient> &coeffs,
                          std::vector<Coefficient> &grad_coeffs,
                          size_t idx) const;

  double get(size_t g, size_t t, size_t s) const;  // rename to operator()
  double &get(size_t g, size_t t, size_t s);       // rename to operator()
  void store(const std::string &path, CompressionMode,
             const std::vector<std::string> &gene_names,
             const std::vector<std::string> &spot_names,
             const std::vector<std::string> &factor_names,
             std::vector<size_t> col_order) const;
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

std::string to_string(const Coefficient::Variable &variable);
std::string to_string(const Coefficient::Kind &kind);
std::string to_string(const Coefficient::Distribution &distribution);
std::string to_token(const Coefficient::Kind &kind);

std::ostream &operator<<(std::ostream &os, const Coefficient &coeff);

inline constexpr Coefficient::Kind operator&(Coefficient::Kind a,
                                             Coefficient::Kind b) {
  return static_cast<Coefficient::Kind>(static_cast<int>(a)
                                        & static_cast<int>(b));
}

inline constexpr Coefficient::Kind operator|(Coefficient::Kind a,
                                             Coefficient::Kind b) {
  return static_cast<Coefficient::Kind>(static_cast<int>(a)
                                        | static_cast<int>(b));
}

inline constexpr Coefficient::Kind operator^(Coefficient::Kind a,
                                             Coefficient::Kind b) {
  return static_cast<Coefficient::Kind>(static_cast<int>(a)
                                        & static_cast<int>(b));
}

inline constexpr Coefficient::Kind operator~(Coefficient::Kind a) {
  return static_cast<Coefficient::Kind>((~static_cast<int>(a))
                                        & ((1 << 11) - 1));
}

Coefficient::Distribution choose_distribution(Coefficient::Variable variable,
                                              Coefficient::Kind kind,
                                              DistributionMode mode,
                                              bool use_gp);
#endif
