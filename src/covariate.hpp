#ifndef COVARIATE_HPP
#define COVARIATE_HPP

#include <string>
#include <vector>
#include "compression_mode.hpp"
#include "gamma_func.hpp"
#include "odds.hpp"
#include "parallel.hpp"
#include "types.hpp"

struct Covariate {
  std::string label;
  std::vector<std::string> values;
  std::string to_string() const;
};

std::ostream &operator<<(std::ostream &os, const Covariate &covariate);

using Covariates = std::vector<Covariate>;

/** factor loading matrix */
struct CovariateInformation {
  using idxs_t = std::vector<size_t>;
  idxs_t idxs;
  idxs_t vals;
  std::string to_string(const Covariates &covariates) const;
};

struct Coefficient {
  // TODO rename variance to odds
  enum class Variable { rate, variance, prior };
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
    beta_prime
    // linear term
    // log_normal
    // log_gp
  };
  Coefficient(size_t G, size_t T, size_t S, Variable variable, Kind kind,
              CovariateInformation info);
  Variable variable;
  Kind kind;
  Distribution distribution;
  bool gene_dependent() const;
  bool type_dependent() const;
  bool spot_dependent() const;
  size_t size() const;
  STD::Vector setZero();
  STD::Vector vectorize() const;
  CovariateInformation info;
  STD::Matrix values;
  std::vector<size_t> prior_idxs;

  template <typename Fnc>
  void visit(Fnc fnc) const {
    switch (kind) {
      case Kind::scalar:
        fnc(0, 0, 0);
        break;
      case Kind::gene:
#pragma omp parallel for if (DO_PARALLEL)
        for (int g = 0; g < values.rows(); ++g)
          fnc(g, 0, 0);
        break;
      case Kind::type:
#pragma omp parallel for if (DO_PARALLEL)
        for (int t = 0; t < values.rows(); ++t)
          fnc(0, t, 0);
        break;
      case Kind::spot:
#pragma omp parallel for if (DO_PARALLEL)
        for (int s = 0; s < values.rows(); ++s)
          fnc(0, 0, s);
        break;
      case Kind::gene_type:
#pragma omp parallel for if (DO_PARALLEL)
        for (int g = 0; g < values.rows(); ++g)
          for (int t = 0; t < values.cols(); ++t)
            fnc(g, t, 0);
        break;
      case Kind::spot_type:
#pragma omp parallel for if (DO_PARALLEL)
        for (int s = 0; s < values.rows(); ++s)
          for (int t = 0; t < values.cols(); ++t)
            fnc(0, t, s);
        break;
    }
  }

  void compute_gradient(const std::vector<Coefficient> &coeffs,
                        std::vector<Coefficient> &grad_coeffs,
                        size_t idx) const;

  template <typename Iter>
  void from_log_vector(Iter &iter) {
    for (auto &x : values)
      x = exp(*iter++);
  };

  double get(size_t g, size_t t, size_t s) const;  // rename to operator()
  double &get(size_t g, size_t t, size_t s);       // rename to operator()
  void store(const std::string &path, CompressionMode,
             const std::vector<std::string> &gene_names,
             const std::vector<std::string> &spot_names,
             const std::vector<std::string> &factor_names,
             std::vector<size_t> col_order) const;
  void restore(const std::string &path);
};

std::string to_string(const Coefficient::Variable &variable);
std::string to_string(const Coefficient::Kind &kind);
std::string to_token(const Coefficient::Kind &kind);

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

#endif
