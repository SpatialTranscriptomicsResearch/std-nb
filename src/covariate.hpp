#ifndef COVARIATE_HPP
#define COVARIATE_HPP

#include <string>
#include <vector>
#include "compression_mode.hpp"
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

struct CovariateTerm {
  enum class Kind {
    scalar = 0,
    gene = 1,
    spot = 2,
    type = 4,
    gene_type = 5,
    spot_type = 6
  };
  CovariateTerm(size_t G, size_t T, size_t S, Kind kind,
                CovariateInformation info);
  Kind kind;
  bool gene_dependent() const;
  bool type_dependent() const;
  bool spot_dependent() const;
  CovariateInformation info;
  STD::Matrix values;
  double get(size_t g, size_t t, size_t s) const;  // rename to operator()
  double &get(size_t g, size_t t, size_t s);  // rename to operator()
  void store(const std::string &path, CompressionMode,
             const std::vector<std::string> &gene_names,
             const std::vector<std::string> &spot_names,
             const std::vector<std::string> &factor_names,
             std::vector<size_t> col_order) const;
  void restore(const std::string &path);
};

std::string to_string(const CovariateTerm::Kind &kind);
std::string to_token(const CovariateTerm::Kind &kind);

inline constexpr CovariateTerm::Kind operator&(CovariateTerm::Kind a,
                                               CovariateTerm::Kind b) {
  return static_cast<CovariateTerm::Kind>(static_cast<int>(a)
                                          & static_cast<int>(b));
}

inline constexpr CovariateTerm::Kind operator|(CovariateTerm::Kind a,
                                               CovariateTerm::Kind b) {
  return static_cast<CovariateTerm::Kind>(static_cast<int>(a)
                                          | static_cast<int>(b));
}

inline constexpr CovariateTerm::Kind operator^(CovariateTerm::Kind a,
                                               CovariateTerm::Kind b) {
  return static_cast<CovariateTerm::Kind>(static_cast<int>(a)
                                          & static_cast<int>(b));
}

inline constexpr CovariateTerm::Kind operator~(CovariateTerm::Kind a) {
  return static_cast<CovariateTerm::Kind>((~static_cast<int>(a))
                                          & ((1 << 11) - 1));
}

#endif
