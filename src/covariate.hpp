#ifndef COVARIATE_HPP
#define COVARIATE_HPP

#include <string>
#include <vector>
#include "compression_mode.hpp"
#include "gamma_func.hpp"
#include "odds.hpp"
#include "parallel.hpp"

struct Covariate {
  std::string label;
  std::vector<std::string> values;
  std::string to_string() const;
};

std::ostream &operator<<(std::ostream &os, const Covariate &covariate);

using Covariates = std::vector<Covariate>;

struct CovariateInformation {
  using idxs_t = std::vector<size_t>;
  idxs_t idxs;
  idxs_t vals;
  std::string to_string(const Covariates &covariates) const;
  bool operator==(const CovariateInformation &other) const;
  bool operator<(const CovariateInformation &other) const;
};

#endif
