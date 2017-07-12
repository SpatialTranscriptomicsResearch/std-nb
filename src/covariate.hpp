#ifndef COVARIATE_HPP
#define COVARIATE_HPP

#include <string>
#include <vector>

struct Covariate {
  std::string label;
  std::vector<std::string> values;
  std::string to_string() const;
};

std::ostream &operator<<(std::ostream &os, const Covariate &covariate);

using Covariates = std::vector<Covariate>;

#endif
