#ifndef TARGET_HPP
#define TARGET_HPP

#include <iostream>

namespace STD {

enum class Target {
  empty = 0,
  covariates_scalar =    1 << 0,
  covariates_gene =      1 << 1,
  covariates_type =      1 << 2,
  covariates_gene_type = 1 << 3,
  covariates =          (1 << 4) - 1,
  gamma_prior =          1 << 4,
  theta =                1 << 5,
  theta_prior =          1 << 6,
  spot =                 1 << 7,
  field =                1 << 8,
  rho =                  1 << 9,
  rho_prior =            1 << 10,
};

std::ostream &operator<<(std::ostream &os, const Target &which);
std::istream &operator>>(std::istream &is, Target &which);

inline constexpr Target operator&(Target a, Target b) {
  return static_cast<Target>(static_cast<int>(a) & static_cast<int>(b));
}

inline constexpr Target operator|(Target a, Target b) {
  return static_cast<Target>(static_cast<int>(a) | static_cast<int>(b));
}

inline constexpr Target operator^(Target a, Target b) {
  return static_cast<Target>(static_cast<int>(a) & static_cast<int>(b));
}

inline constexpr Target operator~(Target a) {
  return static_cast<Target>((~static_cast<int>(a)) & ((1 << 11) - 1));
}

inline constexpr Target DefaultTarget() {
  return Target::covariates | Target::rho | Target::theta | Target::theta_prior
         | Target::spot;
}

inline constexpr Target DefaultForget() {
  return Target::theta;
}

inline bool flagged(Target x) { return (Target::empty | x) != Target::empty; }
}

#endif
