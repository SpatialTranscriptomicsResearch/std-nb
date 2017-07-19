#ifndef TARGET_HPP
#define TARGET_HPP

#include <iostream>

namespace STD {

enum class Target {
  empty = 0,
  covariates = 1 << 0,
  gamma_prior = 1 << 1,
  theta = 1 << 2,
  theta_prior = 1 << 3,
  spot = 1 << 4,
  field = 1 << 5,
  rho = 1 << 6,
  rho_prior = 1 << 7,
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
  return static_cast<Target>((~static_cast<int>(a)) & ((1 << 8) - 1));
}

inline constexpr Target DefaultTarget() {
  return Target::covariates | Target::rho | Target::theta | Target::theta_prior
         | Target::spot;
}

inline bool flagged(Target x) { return (Target::empty | x) != Target::empty; }
}

#endif
