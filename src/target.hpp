#ifndef TARGET_HPP
#define TARGET_HPP

#include <iostream>

namespace STD {

enum class Target {
  empty = 0,
  gamma = 1 << 0,
  gamma_prior = 1 << 1,
  lambda = 1 << 2,
  beta = 1 << 3,
  theta = 1 << 4,
  theta_prior = 1 << 5,
  spot = 1 << 6,
  field = 1 << 7,
  rho = 1 << 8,
  rho_prior = 1 << 9,
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
  return static_cast<Target>((~static_cast<int>(a)) & ((1 << 10) - 1));
}

inline constexpr Target DefaultTarget() {
  return Target::gamma | Target::rho | Target::theta | Target::theta_prior
         | Target::lambda | Target::beta | Target::spot;
}

inline bool flagged(Target x) { return (Target::empty | x) != Target::empty; }
}

#endif
