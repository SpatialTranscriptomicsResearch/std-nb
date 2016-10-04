#ifndef TARGET_HPP
#define TARGET_HPP

#include <iostream>

namespace PoissonFactorization {

enum class Target {
  empty = 0,
  contributions = 1 << 0,
  phi = 1 << 1,
  phi_prior = 1 << 2,
  phi_local = 1 << 3,
  phi_prior_local = 1 << 4,
  theta = 1 << 5,
  theta_prior = 1 << 6,
  spot = 1 << 7,
  baseline = 1 << 8,
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
  return static_cast<Target>((~static_cast<int>(a)) & ((1 << 9) - 1));
}

inline constexpr Target DefaultTarget() {
  return Target::contributions | Target::phi | Target::phi_prior
         | Target::phi_local | Target::theta | Target::theta_prior
         | Target::spot | Target::baseline;
}

inline bool flagged(Target x) { return (Target::empty | x) != Target::empty; }
}

#endif
