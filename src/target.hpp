#ifndef TARGET_HPP
#define TARGET_HPP

#include <iostream>

namespace STD {

enum class Target {
  empty = 0,
  global = 1 << 0,
  local = 1 << 1,
  baseline = 1 << 2,
  theta = 1 << 3,
  theta_prior = 1 << 4,
  spot = 1 << 5,
  field = 1 << 6,
  variance = 1 << 7,
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
  return Target::global | Target::variance | Target::theta | Target::theta_prior
         | Target::local | Target::baseline | Target::spot;
}

inline bool flagged(Target x) { return (Target::empty | x) != Target::empty; }
}

#endif
