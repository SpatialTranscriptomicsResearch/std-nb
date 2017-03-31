#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/trigamma.hpp>
#include "parameters.hpp"
#include "aux.hpp"

using namespace std;

namespace STD {
Float digamma(Float x) { return boost::math::digamma(x); }

Float trigamma(Float x) { return boost::math::trigamma(x); }

bool Parameters::targeted(Target target) const {
  return flagged(targets & target);
}
}
