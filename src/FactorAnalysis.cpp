#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/trigamma.hpp>
#include "FactorAnalysis.hpp"

namespace FactorAnalysis {
Float digamma(Float x) { return boost::math::digamma(x); }

Float trigamma(Float x) { return boost::math::trigamma(x); }
}
