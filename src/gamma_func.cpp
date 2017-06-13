#include "gamma_func.hpp"
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/trigamma.hpp>

double digamma(double x) { return boost::math::digamma(x); }

double trigamma(double x) { return boost::math::trigamma(x); }
