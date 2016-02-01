#ifndef TYPES_HPP
#define TYPES_HPP

#include <cstdint>
#include <boost/multi_array.hpp>

namespace FactorAnalysis {
using Int = uint32_t;
using Float = double;
using Vector = boost::multi_array<Float, 1>;
using Matrix = boost::multi_array<Float, 2>;
using IMatrix = boost::multi_array<Int, 2>;
using Tensor = boost::multi_array<Float, 3>;
using ITensor = boost::multi_array<Int, 3>;
}

#endif
