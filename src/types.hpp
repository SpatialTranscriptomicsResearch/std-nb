#ifndef TYPES_HPP
#define TYPES_HPP

#include <cstdint>
#include <armadillo>

namespace FactorAnalysis {
using Int = uint32_t;
using Float = double;
using Vector = arma::Col<Float>;
using Matrix = arma::Mat<Float>;
using IMatrix = arma::Mat<Int>;
using Tensor = arma::Cube<Float>;
using ITensor = arma::Cube<Int>;
}

#endif
