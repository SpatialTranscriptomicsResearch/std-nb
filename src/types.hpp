#ifndef TYPES_HPP
#define TYPES_HPP

#include <cstdint>
// #define ARMA_NO_DEBUG
#include <armadillo>

namespace STD {
using Int = uint32_t;
using Float = double;
using Vector = arma::Col<Float>;
using IVector = arma::Col<Int>;
using Matrix = arma::Mat<Float>;
using IMatrix = arma::Mat<Int>;
using Tensor = arma::Cube<Float>;
using ITensor = arma::Cube<Int>;
}

#endif
