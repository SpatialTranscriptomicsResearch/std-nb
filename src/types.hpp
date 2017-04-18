#ifndef TYPES_HPP
#define TYPES_HPP

#include <cstdint>
#include <eigen3/Eigen/Dense>

namespace STD {
using Int = uint32_t;
using Float = double;
using Index = Eigen::MatrixXd::Index;
using Vector = Eigen::VectorXd;
using IVector = Eigen::Matrix<Int, Eigen::Dynamic, 1>;
using Matrix = Eigen::MatrixXd;
}

namespace Eigen {
MatrixXd::Scalar* begin(MatrixXd& m);
MatrixXd::Scalar* end(MatrixXd& m);
VectorXd::Scalar* begin(VectorXd& m);
VectorXd::Scalar* end(VectorXd& m);

const MatrixXd::Scalar* begin(const MatrixXd& m);
const MatrixXd::Scalar* end(const MatrixXd& m);
const VectorXd::Scalar* begin(const VectorXd& m);
const VectorXd::Scalar* end(const VectorXd& m);
}
#endif
