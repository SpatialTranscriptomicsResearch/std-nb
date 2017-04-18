#include "types.hpp"
namespace Eigen {
MatrixXd::Scalar* begin(MatrixXd& m) { return m.data(); }
MatrixXd::Scalar* end(MatrixXd& m) { return m.data() + m.size(); }
VectorXd::Scalar* begin(VectorXd& v) { return v.data(); }
VectorXd::Scalar* end(VectorXd& v) { return v.data() + v.size(); }
ArrayXd::Scalar* begin(ArrayXd& v) { return v.data(); }
ArrayXd::Scalar* end(ArrayXd& v) { return v.data() + v.size(); }

const MatrixXd::Scalar* begin(const MatrixXd& m) { return m.data(); }
const MatrixXd::Scalar* end(const MatrixXd& m) { return m.data() + m.size(); }
const VectorXd::Scalar* begin(const VectorXd& v) { return v.data(); }
const VectorXd::Scalar* end(const VectorXd& v) { return v.data() + v.size(); }
const ArrayXd::Scalar* begin(const ArrayXd& v) { return v.data(); }
const ArrayXd::Scalar* end(const ArrayXd& v) { return v.data() + v.size(); }
}
