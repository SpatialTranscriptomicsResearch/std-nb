#ifndef GP_HPP
#define GP_HPP
#include "types.hpp"

namespace GP {
using Matrix = STD::Matrix;
using Vector = STD::Vector;

struct GaussianProcess {
  GaussianProcess(const Matrix &x, double len_scale);
  Matrix covariance(double spatial_var, double indep_var) const;

  Matrix inverse_covariance(double spatial_var, double indep_var) const;

  // negative one-half squared distances divided by squared length scale
  Matrix rbf_kernel(const Matrix &x, double l);
  size_t n;
  double length_scale;
  // std::shared_ptr<Matrix> x;
  Matrix eigenvectors;
  Vector eigenvalues;
  double calc_mean(Vector y, double delta) const;
  double calc_spatial_variance(const Vector &y, double mean,
                               double delta) const;
  void predict_means_and_vars(const Vector &y, double delta, Vector &mu,
                              Vector &var) const;
};
}
#endif
