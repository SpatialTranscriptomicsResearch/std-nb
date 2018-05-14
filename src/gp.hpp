#ifndef GP_HPP
#define GP_HPP
#include <vector>
#include "types.hpp"
#include "log.hpp"

namespace GP {
using Matrix = STD::Matrix;
using Vector = STD::Vector;

struct GaussianProcess {
  GaussianProcess() = delete;
  GaussianProcess(const Matrix &x, double len_scale);
  Matrix covariance(double spatial_var, double indep_var) const;

  Matrix inverse_covariance_eigen(double spatial_var, double indep_var) const;
  Matrix inverse_covariance(double spatial_var, double indep_var) const;

  // negative one-half squared distances divided by squared length scale
  Matrix rbf_kernel(const Matrix &x, double l);
  size_t n;
  double length_scale;
  Matrix eigenvectors;
  Vector eigenvalues;
  template <typename V>
  void predict_means_and_vars(const Vector &y, const Vector &mean, double sv,
                              double delta, V &&mu, V &&var) const {
    LOG(verbose) << "Predicting means and variances for a vector of length " << y.size() << " sv = " << sv << " delta = " << delta;
    assert(sv > 0);

    Matrix inverse = inverse_covariance_eigen(sv, delta);

    Vector y_minus_mean = y - mean;
    for (size_t i = 0; i < n; ++i) {
      mu(i) = 0;
      for (size_t j = 0; j < n; ++j)
        if (i != j)
          mu(i) += y_minus_mean(j) * inverse(i, j);
      mu(i) *= -1 / inverse(i, i);
      var(i) = 1 / inverse(i, i);
    }
    mu = mu + mean;
  }

  void predict_means_and_vars(const Matrix &ys, const Matrix &means,
                              const Vector &sv, const Vector &delta, Matrix &mu,
                              Matrix &var) const;
};

}  // namespace GP
#endif
