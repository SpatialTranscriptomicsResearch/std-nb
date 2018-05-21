#ifndef GP_HPP
#define GP_HPP
#include <vector>
#include "entropy.hpp"
#include "log.hpp"
#include "types.hpp"

namespace GP {
using Matrix = STD::Matrix;
using Vector = STD::Vector;

struct GaussianProcess {
  GaussianProcess() = delete;
  GaussianProcess(const Matrix &x, double len_scale);
  Matrix covariance(double spatial_var, double indep_var) const;
  Matrix covariance_sqroot(double spatial_var, double indep_var) const;

  Matrix inverse_covariance_eigen(double spatial_var, double indep_var) const;
  Matrix inverse_covariance(double spatial_var, double indep_var) const;

  // negative one-half squared distances divided by squared length scale
  Matrix rbf_kernel(const Matrix &x, double l);
  size_t n;
  double length_scale;
  Matrix eigenvectors;
  Vector eigenvalues;
  struct VarGrad {
    double sv;
    double delta;
  };
  template <typename V>
  VarGrad predict_means_and_vars(const Vector &y, const Vector &mean, double sv,
                                 double delta, V &&mu, V &&var) const {
    LOG(debug) << "Predicting means and variances for a vector of length "
               << y.size() << " sv = " << sv << " delta = " << delta;
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

    // TODO compute variance gradient
    double standard_score = y_minus_mean.transpose() * inverse * y_minus_mean;
    // TODO why is it not -n rather than -1 ?
    double grad_sv = 0.5 * (standard_score - 1);

    VarGrad grad = {grad_sv, 0.0};
    return grad;
  }

  std::vector<VarGrad> predict_means_and_vars(const Matrix &ys,
                                              const Matrix &means,
                                              const Vector &sv,
                                              const Vector &delta, Matrix &mu,
                                              Matrix &var) const;

  Vector sample(const Vector &mean, double sv, double delta) const;
  Matrix sample(const Matrix &mean, const Vector &sv,
                const Vector &delta) const;
};

}  // namespace GP
#endif
