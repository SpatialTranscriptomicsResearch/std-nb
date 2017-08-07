#include "gp.hpp"
#include <eigen3/Eigen/Eigenvalues>
#include "log.hpp"
#include "types.hpp"

namespace GP {
GaussianProcess::GaussianProcess(const Matrix &x, double len_scale)
    : n(x.rows()), length_scale(len_scale) {
  LOG(verbose) << "GaussianProcess::GaussianProcess()";
  Matrix K = rbf_kernel(x, length_scale);
  Eigen::SelfAdjointEigenSolver<Matrix> es(K);
  eigenvectors = es.eigenvectors();
  eigenvalues = es.eigenvalues();
}

Matrix GaussianProcess::covariance(double spatial_var, double indep_var) const {
  Vector diag = eigenvalues.array() + indep_var;
  return 1 / spatial_var * eigenvectors * diag.asDiagonal()
         * eigenvectors.transpose();
}

Matrix GaussianProcess::inverse_covariance(double spatial_var,
                                           double indep_var) const {
  Vector diag = 1 / (eigenvalues.array() + indep_var);
  return 1 / spatial_var * eigenvectors * diag.asDiagonal()
         * eigenvectors.transpose();
}

// negative one-half squared distances divided by squared length scale
Matrix GaussianProcess::rbf_kernel(const Matrix &x, double l) {
  const size_t n = x.rows();
  const double l_square = l * l;
  Matrix k = Matrix::Zero(n, n);
  for (size_t i = 0; i < n; ++i)
    for (size_t j = i + 1; j < n; ++j) {
      Vector diff = x.row(i) - x.row(j);
      double d = diff.dot(diff);
      k(i, j) = k(j, i) = exp(-1 / 2.0 * d / l_square);
    }
  k.diagonal().array() = 1;
  return k;
}

double GaussianProcess::calc_mean(Vector y, double delta) {
  Vector u1 = eigenvectors.transpose() * Vector::Ones(n);
  Vector uy = eigenvectors.transpose() * y;
  double numerator = 0;
  for (size_t i = 0; i < n; ++i)
    numerator += u1(i) * uy(i) / (eigenvalues(i) + delta);
  double denominator = 0;
  for (size_t i = 0; i < n; ++i)
    denominator += u1(i) * u1(i) / (eigenvalues(i) + delta);
  return numerator / denominator;
}

double GaussianProcess::calc_spatial_variance(const Vector &y, double mean,
                                              double delta) {
  Vector u1 = eigenvectors.transpose() * Vector::Ones(n) * mean;
  Vector uy = eigenvectors.transpose() * y;
  double sigma = 0;
  for (size_t i = 0; i < n; ++i) {
    double diff = uy(i) - u1(i);
    sigma += diff * diff / (eigenvalues(i) + delta);
  }
  return sigma / n;
}

void GaussianProcess::predict_means_and_vars(const Vector &y, double delta,
                                             Vector &mu, Vector &var) {
  double mean = calc_mean(y, delta);
  double sv = calc_spatial_variance(y, mean, delta);
  LOG(verbose) << "GP: mean = " << mean << " sv = " << sv
               << " delta = " << delta;

  Matrix inverse = inverse_covariance(sv, delta);

  Vector y_minus_mean = y.array() - mean;
  for (size_t i = 0; i < n; ++i) {
    mu(i) = 0;
    for (size_t j = 0; j < n; ++j)
      if (i != j)
        mu(i) += y_minus_mean(j) * inverse(i, j);
    mu(i) *= -1 / inverse(i, i);
    var(i) = 1 / inverse(i, i);
  }
}
}
