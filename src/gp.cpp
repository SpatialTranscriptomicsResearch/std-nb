#include "gp.hpp"
#include <eigen3/Eigen/Eigenvalues>
#include "log.hpp"
#include "parallel.hpp"
#include "types.hpp"

using namespace std;

namespace GP {

GaussianProcess::GaussianProcess(const Matrix &x, double len_scale)
    : n(x.rows()), length_scale(len_scale) {
  LOG(debug) << "GaussianProcess::GaussianProcess(n=" << n
             << ", length_scale=" << length_scale << ")";
  Matrix K = rbf_kernel(x, length_scale);
  Eigen::SelfAdjointEigenSolver<Matrix> es(K);
  eigenvectors = es.eigenvectors();
  eigenvalues = es.eigenvalues();
}

Matrix GaussianProcess::covariance(double spatial_var, double indep_var) const {
  Vector diag = eigenvalues.array() + indep_var;
  return spatial_var * eigenvectors * diag.asDiagonal()
         * eigenvectors.transpose();
}

Matrix GaussianProcess::covariance_sqroot(double spatial_var,
                                          double indep_var) const {
  Vector diag = (spatial_var * (eigenvalues.array() + indep_var)).sqrt();
  return eigenvectors * diag.asDiagonal();
}

Matrix GaussianProcess::inverse_covariance_eigen(double spatial_var,
                                                 double indep_var) const {
  LOG(debug) << "Computing inverse covariance matrix for spatial variance = "
             << spatial_var << " and iid noise = " << indep_var;
  Vector diag = 1 / (spatial_var * (eigenvalues.array() + indep_var));
  return eigenvectors * diag.asDiagonal() * eigenvectors.transpose();
}

Matrix GaussianProcess::inverse_covariance(double spatial_var,
                                           double indep_var) const {
  LOG(debug) << "Computing inverse covariance matrix for spatial variance = "
             << spatial_var << " and iid noise = " << indep_var;
  Vector diag = 1 / (eigenvalues.array() + indep_var);
  Matrix m = Matrix::Zero(n, n);
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t i = 0; i < n; ++i)
    for (size_t j = 0; j < n; ++j)
      for (size_t k = 0; k < n; ++k)
        m(i, j) += eigenvectors(i, k) * diag(k) * eigenvectors(j, k);
  for (size_t i = 0; i < n; ++i)
    for (size_t j = 0; j < n; ++j)
      m(i, j) /= spatial_var;
  return m;
}

// negative one-half squared distances divided by squared length scale
Matrix GaussianProcess::rbf_kernel(const Matrix &x, double l) {
  assert(static_cast<size_t>(x.rows()) == n);
  const double l_square = l * l;
  Matrix k = Matrix::Zero(n, n);
#pragma omp parallel if (DO_PARALLEL)
  for (size_t i = 0; i < n; ++i)
    for (size_t j = i + 1; j < n; ++j) {
      Vector diff = x.row(i) - x.row(j);
      double d2 = diff.squaredNorm();
      k(i, j) = k(j, i) = exp(-0.5 * d2 / l_square);
    }
  k.diagonal() = Vector::Ones(n);
  return k;
}

GaussianProcess::VarGrad GaussianProcess::predict_means_and_vars(
    const Vector &y, const Vector &mean, double sv, double delta) const {
  LOG(verbose) << "Predicting means and variances for a vector of length "
               << y.size() << " sv = " << sv << " delta = " << delta;

  assert(sv > 0);
  assert(delta > 0);

  const double lower_limit = 1e-6;
  bool lower_limit_reached = false;
  if (lower_limit_reached = delta < lower_limit)
    delta = lower_limit;

  Vector Uy = eigenvectors.transpose() * y;
  Vector Umu = eigenvectors.transpose() * mean;

  Vector diff = Uy - Umu;
  Vector diag = (eigenvalues.array() + delta).inverse();
  double standard_score = 1 / sv * diff.transpose() * diag.asDiagonal() * diff;

  double grad_sv = 0.5 * (standard_score - n);

  double grad_delta_det = (1 / (eigenvalues.array() + delta)).sum();
  double grad_delta_score = -1 / sv * diff.transpose() * diag.asDiagonal()
                            * diag.asDiagonal() * diff;
  // NOTE: for Gamma distributions (which the priors are enforced to be),
  // we compute the gradient with respect to a transformed variable,
  // whence the * delta
  double grad_delta = -0.5 * delta * (grad_delta_det + grad_delta_score);

  Vector grad_pts
      = -sv * diff.transpose() * diag.asDiagonal() * eigenvectors.transpose();

  double score
      = -0.5
        * (n * (log(2 * M_PI) + log(sv))
           + (eigenvalues.array() + delta).log().sum() + standard_score);

  LOG(debug) << "GP score = " << n * (log(2 * M_PI) + log(sv)) << " + "
             << (eigenvalues.array() + delta).log().sum() << " + "
             << standard_score;
  if (lower_limit_reached and grad_delta < 0) {
    LOG(warning) << "WARNING!";
    grad_delta = 0;
  }
  VarGrad grad = {grad_sv, grad_delta, grad_pts, score};
  return grad;
};

vector<GaussianProcess::VarGrad> GaussianProcess::predict_means_and_vars(
    const Matrix &y, const Matrix &mean, const Vector &sv,
    const Vector &delta) const {
  vector<VarGrad> res;
  for (int i = 0; i < y.cols(); ++i)
    res.push_back(
        predict_means_and_vars(y.col(i), mean.col(i), sv(i), delta(i)));
  return res;
}

Vector GaussianProcess::sample(const Vector &mean, double sv,
                               double delta) const {
  assert(static_cast<size_t>(mean.size()) == n);
  Vector z(n);
  for (size_t i = 0; i < n; ++i)
    z(i) = normal_distribution<double>(0, 1)(EntropySource::rng);
  Matrix A = covariance_sqroot(sv, delta);
  return mean + A * z;
}

Matrix GaussianProcess::sample(const Matrix &mean, const Vector &sv,
                               const Vector &delta) const {
  assert(mean.cols() == sv.size());
  assert(mean.cols() == delta.size());
  Matrix x = mean;
  for (int i = 0; i < mean.cols(); ++i)
    x.col(i) = sample(mean.col(i), sv(i), delta(i));
  return x;
}
}  // namespace GP
