#include "gp.hpp"
#include <eigen3/Eigen/Eigenvalues>
#include "log.hpp"
#include "parallel.hpp"
#include "types.hpp"

using namespace std;

namespace GP {
GaussianProcess::GaussianProcess() : length_scale(1) {}

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

Matrix GaussianProcess::inverse_covariance_eigen(double spatial_var,
                                                 double indep_var) const {
  LOG(debug) << "Computing inverse covariance matrix for spatial variance = "
             << spatial_var << " and iid noise = " << indep_var;
  Vector diag = 1 / (eigenvalues.array() + indep_var);
  return 1 / spatial_var * eigenvectors * diag.asDiagonal()
         * eigenvectors.transpose();
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
      double d = diff.dot(diff);
      k(i, j) = k(j, i) = exp(-1 / 2.0 * d / l_square);
    }
  k.diagonal().array() = 1;
  return k;
}

double GaussianProcess::calc_mean(Vector y, double delta) const {
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
                                              double delta) const {
  Vector u1 = eigenvectors.transpose() * Vector::Ones(n) * mean;
  Vector uy = eigenvectors.transpose() * y;
  double sigma = 0;
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t i = 0; i < n; ++i) {
    double diff = uy(i) - u1(i);
    sigma += diff * diff / (eigenvalues(i) + delta);
  }
  return sigma / n;
}

void GaussianProcess::predict_means_and_vars(const Vector &y, double delta,
                                             Vector &mu, Vector &var) const {
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

Vector predict_means_and_vars(const vector<const GaussianProcess *> &gps,
                              const vector<Matrix> &ys,
                              const vector<double> &delta,
                              MeanTreatment mean_treatment, vector<Matrix> &mus,
                              vector<Matrix> &vars) {
  assert(not ys.empty());
  Vector res(ys.front().cols());
  for (int i = 0; i < ys.front().cols(); ++i) {
    vector<Vector> vys, vmus, vvars;
    for (size_t j = 0; j < ys.size(); ++j) {
      vys.push_back(ys[j].col(i));
      vmus.push_back(mus[j].col(i));
      vvars.push_back(vars[j].col(i));
    }
    res(i) = predict_means_and_vars(gps, vys, delta[i], mean_treatment, vmus,
                                    vvars);
    for (size_t j = 0; j < ys.size(); ++j) {
      mus[j].col(i) = vmus[j];
      vars[j].col(i) = vvars[j];
    }
  }
  return res;
}

double predict_means_and_vars(const vector<const GaussianProcess *> &gps,
                              const vector<Vector> &ys, double delta,
                              MeanTreatment mean_treatment, vector<Vector> &mus,
                              vector<Vector> &vars) {
  const size_t num_gp = gps.size();
  size_t n = 0;
  for (auto &gp : gps)
    n += gp->n;

  assert(ys.size() == num_gp);
  assert(mus.size() == num_gp);
  assert(vars.size() == num_gp);

  Vector means = Vector::Zero(num_gp);
  switch (mean_treatment) {
    case MeanTreatment::zero:
      break;
    case MeanTreatment::shared:
      // TODO sum numerators and denominators
      break;
    case MeanTreatment::independent:
      for (size_t idx = 0; idx < num_gp; ++idx)
        means(idx) = gps[idx]->calc_mean(ys[idx], delta);
      break;
  }
  double sv = 0;
  for (size_t idx = 0; idx < num_gp; ++idx)
    sv += gps[idx]->calc_spatial_variance(ys[idx], means[idx], delta)
          * gps[idx]->n;
  sv /= n;
  LOG(verbose) << "GP: means = " << means.transpose() << " sv = " << sv
               << " delta = " << delta;

  if (sv > 0) {
    for (size_t idx = 0; idx < num_gp; ++idx) {
      LOG(debug) << "Doing coordinate system " << idx;
      Matrix inverse = gps[idx]->inverse_covariance(sv, delta);

      Vector y_minus_mean = ys[idx].array() - means[idx];
#pragma omp parallel for if (DO_PARALLEL)
      for (size_t i = 0; i < gps[idx]->n; ++i) {
        mus[idx](i) = 0;
        for (size_t j = 0; j < gps[idx]->n; ++j)
          if (i != j)
            mus[idx](i) += y_minus_mean(j) * inverse(i, j);
        mus[idx](i) *= -1 / inverse(i, i);
        vars[idx](i) = 1 / inverse(i, i);

        mus[idx](i) += means[idx];
      }
    }
  }
  return sv;
}

}
