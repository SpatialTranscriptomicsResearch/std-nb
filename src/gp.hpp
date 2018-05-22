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
    Vector points;
    double score;
  };
  VarGrad predict_means_and_vars(const Vector &y, const Vector &mean, double sv,
                                 double delta) const;
  std::vector<VarGrad> predict_means_and_vars(const Matrix &ys,
                                              const Matrix &means,
                                              const Vector &sv,
                                              const Vector &delta) const;

  Vector sample(const Vector &mean, double sv, double delta) const;
  Matrix sample(const Matrix &mean, const Vector &sv,
                const Vector &delta) const;
};  // namespace GP

}  // namespace GP
#endif
