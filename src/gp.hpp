#ifndef GP_HPP
#define GP_HPP
#include <vector>
#include "types.hpp"

namespace GP {
using Matrix = STD::Matrix;
using Vector = STD::Vector;
enum class MeanTreatment { zero, shared, independent };

struct GaussianProcess {
  GaussianProcess();
  GaussianProcess(const Matrix &x, double len_scale);
  Matrix covariance(double spatial_var, double indep_var) const;

  Matrix inverse_covariance_eigen(double spatial_var, double indep_var) const;
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

STD::Vector predict_means_and_vars(const std::vector<const GaussianProcess *> &gps,
                            const std::vector<Matrix> &ys,
                            const std::vector<double> &delta,
                            MeanTreatment mean_treatment,
                            std::vector<Matrix> &mus,
                            std::vector<Matrix> &vars);

double predict_means_and_vars(const std::vector<const GaussianProcess *> &gps,
                            const std::vector<Vector> &ys, double delta,
                            MeanTreatment mean_treatment,
                            std::vector<Vector> &mus,
                            std::vector<Vector> &vars);
}
#endif
