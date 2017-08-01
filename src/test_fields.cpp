#include <LBFGS.h>
#include <iostream>
#include <fstream>
#include "Mesh.hpp"
#include "sampling.hpp"

using namespace std;

const bool verbose = false;

int main(int argc, char **argv) {
  EntropySource::seed();
  size_t n = 2000;
  size_t num_fixed = 50;
  size_t dim = 3;
  const size_t oversampling_factor = 400;
  const double grid_scale = 10;
  const size_t n_iter = 1000000;
  const size_t report_interval = 100;

  vector<Mesh::Point> pts;
  STD::Vector fnc;

  if (argc == 2) {
    vector<double> val;
    ifstream ifs(argv[1]);
    ifs >> dim;
    Mesh::Point pt(dim);
    double val_;
    while (ifs >> pt[0] >> pt[1]) {
      if (dim == 3)
        ifs >> pt[2];
      if (ifs >> val_) {
        pts.push_back(pt);
        val.push_back(val_);
      }
    }

    n = pts.size();
    num_fixed = pts.size();

    double max_x = -numeric_limits<double>::infinity();
    double min_x = numeric_limits<double>::infinity();
    double max_y = -numeric_limits<double>::infinity();
    double min_y = numeric_limits<double>::infinity();
    double max_z = -numeric_limits<double>::infinity();
    double min_z = numeric_limits<double>::infinity();

    for (auto &p : pts) {
      if (p[0] < min_x)
        min_x = p[0];
      if (p[1] < min_y)
        min_y = p[1];
      if (dim == 3 and p[2] < min_z)
        min_z = p[2];

      if (p[0] > max_x)
        max_x = p[0];
      if (p[1] > max_y)
        max_y = p[1];
      if (dim == 3 and p[2] > max_z)
        max_z = p[2];
    }

    for (size_t i = 0; i < oversampling_factor * n; ++i) {
      Mesh::Point p(dim);
      p[0]
          = min_x
            + (max_x - min_x) * RandomDistribution::Uniform(EntropySource::rng);
      p[1]
          = min_y
            + (max_y - min_y) * RandomDistribution::Uniform(EntropySource::rng);
      if (dim == 3)
        p[2] = min_z
               + (max_z - min_z)
                     * RandomDistribution::Uniform(EntropySource::rng);
      pts.push_back(p);
    }

    n = pts.size();
    fnc = STD::Vector(n);
    for (size_t i = 0; i < num_fixed; ++i)
      fnc[i] = val[i];
    for (size_t i = num_fixed; i < n; ++i)
      fnc[i] = 1000 * RandomDistribution::Uniform(EntropySource::rng);
  } else {
    for (size_t i = 0; i < n; ++i) {
      Mesh::Point pt(dim);
      pt[0] = grid_scale * RandomDistribution::Uniform(EntropySource::rng);
      pt[1] = grid_scale * RandomDistribution::Uniform(EntropySource::rng);
      if (dim == 3)
        pt[2] = grid_scale * RandomDistribution::Uniform(EntropySource::rng);
      pts.push_back(pt);
    }

    fnc = STD::Vector(n);
    for (size_t i = 0; i < n; ++i)
      fnc[i] = 1000 * RandomDistribution::Uniform(EntropySource::rng);
  }

  Mesh mesh(dim, pts);

  if (verbose)
    cerr << "Mesh:\n" << mesh << endl;

  /*
  std::cerr << "initial fnc: " << fnc.t() << std::endl;
  for (size_t i = 0; i < n; ++i) {
    cout << "before"
         << "\t" << i << "\t" << pts[i][0] << "\t" << pts[i][1] << "\t"
         << fnc[i] << endl;
  }
  */

  using namespace LBFGSpp;
  LBFGSParam<double> param;
  param.epsilon = 1e-9;
  param.max_iterations = n_iter;
  // Create solver and function object
  LBFGSSolver<double> solver(param);

  // using Vec = LBFGSSolver<double>::Vector;
  using Vec = Eigen::VectorXd;
  Vec v(n);
  Vec grad(n);
  for (size_t i = 0; i < n; ++i)
    v[i] = fnc[i];

  size_t call_cnt = 0;
  auto fun = [&](const Vec &vec, Vec &g) {
    // double score = mesh.sum_sq_laplace_operator(vec);
    double score = mesh.sum_dirichlet_energy(vec);

    // g = mesh.grad_sq_laplace_operator(vec);
    g = mesh.grad_dirichlet_energy(vec);

    for (size_t i = 0; i < num_fixed; ++i)
      g[i] = 0;

    if (call_cnt++ % report_interval == 0) {
      cerr << "score = " << score << endl;
      cerr << "sum_sq_lap = " << mesh.sum_sq_laplace_operator(vec) << endl;
      cerr << "dir = " << mesh.sum_dirichlet_energy(vec) << endl;
    }
    return score;
  };

  // Initial guess
  // VectorXd x = VectorXd::Zero(n);
  // x will be overwritten to be the best point found
  double fx;
  int niter = solver.minimize(fun, v, fx);
  std::cout << niter << " iterations" << std::endl;
  std::cout << "x = \n" << v.transpose() << std::endl;
  std::cout << "f(x) = " << fx << std::endl;

  for (size_t i = 0; i < n; ++i)
    fnc[i] = v[i];

  for (size_t i = 0; i < n; ++i) {
    cout << "final"
         << "\t" << i << "\t" << pts[i][0] << "\t" << pts[i][1] << "\t"
         << pts[i][2] << "\t" << fnc[i] << endl;
  }
  return EXIT_SUCCESS;
}
