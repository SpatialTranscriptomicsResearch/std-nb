#include "field.hpp"
#include <LBFGS.h>
#include <iostream>
#include "sampling.hpp"

using namespace std;

const double large_val = 100;
const bool verbose = false;

/** Shoelace formula, assumes that pts are given in sequences of the order on
 * the polytope */
template <typename P>
double polytope_area(const std::vector<P> &pts) {
  double area = 0;
  for (size_t i = 0; i < pts.size() - 1; i++)
    area += pts[i][0] * pts[i + 1][1] - pts[i + 1][0] * pts[i][1];
  area += pts[pts.size() - 1][0] * pts[0][1]
          - pts[0][0] * pts[pts.size() - 1][1];
  return 0.5 * fabs(area);
}

void build_voronoi_qhull(const std::vector<Point> &points,
                         std::vector<std::vector<size_t>> &adj,
                         std::vector<std::vector<double>> &voronoi_weights) {
  const std::string coord_path = "/tmp/coord.txt";
  const std::string voronoi_path = "/tmp/voronoi.txt";
  const size_t num_points = points.size();
  if (num_points == 0)
    return;
  std::ofstream ofs(coord_path);
  const size_t dim = points[0].size();

  ofs << dim << " rbox " << num_points << " D" << dim << std::endl
      << num_points << std::endl;
  for (auto &pt : points) {
    for (size_t i = 0; i < pt.size(); ++i)
      ofs << (i == 0 ? "" : " ") << pt[i];
    ofs << std::endl;
  }
  int res = system((std::string() + "cat " + coord_path
                    + " | qvoronoi s o Fv TO " + voronoi_path)
                       .c_str());
  std::cerr << "return val = " << res << std::endl;

  ifstream ifs(voronoi_path);
  double bla;
  ifs >> bla;
  size_t num_voronoi_vertices, num_voronoi_cells, num_unknown;
  ifs >> num_voronoi_vertices;
  ifs >> num_voronoi_cells;
  ifs >> num_unknown;
  std::cerr << "num_voronoi_vertices = " << num_voronoi_vertices << std::endl;
  std::cerr << "num_voronoi_cells = " << num_voronoi_cells << std::endl;
  std::cerr << "num_unknown = " << num_unknown << std::endl;
  std::vector<std::vector<double>> voronoi_vertices(
      num_voronoi_vertices, std::vector<double>(dim, 0));
  for (size_t i = 0; i < num_voronoi_vertices; ++i)
    for (size_t j = 0; j < dim; ++j)
      ifs >> voronoi_vertices[i][j];

  std::vector<std::vector<size_t>> voronoi_cells;
  for (size_t i = 0; i < num_voronoi_cells; ++i) {
    size_t num_vertices;
    ifs >> num_vertices;
    voronoi_cells.push_back(std::vector<size_t>(num_vertices, 0));
    for (size_t j = 0; j < num_vertices; ++j)
      ifs >> voronoi_cells[i][j];
  }
  size_t num_ridges;
  ifs >> num_ridges;
  std::cerr << "num_ridges = " << num_ridges << std::endl;
  std::vector<std::pair<std::pair<size_t, size_t>, std::vector<size_t>>> ridges;

  adj = std::vector<std::vector<size_t>>(num_points);
  voronoi_weights = std::vector<std::vector<double>>(num_points);

  for (size_t i = 0; i < num_ridges; ++i) {
    size_t num_elem, first_point, second_point, x;
    ifs >> num_elem;
    ifs >> first_point;
    ifs >> second_point;
    std::vector<size_t> vertices;
    for (size_t j = 3; j <= num_elem; ++j) {
      ifs >> x;
      vertices.push_back(x);
    }
    ridges.push_back({{first_point, second_point}, vertices});
    adj[first_point].push_back(second_point);
    adj[second_point].push_back(first_point);

    double m = 0;
    if (dim == 2) {
      assert(vertices.size() == 2);
      // TODO when vertices[0] == 0 then this is the infinite vertex and needs
      // special treatment
      auto a = voronoi_vertices[vertices[0]];
      auto b = voronoi_vertices[vertices[1]];
      double dx = a[0] - b[0];
      double dy = a[1] - b[1];
      double d = sqrt(dx * dx + dy * dy);
      m = d;
    } else if (dim == 3) {
      // TODO when vertex == 0 then this is the infinite vertex and needs
      // special treatment
      std::vector<std::vector<double>> polytope_vertices;
      for (auto &vertex : vertices)
        polytope_vertices.push_back(voronoi_vertices[vertex]);
      m = polytope_area(polytope_vertices);
    } else {
      assert(false);
    }
    // m = 1;
    voronoi_weights[first_point].push_back(m);
    voronoi_weights[second_point].push_back(m);
  }
}

int main(int argc, char **argv) {
  EntropySource::seed();
  size_t n = 2000;
  size_t num_fixed = 50;
  size_t dim = 3;
  const size_t oversampling_factor = 400;
  const double grid_scale = 10;
  const size_t n_iter = 1000000;
  const size_t report_interval = 100;

  vector<Point> pts;
  PoissonFactorization::Vector fnc;

  if (argc == 2) {
    vector<double> val;
    ifstream ifs(argv[1]);
    ifs >> dim;
    Point pt(dim);
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
      Point p(dim);
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
    fnc = PoissonFactorization::Vector(n);
    for (size_t i = 0; i < num_fixed; ++i)
      fnc[i] = val[i];
    for (size_t i = num_fixed; i < n; ++i)
      fnc[i] = 1000 * RandomDistribution::Uniform(EntropySource::rng);
  } else {
    for (size_t i = 0; i < n; ++i) {
      Point pt(dim);
      pt[0] = grid_scale * RandomDistribution::Uniform(EntropySource::rng);
      pt[1] = grid_scale * RandomDistribution::Uniform(EntropySource::rng);
      if (dim == 3)
        pt[2] = grid_scale * RandomDistribution::Uniform(EntropySource::rng);
      pts.push_back(pt);
    }

    fnc = PoissonFactorization::Vector(n);
    for (size_t i = 0; i < n; ++i)
      fnc[i] = 1000 * RandomDistribution::Uniform(EntropySource::rng);
  }

  Field field(dim, pts);

  if (verbose)
    cerr << "Field:\n" << field << endl;

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
    // double score = field.sum_sq_laplace_operator(vec);
    double score = field.sum_dirichlet_energy(vec);

    // g = field.grad_sq_laplace_operator(vec);
    g = field.grad_dirichlet_energy(vec);

    for (size_t i = 0; i < num_fixed; ++i)
      g[i] = 0;

    if (call_cnt++ % report_interval == 0) {
      cerr << "score = " << score << endl;
      cerr << "sum_sq_lap = " << field.sum_sq_laplace_operator(vec) << endl;
      cerr << "dir = " << field.sum_dirichlet_energy(vec) << endl;
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
