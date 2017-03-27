#include "mesh.hpp"
#include <iostream>
#include "sampling.hpp"

using namespace std;

const double large_val = 100;
const bool verbose = false;

/** Shoelace formula, assumes that pts are given in sequences of the order on
 * the polytope */
template <typename P>
double polytope_area(const vector<P> &pts) {
  double area = 0;
  for (size_t i = 0; i < pts.size() - 1; i++)
    area += pts[i][0] * pts[i + 1][1] - pts[i + 1][0] * pts[i][1];
  area += pts[pts.size() - 1][0] * pts[0][1]
          - pts[0][0] * pts[pts.size() - 1][1];
  return 0.5 * fabs(area);
}

void build_voronoi_qhull(const vector<Point> &points,
                         vector<vector<size_t>> &adj,
                         vector<vector<double>> &voronoi_weights) {
  const string coord_path = "/tmp/coord.txt";
  const string voronoi_path = "/tmp/voronoi.txt";
  const size_t num_points = points.size();
  if (num_points == 0)
    return;
  ofstream ofs(coord_path);
  const size_t dim = points[0].size();

  ofs << dim << " rbox " << num_points << " D" << dim << endl
      << num_points << endl;
  for (auto &pt : points) {
    for (size_t i = 0; i < pt.size(); ++i)
      ofs << (i == 0 ? "" : " ") << pt[i];
    ofs << endl;
  }
  int res = system(
      (string() + "cat " + coord_path + " | qvoronoi s o Fv TO " + voronoi_path)
          .c_str());
  cerr << "return val = " << res << endl;

  ifstream ifs(voronoi_path);
  double bla;
  ifs >> bla;
  size_t num_voronoi_vertices, num_voronoi_cells, num_unknown;
  ifs >> num_voronoi_vertices;
  ifs >> num_voronoi_cells;
  ifs >> num_unknown;
  cerr << "num_voronoi_vertices = " << num_voronoi_vertices << endl;
  cerr << "num_voronoi_cells = " << num_voronoi_cells << endl;
  cerr << "num_unknown = " << num_unknown << endl;
  vector<vector<double>> voronoi_vertices(num_voronoi_vertices,
                                          vector<double>(dim, 0));
  for (size_t i = 0; i < num_voronoi_vertices; ++i)
    for (size_t j = 0; j < dim; ++j)
      ifs >> voronoi_vertices[i][j];

  vector<vector<size_t>> voronoi_cells;
  for (size_t i = 0; i < num_voronoi_cells; ++i) {
    size_t num_vertices;
    ifs >> num_vertices;
    voronoi_cells.push_back(vector<size_t>(num_vertices, 0));
    for (size_t j = 0; j < num_vertices; ++j)
      ifs >> voronoi_cells[i][j];
  }
  size_t num_ridges;
  ifs >> num_ridges;
  cerr << "num_ridges = " << num_ridges << endl;
  vector<pair<pair<size_t, size_t>, vector<size_t>>> ridges;

  adj = vector<vector<size_t>>(num_points);
  voronoi_weights = vector<vector<double>>(num_points);

  for (size_t i = 0; i < num_ridges; ++i) {
    size_t num_elem, first_point, second_point, x;
    ifs >> num_elem;
    ifs >> first_point;
    ifs >> second_point;
    vector<size_t> vertices;
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
      vector<vector<double>> polytope_vertices;
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

Mesh::Mesh(size_t dim_, const vector<Point> &pts)
    : dim(dim_), N(pts.size()), points(pts), A(N, 0) {
  vector<vector<double>> voronoi_weights;
  build_voronoi_qhull(points, adj, voronoi_weights);

  LOG(verbose) << "Constructing mesh. N=" << N << " dim=" << dim;
  if (verbose) {
    cerr << "adj.size()=" << adj.size() << endl;
    for (size_t i = 0; i < N; ++i) {
      cerr << "a\t" << i;
      for (size_t j = 0; j < adj[i].size(); ++j)
        cerr << "\t" << adj[i][j];
      cerr << endl;
    }
    for (size_t i = 0; i < N; ++i) {
      cerr << "v\t" << i;
      for (size_t j = 0; j < voronoi_weights[i].size(); ++j)
        cerr << "\t" << voronoi_weights[i][j];
      cerr << endl;
    }
  }

  for (size_t i = 0; i < N; ++i) {
    vector<double> a;
    for (size_t k = 0; k < adj[i].size(); ++k) {
      size_t j = adj[i][k];
      double d = arma::norm(pts[i] - pts[j]);
      double current_a = voronoi_weights[i][k] / d;
      a.push_back(current_a);
      A[i] += voronoi_weights[i][k] * d;
      LOG(debug) << "Constructing mesh. i=" << i << " j=" << j << " k=" << k
                 << " v=" << voronoi_weights[i][k] << " d=" << d
                 << " cur_a=" << current_a;
    }
    A[i] *= 0.25;
    alpha.push_back(a);
    LOG(debug) << "Constructing mesh. i=" << i << " A=" << A[i];
  }
}

void Mesh::store(const string &path,
                  const PoissonFactorization::Matrix &m) const {}

void Mesh::restore(const string &path, PoissonFactorization::Matrix &m) {}

ostream &operator<<(ostream &os, const Mesh &mesh) {
  os << "N = " << mesh.N << endl;
  os << "points = ";
  for (size_t i = 0; i < mesh.N; ++i) {
    os << i << ":\tA=" << mesh.A[i];
    for (size_t j = 0; j < mesh.adj[i].size(); ++j)
      os << "\t" << mesh.adj[i][j] << "/" << mesh.alpha[i][j];
    os << endl;
  }
  for (size_t i = 0; i < mesh.N; ++i)
    for (size_t j = 0; j < mesh.adj[i].size(); ++j)
      os << "edge" << mesh.points[i].t() << "edge"
         << mesh.points[mesh.adj[i][j]].t();
  return os;
}
