#include "field.hpp"
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
