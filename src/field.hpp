#ifndef FIELD_HPP
#define FIELD_HPP

#include <unistd.h>
#include <vector>
#include "log.hpp"
#include "parallel.hpp"
#include "types.hpp"

// using Point = std::vector<double>;
using Point = PoissonFactorization::Vector;

void build_voronoi_qhull(const std::vector<Point> &points,
                         std::vector<std::vector<size_t>> &adj,
                         std::vector<std::vector<double>> &voronoi_weights);
struct Field {
  size_t dim;
  size_t N;
  std::vector<Point> points;
  std::vector<std::vector<size_t>> adj;
  std::vector<std::vector<double>> alpha;
  std::vector<double> A;

  Field(size_t dim_, const std::vector<Point> &pts)
      : dim(dim_), N(pts.size()), points(pts), A(N, 0) {
    std::vector<std::vector<double>> voronoi_weights;
    build_voronoi_qhull(points, adj, voronoi_weights);

    const bool verbose = false;
    if (verbose) {
      std::cerr << "N=" << N << std::endl;
      std::cerr << "adj.size()=" << adj.size() << std::endl;
      for (size_t i = 0; i < N; ++i) {
        std::cerr << "a\t" << i;
        for (size_t j = 0; j < adj[i].size(); ++j)
          std::cerr << "\t" << adj[i][j];
        std::cerr << std::endl;
      }
      for (size_t i = 0; i < N; ++i) {
        std::cerr << "v\t" << i;
        for (size_t j = 0; j < voronoi_weights[i].size(); ++j)
          std::cerr << "\t" << voronoi_weights[i][j];
        std::cerr << std::endl;
      }
    }

    LOG(verbose) << "Constructing field. N=" << N;
    for (size_t i = 0; i < N; ++i) {
      std::vector<double> a;
      for (size_t k = 0; k < adj[i].size(); ++k) {
        size_t j = adj[i][k];
        double d = arma::norm(pts[i] - pts[j]);
        double current_a = voronoi_weights[i][k] / d;
        a.push_back(current_a);
        A[i] += voronoi_weights[i][k] * d;
        LOG(debug) << "Constructing field. i=" << i << " j=" << j << " k=" << k
                   << " v=" << voronoi_weights[i][k] << " d=" << d
                   << " cur_a=" << current_a;
      }
      A[i] *= 0.25;
      alpha.push_back(a);
      LOG(debug) << "Constructing field. i=" << i << " A=" << A[i];
    }
  };

  template <typename V>
  V dirichlet_energy(const V &fnc) const {
    V z(N);
#pragma omp parallel for if (DO_PARALLEL)
    for (size_t i = 0; i < N; ++i) {
      z[i] = 0;
      if (adj[i].size() > 0) {
        for (size_t k = 0; k < adj[i].size(); ++k) {
          size_t j = adj[i][k];
          double diff = fnc[j] - fnc[i];
          /*
          double prod = alpha[i][k] * diff * diff;
          z[i] += prod / A[i];
          */
          z[i] += diff * diff;
        }
        if (std::isnan(z[i])) {
          LOG(fatal) << "ohoh!\t"
                     << "i=" << i << "\t"
                     << "A=" << A[i] << "\t"
                     << "z=" << z[i];
          for (size_t k = 0; k < adj[i].size(); ++k) {
            size_t j = adj[i][k];
            LOG(fatal) << i << "\t" << j << "\t" << alpha[i][k] << "\t"
                       << fnc[j];
          }
        }
      }
    }
    return z;
  };

  template <typename V>
  V grad_dirichlet_energy(const V &fnc) const {
    V z(N);
#pragma omp parallel for if (DO_PARALLEL)
    for (size_t i = 0; i < N; ++i) {
      z[i] = 0;
      if (adj[i].size() > 0) {
        for (size_t k = 0; k < adj[i].size(); ++k) {
          size_t j = adj[i][k];

          double diff = fnc[j] - fnc[i];
          /* TODO use this one!
          double prod = 2 * alpha[i][k] * diff;
          z[i] += - prod * (1 / A[j] + 1 / A[i]);
          */

          z[i] += -2 * diff;
        }
        if (std::isnan(z[i])) {
          LOG(fatal) << "ohoh!\t"
                     << "i=" << i << "\t"
                     << "A=" << A[i] << "\t"
                     << "z=" << z[i];
          for (size_t k = 0; k < adj[i].size(); ++k) {
            size_t j = adj[i][k];
            LOG(fatal) << i << "\t" << j << "\t" << alpha[i][k] << "\t"
                       << fnc[j];
          }
        }
      }
    }
    return z;
  };

  template <typename V>
  V grad_dirichlet_energy_bla(const V &fnc) const {
    V dir = dirichlet_energy(fnc);
    V grad(N);
#pragma omp parallel for if (DO_PARALLEL)
    for (size_t i = 0; i < N; ++i) {
      auto part_a = 0;
      double b = 0;
      for (size_t k = 0; k < adj[i].size(); ++k) {
        size_t j = adj[i][k];
        part_a += alpha[i][k] * fnc[j];
        b += alpha[i][k];
      }
      auto part_b = b * fnc[i];
      double diff = part_a - part_b;
      grad[i] = 2 / A[i] * diff;
      if (std::isnan(grad[i]))
        LOG(fatal) << "uhuh!\ti=" << i << "\tpart_a=" << part_a
                   << "\tpart_b=" << part_b << "\tA=" << A[i]
                   << "\tgrad=" << grad[i];
    }
    return grad;
  };

  template <typename V>
  double sum_dirichlet_energy(const V &fnc) const {
    V dir = dirichlet_energy(fnc);
    double s = 0;
    for (size_t i = 0; i < N; ++i)
      s += dir[i];
    return s;
  }

  template <typename V>
  V laplace_operator(const V &fnc) const {
    V z(N);
#pragma omp parallel for if (DO_PARALLEL)
    for (size_t i = 0; i < N; ++i) {
      z[i] = 0;
      if (adj[i].size() > 0) {
        double y = 0;
        double b = 0;
        for (size_t k = 0; k < adj[i].size(); ++k) {
          size_t j = adj[i][k];
          y += alpha[i][k] * fnc[j];
          b += alpha[i][k];
        }
        z[i] = 1 / A[i] * (y - b * fnc[i]);
        if (std::isnan(y) or std::isnan(b) or std::isnan(A[i])
            or std::isnan(z[i])) {
          LOG(fatal) << "ohoh!\ti=" << i << "\ty=" << y << "\tb=" << b
                     << "\tA=" << A[i] << "\tz=" << z[i];
          for (size_t k = 0; k < adj[i].size(); ++k) {
            size_t j = adj[i][k];
            LOG(fatal) << i << "\t" << j << "\t" << alpha[i][k] << "\t"
                       << fnc[j];
          }
        }
      }
    }
    return z;
  };

  template <typename V>
  V sq_laplace_operator(const V &fnc) const {
    V sq_lap = laplace_operator(fnc);
    for (size_t i = 0; i < N; ++i)
      sq_lap[i] *= sq_lap[i];
    return sq_lap;
  };

  template <typename V>
  double sum_sq_laplace_operator(const V &fnc) const {
    V sq_lap = sq_laplace_operator(fnc);
    double s = 0;
    for (size_t i = 0; i < N; ++i)
      s += sq_lap[i];
    return s;
  };

  /** Compute the gradient of the squared discrete laplace operator */
  template <typename V>
  V grad_laplace_operator(const V &fnc) const {
    V grad(N);
#pragma omp parallel for if (DO_PARALLEL)
    for (size_t i = 0; i < N; ++i) {
      auto part_a = 0;
      double b = 0;
      for (size_t k = 0; k < adj[i].size(); ++k) {
        size_t j = adj[i][k];
        part_a += alpha[i][k] / A[j];
        b += alpha[i][k];
      }
      auto part_b = b / A[i];
      grad[i] = 2 * (part_a - part_b);
      if (std::isnan(grad[i]))
        LOG(fatal) << "uhuh!\ti=" << i << "\tpart_a=" << part_a
                   << "\tpart_b=" << part_b << "\tA=" << A[i]
                   << "\tgrad=" << grad[i];
    }
    return grad;
  };

  /** Compute the gradient of the squared discrete laplace operator */
  template <typename V>
  V grad_sq_laplace_operator(const V &fnc) const {
    V lap = laplace_operator(fnc);
    V grad(N);
#pragma omp parallel for if (DO_PARALLEL)
    for (size_t i = 0; i < N; ++i) {
      auto part_a = 0;
      double b = 0;
      for (size_t k = 0; k < adj[i].size(); ++k) {
        size_t j = adj[i][k];
        part_a += lap[j] * alpha[i][k] / A[j];
        b += alpha[i][k];
      }
      auto part_b = lap[i] * b / A[i];
      grad[i] = 2 * (part_a - part_b);
      if (std::isnan(grad[i]))
        LOG(fatal) << "uhuh!\ti=" << i << "\tpart_a=" << part_a
                   << "\tpart_b=" << part_b << "\tA=" << A[i]
                   << "\tgrad=" << grad[i];
    }
    return grad;
  };
};

std::ostream &operator<<(std::ostream &os, const Field &field) {
  os << "N = " << field.N << std::endl;
  os << "points = ";
  for (size_t i = 0; i < field.N; ++i) {
    os << i << ":\tA=" << field.A[i];
    for (size_t j = 0; j < field.adj[i].size(); ++j)
      os << "\t" << field.adj[i][j] << "/" << field.alpha[i][j];
    os << std::endl;
  }
  for (size_t i = 0; i < field.N; ++i)
    for (size_t j = 0; j < field.adj[i].size(); ++j)
      os << "edge" << field.points[i].t() << "edge"
         << field.points[field.adj[i][j]].t();
  return os;
};

#endif
