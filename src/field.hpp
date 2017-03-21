#ifndef FIELD_HPP
#define FIELD_HPP

#include <unistd.h>
#include <vector>
#include "parallel.hpp"
#include "types.hpp"

// using Point = std::vector<double>;
using Point = PoissonFactorization::Vector;

struct Field {
  size_t N;
  std::vector<Point> points;
  std::vector<std::vector<size_t>> adj;
  std::vector<std::vector<double>> alpha;
  std::vector<double> A;

  Field(const std::vector<Point> &pts,
        const std::vector<std::vector<size_t>> &adj_,
        const std::vector<std::vector<double>> &voronoi_weights)
      : N(pts.size()), points(pts), adj(adj_), alpha(), A(N, 0) {
    std::cerr << "Constructing field. N=" << N << std::endl;
    for (size_t i = 0; i < N; ++i) {
      std::vector<double> a;
      for (size_t k = 0; k < adj[i].size(); ++k) {
        size_t j = adj[i][k];
        // for (auto j : adj[i]) {
        double d = arma::norm(pts[i] - pts[j]);
        double current_a = voronoi_weights[i][k] / d;
        a.push_back(current_a);
        A[i] += voronoi_weights[i][k] * d;
        std::cerr << "Constructing field. i=" << i << " j=" << j << " k=" << k
                  << " v=" << voronoi_weights[i][k] << " d=" << d
                  << " cur_a=" << current_a << std::endl;
      }
      A[i] *= 0.25;
      alpha.push_back(a);
      std::cerr << "Constructing field. i=" << i << " A=" << A[i] << std::endl;
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
          std::cerr << "ohoh!\t"
                    << "i=" << i << "\t"
                    << "A=" << A[i] << "\t"
                    << "z=" << z[i] << std::endl;
          for (size_t k = 0; k < adj[i].size(); ++k) {
            size_t j = adj[i][k];
            std::cerr << i << "\t" << j << "\t" << alpha[i][k] << "\t" << fnc[j]
                      << std::endl;
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
          std::cerr << "ohoh!\t"
                    << "i=" << i << "\t"
                    << "A=" << A[i] << "\t"
                    << "z=" << z[i] << std::endl;
          for (size_t k = 0; k < adj[i].size(); ++k) {
            size_t j = adj[i][k];
            std::cerr << i << "\t" << j << "\t" << alpha[i][k] << "\t" << fnc[j]
                      << std::endl;
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
        std::cerr << "uhuh!\ti=" << i << "\tpart_a=" << part_a
                  << "\tpart_b=" << part_b << "\tA=" << A[i]
                  << "\tgrad=" << grad[i] << std::endl;
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
          std::cerr << "ohoh!\ti=" << i << "\ty=" << y << "\tb=" << b
                    << "\tA=" << A[i] << "\tz=" << z[i] << std::endl;
          for (size_t k = 0; k < adj[i].size(); ++k) {
            size_t j = adj[i][k];
            std::cerr << i << "\t" << j << "\t" << alpha[i][k] << "\t" << fnc[j]
                      << std::endl;
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
        std::cerr << "uhuh!\ti=" << i << "\tpart_a=" << part_a
                  << "\tpart_b=" << part_b << "\tA=" << A[i]
                  << "\tgrad=" << grad[i] << std::endl;
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
        std::cerr << "uhuh!\ti=" << i << "\tpart_a=" << part_a
                  << "\tpart_b=" << part_b << "\tA=" << A[i]
                  << "\tgrad=" << grad[i] << std::endl;
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
