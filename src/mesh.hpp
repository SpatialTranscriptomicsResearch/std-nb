#ifndef MESH_HPP
#define MESH_HPP

#include <unistd.h>
#include <vector>
#include "log.hpp"
#include "parallel.hpp"
#include "types.hpp"

// using Point = std::vector<double>;
using Point = PoissonFactorization::Vector;

struct Mesh {
  size_t dim;
  size_t N;
  std::vector<Point> points;
  std::vector<std::vector<size_t>> adj;
  std::vector<std::vector<double>> alpha;
  std::vector<double> A;

  Mesh(size_t dim_ = 0, const std::vector<Point> &pts = {});

  void store(const std::string &path,
             const PoissonFactorization::Matrix &m) const;
  void restore(const std::string &path, PoissonFactorization::Matrix &m);

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
    assert(static_cast<size_t>(fnc.size()) == N);
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
        for (auto &j : adj[i])
          z[i] += fnc[j] - fnc[i];
        if (std::isnan(z[i])) {
          LOG(fatal) << "ohoh!\ti=" << i << "\tz=" << z[i];
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
  V laplace_operator_analytic(const V &fnc) const {
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
      grad[i] = 0;
      for (auto &j : adj[i])
        grad[i] += 2 * (lap[j] - lap[i]);
      if (std::isnan(grad[i]))
        LOG(fatal) << "uhuh!\ti=" << i << "\tgrad=" << grad[i];
    }
    return grad;
  };

  /** Compute the gradient of the squared discrete laplace operator */
  template <typename V>
  V grad_sq_laplace_operator_analytic(const V &fnc) const {
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

std::ostream &operator<<(std::ostream &os, const Mesh &field);

#endif
