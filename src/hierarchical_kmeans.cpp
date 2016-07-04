#include "hierarchical_kmeans.hpp"
#include "entropy.hpp"
#include "parallel.hpp"

namespace PoissonFactorization {

using namespace std;

Hierarchy::Hierarchy(const Profile &p) : profile(p) {}

double dist(const Vector &a, const Vector &b) {
  double d = 0;
  for(size_t i = 0; i < a.size(); ++i)
    d += fabs(a[i] - b[i]);
  return d;
}

KMeansResults kmeans(const Matrix &m, size_t K) {
  size_t G = m.n_rows;
  size_t S = m.n_cols;

  KMeansResults res;
  res.clusters = vector<size_t>(S);

  for (size_t k = 0; k < K; ++k)
    res.profiles.push_back(
        m.col(uniform_int_distribution<size_t>(0, S - 1)(EntropySource::rng)));

  bool ok = true;
  while(ok) {
    ok = false;

    // assign data to closest cluster
#pragma omp parallel for if (DO_PARALLEL)
    for(size_t s = 0; s < S; ++s) {
      size_t arg_min = 0;
      double min = std::numeric_limits<double>::infinity();
      auto col = m.col(s);
      for(size_t k = 0; k < K; ++k) {
        auto d = dist(col, res.profiles[k]);
        if(d < min) {
          min = d;
          arg_min = k;
        }
      }
      if(arg_min != res.clusters[s]) {
        ok = true;
        res.clusters[s] = arg_min;
      }
    }

    // set cluster centers to mean of assigned data
    for(size_t k = 0; k < K; ++k)
      res.profiles[k] = Vector(G, arma::fill::zeros);
#pragma omp parallel for if (DO_PARALLEL)
    for(size_t g = 0; g < G; ++g)
      for(size_t s = 0; s < S; ++s)
        res.profiles[res.clusters[s]](g) += m(g, s);
#pragma omp parallel for if (DO_PARALLEL)
    for(size_t k = 0; k < K; ++k) {
      double z = 0;
      for(size_t g = 0; g < G; ++g)
        z += res.profiles[k](g);
      for(size_t g = 0; g < G; ++g)
        res.profiles[k](g) /= z;
    }
  }
  return res;
}

ostream &operator<<(ostream &os, const KMeansResults &results) {
    os << "clusters:";
    for(auto &cl: results.clusters)
      os << " " << cl;
  return os;
}
}
