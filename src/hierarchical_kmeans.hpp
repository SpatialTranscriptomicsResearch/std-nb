#ifndef HIERARCHICAL_KMEANS_HPP
#define HIERARCHICAL_KMEANS_HPP

#include "types.hpp"

namespace PoissonFactorization {

using Profile = Vector;
using Profiles = std::vector<Profile>;

struct Hierarchy {
  Profile profile;
  std::vector<Hierarchy> children;
  Hierarchy(const Profile &p);
};

struct KMeansResults {
  Profiles profiles;
  std::vector<size_t> clusters;
};

KMeansResults kmeans(const Matrix &m, size_t k);

template <typename I>
Hierarchy hierarchical_kmeans(const Matrix &m, const Profile &profile, I begin,
                              I end) {
  size_t G = m.n_rows;
  size_t S = m.n_cols;
  Hierarchy h(profile);

  if (begin != end and S > 0) {
    size_t k = *begin;

    Matrix rem(m);
    for (size_t s = 0; s < S; ++s) {
      double z = 0;
      for (size_t g = 0; g < G; ++g) {
        rem(g, s) -= profile[g];
        z += rem(g, s) = std::max<Float>(0, m(g, s));
      }
      for (size_t g = 0; g < G; ++g)
        rem(g, s) /= z;
    }

    auto res = kmeans(rem, k);

    begin++;
    for (size_t r = 0; r < res.profiles.size(); ++r) {
      Profiles sub_r;
      for (size_t s = 0; s < S; ++s)
        if (res.clusters[s] == r)
          sub_r.push_back(rem.col(s));
      Matrix sub_m(G, sub_r.size());
      for (size_t s = 0; s < sub_r.size(); ++s)
        for (size_t g = 0; g < G; ++g)
          sub_m(g, s) = sub_r[s](g);
      if (sub_m.n_cols > 0)
        h.children.push_back(
            hierarchical_kmeans(sub_m, res.profiles[r], begin, end));
    }
  }
  return h;
}
}
#endif
