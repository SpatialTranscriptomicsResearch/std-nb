#ifndef EXPERIMENT_DGE_HPP
#define EXPERIMENT_DGE_HPP

#include "Experiment.hpp"

template <typename Type>
template <typename Fnc>
Matrix Experiment<Type>::local_dge(Fnc fnc,
                                   const features_t &global_features) const {
  auto spot_weights = marginalize_spots();
  Matrix m(G, T);
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t g = 0; g < G; ++g)
    for (size_t t = 0; t < T; ++t)
      m(g, t) = local_dge_sub(fnc, global_features, g, t, spot_weights[t]);
  return m;
}

template <typename Type>
template <typename Fnc>
Float Experiment<Type>::local_dge_sub(Fnc fnc,
                                      const features_t &global_features,
                                      size_t g, size_t t, Float theta_,
                                      Float p) const {
  const Float eps = 1e-6;

  Float q = 1 - p;
  Float lp = log(p);
  Float lq = log(q);

  double mi = 0;
  size_t x = 0;
  double cumsum = 0;
  while (cumsum < 2 - 2 * eps) {
    Float l1 = log_negative_binomial(x, global_features.prior.r(g, t),
                                     baseline_phi(g) * phi(g, t) * theta_,
                                     global_features.prior.p(g, t));
    Float l2 = log_negative_binomial(x, global_features.prior.r(g, t),
                                     fnc(baseline_phi(g), phi(g, t)) * theta_,
                                     global_features.prior.p(g, t));

    cumsum += exp(l1);
    cumsum += exp(l2);

    l1 += lp;
    l2 += lq;

    Float lz = logSumExp(l1, l2);

    Float p1 = exp(l1);
    Float p2 = exp(l2);

    mi += p1 * (l1 - lz - lp) + p2 * (l2 - lz - lq);
    x++;
  }

  // LOG(debug) << "t = " << t << " g = " << g << " x = " << x;

  return mi / log(2.0);
}

template <typename Type>
Matrix Experiment<Type>::pairwise_dge(const features_t &global_features) const {
  size_t T_ = 0;
  for (size_t t1 = 0; t1 < T; ++t1)
    for (size_t t2 = t1 + 1; t2 < T; ++t2)
      T_++;
  Matrix m(G, T_);
  size_t t_ = 0;
  for (size_t t1 = 0; t1 < T; ++t1)
    for (size_t t2 = t1 + 1; t2 < T; ++t2) {
      auto v = pairwise_dge_sub(global_features, t1, t2);
      for (size_t g = 0; g < G; ++g)
        m(g, t_) = v(g);
      t_++;
    }
  return m;
}

template <typename Type>
Vector Experiment<Type>::pairwise_dge_sub(const features_t &global_features,
                                          size_t t1, size_t t2) const {
  LOG(verbose) << "Performing DGE for factor " << t1 << " and factor " << t2;
  Vector v(G);
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t g = 0; g < G; ++g)
    v(g) = pairwise_dge_sub(global_features, t1, t2, g);
  return v;
}

template <typename Type>
Float Experiment<Type>::pairwise_dge_sub(const features_t &global_features,
                                         size_t t1, size_t t2, size_t g,
                                         Float theta_, Float p) const {
  const Float eps = 1e-6;

  Float q = 1 - p;

  Float lp = log(p);
  Float lq = log(q);

  double mi = 0;
  size_t x = 0;
  double cumsum = 0;
  while (cumsum < 2 - 2 * eps) {
    Float l1 = log_negative_binomial(x, global_features.prior.r(g, t1),
                                     baseline_phi(g) * phi(g, t1) * theta_,
                                     global_features.prior.p(g, t1));
    Float l2 = log_negative_binomial(x, global_features.prior.r(g, t2),
                                     baseline_phi(g) * phi(g, t2) * theta_,
                                     global_features.prior.p(g, t2));

    cumsum += exp(l1);
    cumsum += exp(l2);

    l1 += lp;
    l2 += lq;

    Float lz = logSumExp(l1, l2);

    Float p1 = exp(l1);
    Float p2 = exp(l2);

    mi += p1 * (l1 - lz - lp) + p2 * (l2 - lz - lq);

    x++;
  }

  // LOG(debug) << "t1 = " << t1 << " t2 = " << t2 << " g = " << g << " x = " << x;

  return mi / log(2.0);
}

#endif
