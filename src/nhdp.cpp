#include <list>
#include "entropy.hpp"
#include "log.hpp"
#include "sampling.hpp"
#include "nhdp.hpp"

using namespace std;

namespace PoissonFactorization {

nHDP::nHDP(size_t g, size_t s, size_t t, const Parameters &params)
    : G(g),
      S(s),
      T(1),
      maxT(t),
      parameters(params),
      counts_gene_type(G, maxT, arma::fill::zeros),
      counts_spot_type(S, maxT, arma::fill::zeros),
      desc_counts_gene_type(G, maxT, arma::fill::zeros),
      desc_counts_spot_type(S, maxT, arma::fill::zeros),
      counts_type(maxT, arma::fill::zeros),
      desc_counts_type(maxT, arma::fill::zeros),
      parent_of(maxT),
      children_of(maxT) {}

void nHDP::add_hierarchy(size_t t, const Hierarchy &hierarchy,
                         double concentration) {
  counts_gene_type.col(t) = hierarchy.profile * concentration;
  for (auto &h : hierarchy.children) {
    size_t child = add_node(t);
    add_hierarchy(child, h, concentration);
  }
}

size_t nHDP::sample_type(size_t g, size_t s, bool independent_switches) const {
  LOG(verbose) << "Sample type for gene " << g << " in spot " << s;

  // the first T components of the vector represent probabilities for the currently active factors
  // the second T components of the vector represent probabilities for new possible factors
  vector<Float> p(2 * T, 0);

  list<size_t> types;

  types.push_front(0);
  p[0] = 1;

  LOG(debug) << "counts_spot_type(s,.) = " << counts_spot_type.row(s);
  LOG(debug) << "desc_counts_spot_type(s,.) = " << desc_counts_spot_type.row(s);

  while (not types.empty()) {
    size_t t = types.front();
    LOG(debug) << "Processing t = " << t;
    types.pop_front();

    auto children = children_of[t];
    const size_t K = children.size();

    for (auto child : children)
      LOG(debug) << "Child: " << child;

    for (auto child : children)
      types.push_back(child);

    if (parameters.empty_root and t == 0) {
      for (auto child : children)
        p[child] = 1;
      p[T + t] = 1;
      p[t] = 0;
    } else {
      Float u;
      if (independent_switches)
        u = sample_beta<Float>(parameters.mix_alpha, parameters.mix_beta);
      else
        u = sample_beta<Float>(
            counts_spot_type(s, t) + parameters.mix_alpha,
            desc_counts_spot_type(s, t) + parameters.mix_beta);

      LOG(verbose) << "u = " << u;

      for (auto child : children)
        p[child] = (1 - u) * p[t];
      p[T + t] = (1 - u) * p[t];
      p[t] *= u;
    }

    if (K > 0) {
      vector<Float> alpha(K + 1, 0);
      vector<size_t> zeros;
      zeros.push_back(K);
      for (size_t k = 0; k < K; ++k)
        /*
        if ((alpha[k] = counts_type(children[k])
                        + desc_counts_type(children[k]))
        */
        if ((alpha[k] = counts_spot_type(s, children[k])
                        + desc_counts_spot_type(s, children[k]))
            == 0)
          zeros.push_back(k);

      if (true) {
        vector<size_t> still_zeros;
        for (auto k : zeros)
          if ((alpha[k]
               = counts_type(children[k]) + desc_counts_type(children[k]))
              == 0)
            still_zeros.push_back(k);
        for (auto k : still_zeros)
          alpha[k] = parameters.tree_alpha / still_zeros.size();

        double z_zero = 0;
        for (auto k : zeros)
          z_zero += alpha[k];
        for (auto k : zeros)
          alpha[k] *= parameters.tree_alpha / z_zero;
      } else {
        for (auto k : zeros)
          alpha[k] = parameters.tree_alpha / zeros.size();
      }

      for (size_t k = 0; k < K + 1; ++k)
        LOG(debug) << "alpha[" << k << "] = " << alpha[k];

      const bool do_dirichlet_distribution = false;
      if (do_dirichlet_distribution) {
        auto p_transition = sample_dirichlet<Float>(begin(alpha), end(alpha),
                                                    EntropySource::rng);

        for (size_t k = 0; k < K + 1; ++k)
          LOG(debug) << "p_transition[" << k << "] = " << p_transition[k];

        p[T + t] *= p_transition[K];
        for (size_t k = 0; k < K; ++k)
          p[children[k]] *= p_transition[k];
      } else {
        double z = 0;
        for (auto &a : alpha)
          z += a;
        for (size_t k = 0; k < K; ++k)
          p[children[k]] *= alpha[k] / z;
        p[T + t] *= alpha[K] / z;
      }
    }
  }

  if (parameters.empty_root)
    assert(p[0] == 0);
    // p[0] = 0;

  for (size_t t = T; t < 2 * T; ++t)
    if (p[t] > 0)
      p[t] *= sample_beta<Float>(parameters.mix_alpha, parameters.mix_beta);

  for (size_t t = 0; t < 2 * T; ++t)
    LOG(verbose) << "p[" << t << "] = " << p[t];

  for (size_t t = 0; t < T; ++t)
    p[t]
        *= sample_beta<Float>(counts_gene_type(g, t) + parameters.feature_alpha,
                              counts_type(t) - counts_gene_type(g, t)
                                  + (G - 1) * parameters.feature_alpha,
                              EntropySource::rng);

  for (size_t t = T; t < 2 * T; ++t)
    p[t] *= sample_beta<Float>(parameters.feature_alpha,
                               (G - 1) * parameters.feature_alpha,
                               EntropySource::rng);

  for (size_t t = 0; t < 2 * T; ++t)
    LOG(verbose) << "p[" << t << "] = " << p[t];

  // TODO figure out when and why NANs are generated
  for (auto &x : p)
    if (std::isnan(x))
      x = 0;

  return std::discrete_distribution<size_t>(begin(p),
                                            end(p))(EntropySource::rng);
}

void nHDP::register_read(size_t g, size_t s, bool independent_switches) {
  LOG(verbose) << "Register read for gene " << g << " in spot " << s
               << ", G = " << G << " S = " << S << " T = " << T;

  size_t t = sample_type(g, s, independent_switches);
  if (t >= T) {
    size_t parent = t - T;
    t = add_node(parent);
  }

  LOG(info) << "gene " << g << " spot " << s << " -> type " << t;

  counts_gene_type(g, t)++;
  counts_spot_type(s, t)++;
  counts_type(t)++;
  LOG(debug) << "Assigning to desc_counts_spot_type";
  while (t != 0) {
    t = parent_of[t];
    desc_counts_spot_type(s, t)++;
    desc_counts_gene_type(g, t)++;
    desc_counts_type(t)++;
    LOG(debug) << "Assigning to desc_counts_spot_type of t = " << t << ": " << desc_counts_spot_type(s, t);
    LOG(debug) << "Assigning to desc_counts_gene_type of t = " << t << ": " << desc_counts_gene_type(g, t);
  }

}

size_t nHDP::add_node(size_t parent) {
  if (T == maxT) {
    LOG(fatal) << "Reached maximum number of factors!";
    exit(-1);
  }

  parent_of[T] = parent;
  children_of[parent].push_back(T);
  return T++;
}

string nHDP::to_dot() const {
  stringstream ss, tt;
  ss << "digraph {\n";
  list<size_t> types;
  types.push_back(0);
  while (not types.empty()) {
    size_t t = types.front();
    types.pop_front();
    size_t x = 0;
    for (size_t s = 0; s < S; ++s)
      x += counts_spot_type(s, t);
    size_t y = 0;
    for (size_t s = 0; s < S; ++s)
      y += desc_counts_spot_type(s, t);
    ss << t << " [label=\"Factor " << t << "\\n" << x << "\"];\n";
    for (auto child : children_of[t]) {
      types.push_back(child);
      tt << t << " -> " << child << "\n";
    }
  }
  ss << tt.str();
  ss << "}\n";
  return ss.str();
}
}
