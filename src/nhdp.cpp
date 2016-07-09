#include <list>
#include "entropy.hpp"
#include "log.hpp"
#include "parallel.hpp"
#include "sampling.hpp"
#include "nhdp.hpp"

using namespace std;

namespace PoissonFactorization {

nHDP::nHDP(size_t g, size_t s, size_t t_, const Parameters &params)
    : G(g),
      S(s),
      T(1),
      maxT(t_),
      parameters(params),
      counts_gene_type(G, maxT, arma::fill::zeros),
      counts_spot_type(S, maxT, arma::fill::zeros),
      desc_counts_gene_type(G, maxT, arma::fill::zeros),
      desc_counts_spot_type(S, maxT, arma::fill::zeros),
      counts_type(maxT, arma::fill::zeros),
      desc_counts_type(maxT, arma::fill::zeros),
      parent_of(maxT),
      children_of(maxT) {
  for (size_t t = 0; t < T; ++t)
    parent_of[t] = t;
}

void nHDP::add_hierarchy(size_t t, const Hierarchy &hierarchy,
                         double concentration) {
  counts_gene_type.col(t) = hierarchy.profile * concentration;
  for (auto &h : hierarchy.children) {
    size_t child = add_node(t);
    add_hierarchy(child, h, concentration);
  }
}

Vector nHDP::sample_switches(size_t s, bool independent_switches,
                             bool extra) const {
  // the first T components of the vector represent probabilities for the
  // currently active factors
  // the second T components of the vector represent probabilities for new
  // possible factors
  Vector u((extra ? 2 : 1) * T, arma::fill::zeros);

  list<size_t> types;

  types.push_front(0);
  u[0] = 1;

  LOG(debug) << "counts_spot_type(s,.) = " << counts_spot_type.row(s);
  LOG(debug) << "desc_counts_spot_type(s,.) = " << desc_counts_spot_type.row(s);

  while (not types.empty()) {
    size_t t = types.front();
    LOG(debug) << "Processing t = " << t;
    types.pop_front();

    for (auto child : children_of[t])
      LOG(debug) << "Child: " << child;

    for (auto child : children_of[t])
      types.push_back(child);

    double current_u;
    if (parameters.empty_root and t == 0)
      current_u = 0;
    else {
      if (independent_switches)
        current_u
            = sample_beta<Float>(parameters.mix_alpha, parameters.mix_beta);
      else
        current_u = sample_beta<Float>(
            counts_spot_type(s, t) + parameters.mix_alpha,
            desc_counts_spot_type(s, t) + parameters.mix_beta);
    }
    LOG(verbose) << "current_u = " << current_u;

    for (auto child : children_of[t])
      u[child] = (1 - current_u) * u[t];
    if (extra)
      u[T + t] = (1 - current_u) * u[t];
    u[t] *= current_u;
  }

  if (extra)
    for (size_t t = T; t < 2 * T; ++t)
      if (u[t] > 0)
        u[t] *= sample_beta<Float>(parameters.mix_alpha, parameters.mix_beta);

  return u;
}

Vector nHDP::compute_prior(size_t s, bool independent_switches) const {
  // the first T components of the vector represent probabilities for the currently active factors
  // the second T components of the vector represent probabilities for new possible factors
  Vector p = sample_switches(s, independent_switches, true);

  list<size_t> types;
  types.push_front(0);

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

    if (K > 0) {
      vector<Float> alpha(K + 1, 0);
      vector<size_t> zeros;
      zeros.push_back(K);
      for (size_t k = 0; k < K; ++k)
        if ((alpha[k] = counts_spot_type(s, children[k])
                        + desc_counts_spot_type(s, children[k]))
            == 0)
          zeros.push_back(k);

      if (true) {
        vector<size_t> still_zeros;
        for (auto k : zeros)
          if (k == K
              or (alpha[k]
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
        for (size_t k = 0; k < K + 1; ++k)
          LOG(debug) << "alpha[" << k << "] = " << alpha[k];
        for (size_t k = 0; k < K + 1; ++k)
          LOG(debug) << "alpha[" << k << "] / z = " << alpha[k] / z;
      }
    }
  }

  if (parameters.empty_root)
    assert(p[0] == 0);

  for (size_t t = 0; t < 2 * T; ++t)
    LOG(verbose) << "p[" << t << "] = " << p[t];

  return p;
}

size_t nHDP::sample_type(size_t g, size_t s, bool independent_switches) const {
  LOG(verbose) << "Sample type for gene " << g << " in spot " << s;

  Vector p = compute_prior(s, independent_switches);

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

void nHDP::register_read(size_t g, size_t s, size_t t, size_t n, bool update_ancestors) {
  counts_gene_type(g, t) += n;
  counts_spot_type(s, t) += n;
  counts_type(t) += n;
  if (update_ancestors) {
    LOG(debug) << "Assigning to desc_counts_spot_type";
    while (t != 0) {
      t = parent_of[t];
      desc_counts_spot_type(s, t) += n;
      desc_counts_gene_type(g, t) += n;
      desc_counts_type(t) += n;
      LOG(debug) << "Assigning to desc_counts_spot_type of t = " << t << ": "
                 << desc_counts_spot_type(s, t);
      LOG(debug) << "Assigning to desc_counts_gene_type of t = " << t << ": "
                 << desc_counts_gene_type(g, t);
    }
  }
}

void nHDP::update_ancestors() {
  for (size_t t = T; t > 0;) {  // skip the root
    t--;
    size_t parent = parent_of[t];
    if (parent != t) {
      desc_counts_spot_type.col(parent) += counts_spot_type.col(t);
      desc_counts_gene_type.col(parent) += counts_gene_type.col(t);
      desc_counts_type(parent) += counts_type(t);
    }
  }
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

  register_read(g, s, t, 1, true);
}

Matrix nHDP::sample_gene_expression() const {
  LOG(info) << "Sampling gene expression";
  Matrix phi(G, T);
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t t = 0; t < T; ++t) {
    LOG(info) << "Sampling gene expression for type " << t;
    Vector alpha = counts_gene_type.col(t) + parameters.feature_alpha;

    // LOG(verbose) << "Alpha = " << t;
    auto col
        = sample_dirichlet<Float>(begin(alpha), end(alpha), EntropySource::rng);
    for (size_t g = 0; g < G; ++g)
      phi(g, t) = col[g];
  }
  LOG(info) << "Done: Sampling gene expression";
  return phi;
}

Vector nHDP::sample_transitions(size_t s) const {
  LOG(info) << "Sampling transition probabilities for spot " << s;
  Vector p(T, arma::fill::zeros);

  list<size_t> types;
  types.push_front(0);

  p[0] = 1;

  for (size_t t = 0; t < T; ++t)
    LOG(verbose) << "p[" << t << "] = " << p[t];

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

    if (K > 0) {
      vector<Float> alpha(K, 0);
      vector<size_t> zeros;
      zeros.push_back(K);
      for (size_t k = 0; k < K; ++k) {
        alpha[k] = counts_spot_type(s, children[k])
                   + desc_counts_spot_type(s, children[k]);
        if (alpha[k] == 0)
          alpha[k] = parameters.tree_alpha;
      }

      for (size_t k = 0; k < K; ++k)
        LOG(debug) << "alpha[" << k << "] = " << alpha[k];

      const bool do_dirichlet_distribution = true;
      if (do_dirichlet_distribution) {
        auto p_transition = sample_dirichlet<Float>(begin(alpha), end(alpha),
                                                    EntropySource::rng);

        for (size_t k = 0; k < K; ++k)
          LOG(debug) << "p_transition[" << k << "] = " << p_transition[k];

        for (size_t k = 0; k < K; ++k) {
          auto x = p_transition[k] * p[t];
          LOG(debug) << "x = " << x;
          LOG(debug) << "children[" << k << "] = " << children[k];
          p[children[k]] = p_transition[k] * p[t];
        }
      } else {
        double z = 0;
        for (auto &a : alpha)
          z += a;
        for (size_t k = 0; k < K; ++k)
          p[children[k]] = alpha[k] / z * p[t];
        for (size_t k = 0; k < K; ++k)
          LOG(debug) << "alpha[" << k << "] = " << alpha[k];
        for (size_t k = 0; k < K; ++k)
          LOG(debug) << "alpha[" << k << "] / z = " << alpha[k] / z;
      }
    }
  }

  for (size_t t = 0; t < T; ++t)
    LOG(verbose) << "p[" << t << "] = " << p[t];

  LOG(info) << "Done: Sampling transition probabilities for spot " << s;
  return p;
}

nHDP nHDP::sample(const IMatrix &counts, bool independent_switches) const {
  LOG(info) << "Performing sampling";
  nHDP model = *this;
  model.counts_gene_type.fill(0);
  model.counts_spot_type.fill(0);
  model.desc_counts_gene_type.fill(0);
  model.desc_counts_spot_type.fill(0);
  model.counts_type.fill(0);
  model.desc_counts_type.fill(0);

  // sample gene expression profile
  Matrix phi = sample_gene_expression();

  for (size_t s = 0; s < S; ++s) {
    LOG(info) << "Performing sampling for spot " << s;
    auto switches = sample_switches(s, independent_switches, false);
    auto transitions = sample_transitions(s);
    LOG(debug) << "switches = " << switches;
    LOG(debug) << "transitions = " << transitions;
    auto p_tree = switches % transitions;
    LOG(debug) << "P(tree) = " << p_tree;
    // sample statistics
    for (size_t g = 0; g < G; ++g) {
      Vector p = p_tree % phi.row(g).t();
      normalize(begin(p), end(p));
      auto split_counts
          = sample_multinomial<size_t>(counts(g, s), begin(p), end(p));
      for (size_t t = 0; t < T; ++t)
        model.register_read(g, s, t, split_counts[t], false);
    }
  }
  model.update_ancestors();
  return model;
}

/*
void nHDP::register_reads(size_t s,  const Vector reads, bool independent_switches) {
  LOG(verbose) << "Register reads in spot " << s
               << ", G = " << G << " S = " << S << " T = " << T;

  vector<Float> p = compute_prior(s, independent_switches);

  for(size_t g = 0; g < G; ++g)
    register_reads(g, s, reads[g], p);
}
*/

size_t nHDP::add_node(size_t parent) {
  if (T == maxT) {
    LOG(fatal) << "Reached maximum number of factors!";
    exit(-1);
  }

  parent_of[T] = parent;
  children_of[parent].push_back(T);
  return T++;
}

string nHDP::to_dot(double threshold) const {
  stringstream ss, tt;
  ss << "digraph {\n";
  list<size_t> types;
  types.push_back(0);
  size_t total = 0;
  vector<bool> skipped(T, true);
  while (not types.empty()) {
    size_t t = types.front();
    types.pop_front();
    size_t x = 0;
    for (size_t s = 0; s < S; ++s)
      x += counts_spot_type(s, t);
    size_t y = 0;
    for (size_t s = 0; s < S; ++s)
      y += desc_counts_spot_type(s, t);
    if (t == 0)
      total = x + y;

    if (1.0 * (x + y) / total > threshold)
      skipped[t] = false;

    if (not skipped[t]) {
      ss << t << " [label=\"Factor " << t << "\\n" << x << " "
         << 100.0 * x / total << "%\\n";
      if (y > 0)
        ss << (x + y) << " " << 100.0 * (x + y) / total << "%";
      ss << "\"];\n";

      for (auto child : children_of[t])
        types.push_back(child);
    }
  }
  types.push_back(0);
  while (not types.empty()) {
    size_t t = types.front();
    types.pop_front();
    if (not skipped[t])
      for (auto child : children_of[t])
        if (not skipped[child]) {
          types.push_back(child);
          tt << t << " -> " << child << "\n";
        }
  }
  ss << tt.str();
  ss << "}\n";
  return ss.str();
}
}
