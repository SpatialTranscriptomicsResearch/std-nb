#include <list>
#include "entropy.hpp"
#include "log.hpp"
#include "sampling.hpp"
#include "nhdp.hpp"

using namespace std;

namespace PoissonFactorization {

nHDP::nHDP(size_t g, size_t s, size_t t)
    : G(g),
      S(s),
      T(t),
      counts_gene_type(G, T, arma::fill::zeros),
      counts_spot_type(S, T, arma::fill::zeros),
      desc_counts_spot_type(S, T, arma::fill::zeros),
      counts_type(T, arma::fill::zeros) {}

size_t nHDP::parent_of(size_t t) const {
  if (t == 0)
    return 0;
  if (t % 2 == 0)
    return t / 2 - 1;
  else
    return (t - 1) / 2;
}

vector<size_t> nHDP::children_of(size_t t) const {
  vector<size_t> v;
  if (t * 2 + 1 < T)
    v.push_back(t * 2 + 1);
  if ((t + 1) * 2 < T)
    v.push_back((t + 1) * 2);
  return v;
}

size_t nHDP::sample_type(size_t g, size_t s) const {
  LOG(verbose) << "Sample type for gene " << g << " in spot " << s;
  vector<Float> p(T, 1);
  list<size_t> types;
  types.push_front(0);
  while (not types.empty()) {
    size_t t = types.front();
    LOG(debug) << "Processing t = " << t;
    types.pop_front();

    auto children = children_of(t);
    const size_t K = children.size();

    if (K > 0) {
      for (auto child : children)
        LOG(debug) << "Child: " << child;

      for (auto child : children)
        types.push_back(child);

      Float u = sample_beta<Float>(counts_spot_type(s, t) + mix_alpha,
                                   desc_counts_spot_type(s, t) + mix_beta);
      p[t] *= u;
      for (auto child : children)
        p[child] *= 1 - u;

      LOG(debug) << "u = " << u;

      vector<Float> alpha(K, tree_alpha);
      for (size_t k = 0; k < K; ++k)
        alpha[k] += counts_spot_type(s, children[k])
                    + desc_counts_spot_type(s, children[k]);

      for (size_t k = 0; k < K; ++k)
        LOG(debug) << "alpha[" << k << "] = " << alpha[k];

      auto p_transition = sample_dirichlet<Float>(alpha, EntropySource::rng);

      for (size_t k = 0; k < K; ++k)
        LOG(debug) << "p_transition[" << k << "] = " << p_transition[k];

      for (size_t k = 0; k < K; ++k)
        p[children[k]] *= p_transition[k];
    }
  }

  for (size_t t = 0; t < T; ++t)
    LOG(debug) << "p[" << t << "] = " << p[t];

  for (size_t t = 0; t < T; ++t)
    p[t] *= sample_beta<Float>(
        counts_gene_type(g, t) + feature_alpha,
        counts_type(t) - counts_gene_type(g, t) + feature_beta,
        EntropySource::rng);

  for (size_t t = 0; t < T; ++t)
    LOG(verbose) << "p[" << t << "] = " << p[t];

  return std::discrete_distribution<size_t>(begin(p),
                                            end(p))(EntropySource::rng);
}

void nHDP::register_read(size_t g, size_t s) {
  LOG(verbose) << "Register read for gene " << g << " in spot " << s;

  size_t t = sample_type(g, s);

  LOG(info) << "gene " << g << " spot " << s << " -> type " << t;

  counts_gene_type(g, t)++;
  counts_spot_type(s, t)++;
  while (t != 0) {
    t = parent_of(t);
    desc_counts_spot_type(s, t)++;
  }

  counts_type(t)++;
}

string nHDP::to_dot() const {
  stringstream ss, tt;
  ss << "digraph {\n";
  list<size_t> types;
  types.push_back(0);
  while(not types.empty()) {
    size_t t = types.front();
    types.pop_front();
    size_t x = 0;
    for(size_t s = 0; s < S; ++s)
      x += counts_spot_type(s, t);
    size_t y = 0;
    for(size_t s = 0; s < S; ++s)
      y += desc_counts_spot_type(s, t);
    ss << t << " [label=\"Factor " << t << "\\n" << x << "\"];\n";
    for(auto child: children_of(t)) {
      types.push_back(child);
      tt << t << " -> " << child << "\n";
    }
  }
  ss << tt.str();
  ss << "}\n";
  return ss.str();
}
}
