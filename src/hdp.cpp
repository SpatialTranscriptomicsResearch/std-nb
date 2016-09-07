#include <list>
#include "aux.hpp"
#include "entropy.hpp"
#include "log.hpp"
#include "parallel.hpp"
#include "sampling.hpp"
#include "hdp.hpp"

using namespace std;

namespace PoissonFactorization {

HDP::HDP(size_t g, size_t s, size_t t_, const Parameters &params)
    : G(g),
      S(s),
      T(0),
      maxT(t_),
      parameters(params),
      counts_gene_type(G, maxT, arma::fill::zeros),
      counts_spot_type(S, maxT, arma::fill::zeros),
      counts_type(maxT, arma::fill::zeros) {}


Vector HDP::compute_prior(const Vector &v) const {
  LOG(debug) << "counts_spot_type(s,.) = " << v;

  // the first T components represent probabilities for active factors
  // the (T+1)-st component represents the probability for a new factor
  vector<Float> p(T + 1, 0);

  vector<size_t> zeros;
  zeros.push_back(T);
  for (size_t t = 0; t < T; ++t)
    p[t] = v(t);

  for (size_t t = 0; t < T; ++t)
    if (p[t] == 0)
      zeros.push_back(t);

  for (auto t : zeros)
    p[t] = parameters.mix_alpha / zeros.size();

  for (size_t t = 0; t < T + 1; ++t)
    LOG(debug) << "p[" << t << "] = " << p[t];

  double z = 0;
  for (auto &a : p)
    z += a;
  for (auto &a : p)
    a /= z;

  for (size_t t = 0; t < T + 1; ++t)
    LOG(verbose) << "p[" << t << "] = " << p[t];

  return p;
}

size_t HDP::sample_type(size_t g, size_t s) const {
  LOG(verbose) << "Sample type for gene " << g << " in spot " << s;

  Vector p = compute_prior(counts_spot_type.row(s));

  for (size_t t = 0; t < T; ++t)
    p[t]
        *= sample_beta<Float>(counts_gene_type(g, t) + parameters.feature_alpha,
                              counts_type(t) - counts_gene_type(g, t)
                                  + (G - 1) * parameters.feature_alpha,
                              EntropySource::rng);

  p[T] *= sample_beta<Float>(parameters.feature_alpha,
                             (G - 1) * parameters.feature_alpha,
                             EntropySource::rng);

  for (size_t t = 0; t <= T; ++t)
    LOG(verbose) << "p[" << t << "] = " << p[t];

  // TODO figure out when and why NANs are generated
  for (auto &x : p)
    if (std::isnan(x))
      x = 0;

  return std::discrete_distribution<size_t>(begin(p),
                                            end(p))(EntropySource::rng);
}

void HDP::register_read(size_t g, size_t s, size_t t, size_t n) {
  counts_gene_type(g, t) += n;
  counts_spot_type(s, t) += n;
  counts_type(t) += n;
}

void HDP::register_read(size_t g, size_t s) {
  LOG(verbose) << "Register read for gene " << g << " in spot " << s
               << ", G = " << G << " S = " << S << " T = " << T;

  size_t t = sample_type(g, s);
  if (t >= T)
    t = add_factor();

  LOG(verbose) << "gene " << g << " spot " << s << " -> type " << t;

  register_read(g, s, t, 1);
}

Matrix HDP::sample_gene_expression() const {
  LOG(info) << "Sampling gene expression";
  Matrix phi(G, T);
#pragma omp parallel for if (DO_PARALLEL)
  for (size_t t = 0; t < T; ++t) {
    LOG(verbose) << "Sampling gene expression for type " << t;
    Vector alpha = counts_gene_type.col(t) + parameters.feature_alpha;

    // LOG(verbose) << "Alpha = " << t;
    auto col = sample_dirichlet<Float>(
        begin(alpha), end(alpha), EntropySource::rngs[omp_get_thread_num()]);
    for (size_t g = 0; g < G; ++g)
      phi(g, t) = col[g];
  }
  LOG(info) << "Done: Sampling gene expression";
  return phi;
}

HDP HDP::sample(const IMatrix &counts) const {
  LOG(info) << "Sampling";
  HDP model = *this;
  model.counts_gene_type.fill(0);
  model.counts_spot_type.fill(0);
  model.counts_type.fill(0);

  // sample gene expression profile
  Matrix phi = sample_gene_expression();

  double l = 0;

  // prepare a random order of the samples
  // The purpose of this is to guarantee an even balancing of the work load
  // across the threads. In default order it would be uneven because spots of
  // different experiments do not have the same relative frequency of zeros.
  vector<size_t> order(S);
  iota(begin(order), end(order), 0);
  shuffle(begin(order), end(order), EntropySource::rng);

#pragma omp parallel if (DO_PARALLEL)
  {
    IMatrix c_gene_type(G, T, arma::fill::zeros);
    IMatrix c_spot_type(S, T, arma::fill::zeros);
    IVector c_type(T, arma::fill::zeros);
    auto rng = EntropySource::rngs[omp_get_thread_num()];
    double ll = 0;
#pragma omp for
    for (size_t s_ = 0; s_ < S; ++s_) {
      size_t s = order[s_];
      LOG(verbose) << "Sampling for spot " << s;
      auto p_tree = compute_prior(counts_spot_type.row(s).t());
      LOG(debug) << "P(tree) = " << p_tree;
      // sample statistics
      for (size_t g = 0; g < G; ++g)
        if (counts(g, s) > 0) {
          Vector phi_(T+1, arma::fill::zeros);
          for(size_t t = 0; t < T; ++t)
            phi_[t] = phi(g,t);
          phi_[T] = 1; // TODO

          Vector p = p_tree % phi_;
          normalize(begin(p), end(p));
          // TODO consider sorting
          auto split_counts
              = sample_multinomial<size_t>(counts(g, s), begin(p), end(p), rng);
          for (size_t t = 0; t < T; ++t) {
            if (split_counts[t] > 0) {
              ll += split_counts[t] * log(p(t)) - lgamma(split_counts[t] + 1);
              c_gene_type(g, t) += split_counts[t];
              c_spot_type(s, t) += split_counts[t];
              c_type(t) += split_counts[t];
            }
          }
          if(split_counts[T] > 0) {
            // auto t = add_factor(); TODO
            size_t t = 0; // TODO
            c_gene_type(g, t) += split_counts[t];
            c_spot_type(s, t) += split_counts[t];
            c_type(t) += split_counts[t];
          }
          ll += lgamma(counts(g, s) + 1);
        }
    }

#pragma omp critical
    {
      l += ll;
      for (size_t t = 0; t < T; ++t) {
        for (size_t g = 0; g < G; ++g)
          model.counts_gene_type(g, t) += c_gene_type(g, t);
        for (size_t s = 0; s < S; ++s)
          model.counts_spot_type(s, t) += c_spot_type(s, t);
        model.counts_type(t) += c_type(t);
      }
    }
  }
  LOG(info) << "Log-likelihood = " << l;
  return model;
}

size_t HDP::add_factor() {
  if (T == maxT) {
    LOG(fatal) << "Reached maximum number of factors!";
    exit(-1);
  }

  return T++;
}


HDP &HDP::operator+=(const HDP &m) {
  counts_gene_type += m.counts_gene_type;
  counts_spot_type += m.counts_spot_type;
  counts_type += m.counts_type;
  return *this;
}
}
