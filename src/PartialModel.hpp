#ifndef MIX_HPP
#define MIX_HPP

#include "aux.hpp"
#include "compression.hpp"
#include "io.hpp"
#include "log.hpp"
#include "parallel.hpp"
#include "pdist.hpp"
#include "priors.hpp"
#include "sampling.hpp"

namespace STD {

const Float phi_scaling = 1.0;

struct Theta {
  using prior_type = PRIOR::THETA::Gamma;
  Theta(size_t dim1_, size_t dim2_, const Parameters &params)
      : dim1(dim1_),
        dim2(dim2_),
        matrix(dim1, dim2),
        parameters(params),
        prior(dim1, dim2, parameters) {
    initialize();
  };
  size_t dim1, dim2;
  Matrix matrix;
  Parameters parameters;
  prior_type prior;

  void initialize_factor(size_t t);
  void initialize();

  void enforce_positive_parameters(const std::string &tag) {
    enforce_positive_and_warn(tag, matrix);
    prior.enforce_positive_parameters(tag);
  };

  // TODO rename to something like sample_features
  template <typename Experiment, typename... Args>
  void sample(const Experiment &experiment, const Args &... args);

  // TODO rename to something like sample_weights
  template <typename Experiment, typename... Args>
  void sample_field(const Experiment &experiment, Matrix &field,
                    const Args &... args);

  std::string gen_path_stem(const std::string &prefix) const {
    return prefix + "mix-hiergamma";
  };

  void store(const std::string &prefix,
             const std::vector<std::string> &spot_names,
             const std::vector<std::string> &factor_names,
             const std::vector<size_t> &order) const {
    const auto path = gen_path_stem(prefix);
    write_matrix(matrix, path + FILENAME_ENDING, parameters.compression_mode,
                 spot_names, factor_names, order);
    prior.store(path, spot_names, factor_names, order);
  };

  void restore(const std::string &prefix) {
    const auto path = gen_path_stem(prefix);
    matrix = parse_file<Matrix>(path + FILENAME_ENDING, read_matrix, "\t");
    prior.restore(path);
  };

  double log_likelihood_factor(size_t t) const;
  double log_likelihood() const;
};

template <typename Type, typename Res>
void perform_sampling(const Type &observed, const Type &explained, Res &m,
                      bool over_relax) {
#pragma omp parallel if (DO_PARALLEL)
  {
    const size_t thread_num = omp_get_thread_num();
    if (not over_relax) {
#pragma omp for
      for (size_t x = 0; x < observed.n_elem; ++x)
        // NOTE: gamma_distribution takes a shape and scale parameter
        m[x] = std::gamma_distribution<Float>(
            observed[x], 1.0 / explained[x])(EntropySource::rngs[thread_num]);
    } else {
#pragma omp for
      for (size_t x = 0; x < observed.n_elem; ++x) {
        // NOTE: gamma_cdf takes a shape and scale parameter
        double u = gamma_cdf(m[x], observed[x], 1.0 / explained[x]);
        const size_t K = 20;  // TODO introduce CLI switch
        double u_prime = u;
        size_t r = std::binomial_distribution<size_t>(
            K, u)(EntropySource::rngs[thread_num]);
        if (r > K - r)
          u_prime = u * sample_beta<Float>(K - r + 1, 2 * r - K,
                                           EntropySource::rngs[thread_num]);
        else if (r < K - r)
          u_prime
              = 1
                - (1 - u) * sample_beta<Float>(r + 1, K - 2 * r,
                                               EntropySource::rngs[thread_num]);
        u_prime = 0.5 + (u_prime - 0.5) * 0.95;
        /*
        double prev = m[x];
        std::cerr << prev << " " << observed[x] << " " << explained[x] << " "
                  << u << " " << u_prime << " -> " << std::flush;
                  */

        // NOTE: inverse_gamma_cdf takes a shape and scale parameter
        m[x] = inverse_gamma_cdf(u_prime, observed[x], 1.0 / explained[x]);
        /*
        std::cerr << m[x] << std::endl;
        */
      }
    }
  }
}

/** sample theta */
template <typename Experiment, typename... Args>
void Theta::sample_field(const Experiment &experiment, Matrix &field,
                         const Args &... args) {
  LOG(verbose) << "Sampling Θ from Gamma distribution";

  const bool convolve = parameters.targeted(Target::field);
  const double PRIOR = parameters.hyperparameters.field_residual_prior;

  const auto intensities = experiment.marginalize_genes(args...);

  Matrix observed = experiment.contributions_spot_type;
  // TODO play with switching
  Matrix explained = experiment.spot * intensities.t();
  // Matrix explained = experiment.theta_explained_spot_type;
  if (convolve) {
    explained %= field;
    observed += PRIOR;
    explained += PRIOR;
  } else {
    observed.each_row() += prior.r.t();
    explained.each_row() += prior.p.t();
  }
  perform_sampling(observed, explained, matrix, parameters.over_relax);
  if (convolve)
    matrix %= field;

#pragma omp parallel for if (DO_PARALLEL)
  for (size_t s = 0; s < dim1; ++s)
    for (size_t t = 0; t < dim2; ++t)
      matrix(s, t) = std::max<Float>(std::numeric_limits<Float>::denorm_min(),
                                     matrix(s, t));
}
}

#endif
