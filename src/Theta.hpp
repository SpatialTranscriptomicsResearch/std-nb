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
  Theta(size_t S_, size_t T_, const Parameters &params,
        const prior_type &prior);
  size_t S, T;
  Matrix matrix;
  Parameters parameters;

  void enforce_positive_parameters(const std::string &tag) {
    enforce_positive_and_warn(tag, matrix);
  };

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
  };

  void restore(const std::string &prefix) {
    const auto path = gen_path_stem(prefix);
    matrix = parse_file<Matrix>(path + FILENAME_ENDING, read_matrix, "\t");
  };
};
}

#endif
