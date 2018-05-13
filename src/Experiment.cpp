#include "Experiment.hpp"
#include "Model.hpp"
#include "aux.hpp"
#include "gamma_func.hpp"
#include "hamiltonian_monte_carlo.hpp"
#include "io.hpp"
#include "metropolis_hastings.hpp"
#include "rprop.hpp"

using namespace std;

namespace STD {

Experiment::Experiment(Model *model_, const Counts &counts_, size_t T_,
                       const Parameters &parameters_)
    : model(model_),
      G(counts_.num_genes()),
      S(counts_.num_samples()),
      T(T_),
      counts(counts_),
      coords(counts.parse_coords()),
      scale_ratio(counts.matrix->sum() / S),
      parameters(parameters_),
      contributions_gene_type(Matrix::Zero(G, T)),
      contributions_spot_type(Matrix::Zero(S, T)),
      contributions_gene(rowSums<Vector>(*counts.matrix)),
      contributions_spot(colSums<Vector>(*counts.matrix)) {
  LOG(debug) << "Experiment G = " << G << " S = " << S << " T = " << T;
  /* TODO consider to initialize:
   * contributions_gene_type
   * contributions_spot_type
   */
  LOG(trace) << "Coords: " << coords;
}

void Experiment::ensure_dimensions() const {
  for (auto &coeffs : {rate_coeffs, odds_coeffs})
    for (auto &coeff : coeffs) {
      int nrow = 0;
      int ncol = 0;
      switch (coeff->kind) {
        case Coefficient::Kind::scalar:
          nrow = ncol = 1;
          break;
        case Coefficient::Kind::gene:
          nrow = G;
          ncol = 1;
          break;
        case Coefficient::Kind::spot:
          nrow = S;
          ncol = 1;
          break;
        case Coefficient::Kind::type:
          nrow = T;
          ncol = 1;
          break;
        case Coefficient::Kind::gene_type:
          nrow = G;
          ncol = T;
          break;
        case Coefficient::Kind::spot_type:
          nrow = S;
          ncol = T;
          break;
      }
      if (coeff->values.rows() != nrow or coeff->values.cols() != ncol)
        throw std::runtime_error("Error: mismatched dimension on coefficient: "
                                 + to_string(nrow) + "x" + to_string(ncol)
                                 + " vs " + coeff->to_string());
    }
}

void Experiment::store(const string &prefix,
                       const vector<size_t> &order) const {
  auto factor_names = form_factor_names(T);
  auto &gene_names = counts.row_names;
  auto &spot_names = counts.col_names;

  string suffix = "";
  string extension = boost::filesystem::path(counts.path).extension().c_str();
  if (extension == ".gz" or extension == ".bz2")
    suffix = extension;
  boost::filesystem::create_symlink(
      boost::filesystem::canonical(counts.path),
      prefix + "counts" + FILENAME_ENDING + suffix);

#pragma omp parallel sections if (DO_PARALLEL)
  {
#pragma omp section
    write_matrix(contributions_gene_type,
                 prefix + "contributions_gene_type" + FILENAME_ENDING,
                 parameters.compression_mode, gene_names, factor_names, order);
#pragma omp section
    write_matrix(contributions_spot_type,
                 prefix + "contributions_spot_type" + FILENAME_ENDING,
                 parameters.compression_mode, spot_names, factor_names, order);
#pragma omp section
    write_vector(contributions_gene,
                 prefix + "contributions_gene" + FILENAME_ENDING,
                 parameters.compression_mode, gene_names);
#pragma omp section
    write_vector(contributions_spot,
                 prefix + "contributions_spot" + FILENAME_ENDING,
                 parameters.compression_mode, spot_names);
  }
}

void Experiment::restore(const string &prefix) {
  contributions_gene_type = parse_file<Matrix>(
      prefix + "contributions_gene_type" + FILENAME_ENDING, read_matrix, "\t");
  contributions_spot_type = parse_file<Matrix>(
      prefix + "contributions_spot_type" + FILENAME_ENDING, read_matrix, "\t");
  contributions_gene
      = parse_file<Vector>(prefix + "contributions_gene" + FILENAME_ENDING,
                           read_vector<Vector>, "\t");
  contributions_spot
      = parse_file<Vector>(prefix + "contributions_spot" + FILENAME_ENDING,
                           read_vector<Vector>, "\t");
}

ostream &operator<<(ostream &os, const Experiment &experiment) {
  size_t reads = experiment.counts.matrix->sum();
  os << "Experiment "
     << "G = " << experiment.G << " "
     << "S = " << experiment.S << " "
     << "T = " << experiment.T << " "
     << "R = " << reads << " -> "
     << "R/S = " << 1.0 * reads / experiment.S << endl;
  return os;
}

Experiment operator+(const Experiment &a, const Experiment &b) {
  Experiment experiment = a;

  experiment.contributions_gene_type += b.contributions_gene_type;
  experiment.contributions_spot_type += b.contributions_spot_type;
  experiment.contributions_gene += b.contributions_gene;
  experiment.contributions_spot += b.contributions_spot;

  return experiment;
}
}  // namespace STD
