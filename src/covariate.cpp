#include "covariate.hpp"
#include "aux.hpp"
#include "compression.hpp"
#include "design.hpp"
#include "io.hpp"

using namespace std;
using STD::Matrix;

string Covariate::to_string() const {
  string str = "Covariate: '" + label + "':";
  for (auto &value : values)
    str += " '" + value + "'";
  return str;
}

std::ostream &operator<<(std::ostream &os, const Covariate &covariate) {
  os << covariate.to_string();
  return os;
}

bool kind_included(CovariateTerm::Kind kind, CovariateTerm::Kind x) {
  return (kind & x) == x;
}

bool CovariateTerm::gene_dependent() const {
  return kind_included(kind, CovariateTerm::Kind::gene);
}

bool CovariateTerm::spot_dependent() const {
  return kind_included(kind, CovariateTerm::Kind::spot);
}

bool CovariateTerm::type_dependent() const {
  return kind_included(kind, CovariateTerm::Kind::type);
}

void CovariateTerm::store(const string &path, CompressionMode compression_mode,
                          const vector<string> &gene_names,
                          const vector<string> &spot_names,
                          const vector<string> &type_names,
                          vector<size_t> type_order) const {
  switch (kind) {
    case CovariateTerm::Kind::scalar:
      write_matrix(values, path, compression_mode, {}, {});
      break;
    case CovariateTerm::Kind::gene:
      write_matrix(values, path, compression_mode, gene_names, {});
      break;
    case CovariateTerm::Kind::spot:
      write_matrix(values, path, compression_mode, spot_names, {});
      break;
    case CovariateTerm::Kind::type:
      // TODO cov type order
      write_matrix(values, path, compression_mode, type_names, {}, {},
                   type_order);
      break;
    case CovariateTerm::Kind::gene_type:
      write_matrix(values, path, compression_mode, gene_names, type_names,
                   type_order);
      break;
    case CovariateTerm::Kind::spot_type:
      write_matrix(values, path, compression_mode, spot_names, type_names,
                   type_order);
      break;
  }
}

void CovariateTerm::restore(const string &path) {
  values = parse_file<Matrix>(path, read_matrix, "\t");
}

string to_string(const CovariateTerm::Kind &kind) {
  switch (kind) {
    case CovariateTerm::Kind::scalar:
      return "scalar";
    case CovariateTerm::Kind::gene:
      return "gene-dependent";
    case CovariateTerm::Kind::spot:
      return "spot-dependent";
    case CovariateTerm::Kind::type:
      return "type-dependent";
    case CovariateTerm::Kind::gene_type:
      return "gene- and type-dependent";
    case CovariateTerm::Kind::spot_type:
      return "spot- and type-dependent";
    default:
      throw std::runtime_error("Error: invalid CovariateTerm::Kind.");
  }
}

string to_token(const CovariateTerm::Kind &kind) {
  switch (kind) {
    case CovariateTerm::Kind::scalar:
      return "scalar";
    case CovariateTerm::Kind::gene:
      return "gene";
    case CovariateTerm::Kind::spot:
      return "spot";
    case CovariateTerm::Kind::type:
      return "type";
    case CovariateTerm::Kind::gene_type:
      return "gene-type";
    case CovariateTerm::Kind::spot_type:
      return "spot-type";
    default:
      throw std::runtime_error("Error: invalid CovariateTerm::Kind.");
  }
}

CovariateTerm::CovariateTerm(size_t G, size_t S, size_t T, Kind kind_,
                             CovariateInformation info_)
    : kind(kind_), info(info_) {
  switch (kind) {
    case Kind::scalar:
      values = Matrix::Ones(1, 1);
      return;
    case Kind::gene:
      values = Matrix::Ones(G, 1);
      return;
    case Kind::spot:
      values = Matrix::Ones(S, 1);
      return;
    case Kind::type:
      values = Matrix::Ones(T, 1);
      return;
    case Kind::gene_type:
      values = Matrix::Ones(G, T);
      return;
    case Kind::spot_type:
      values = Matrix::Ones(S, T);
      return;
  }
}

double &CovariateTerm::get(size_t g, size_t t, size_t s) {
  switch (kind) {
    case Kind::scalar:
      return values(0, 0);
    case Kind::gene:
      return values(g, 0);
    case Kind::spot:
      return values(s, 0);
    case Kind::type:
      return values(t, 0);
    case Kind::gene_type:
      return values(g, t);
    case Kind::spot_type:
      return values(s, t);
    default:
      throw std::runtime_error("Error: invalid CovariateTerm::Kind.");
  }
}

double CovariateTerm::get(size_t g, size_t t, size_t s) const {
  switch (kind) {
    case Kind::scalar:
      return values(0, 0);
    case Kind::gene:
      return values(g, 0);
    case Kind::spot:
      return values(s, 0);
    case Kind::type:
      return values(t, 0);
    case Kind::gene_type:
      return values(g, t);
    case Kind::spot_type:
      return values(s, t);
    default:
      throw std::runtime_error("Error: invalid CovariateTerm::Kind.");
  }
}

string CovariateInformation::to_string(const Covariates &covariates) const {
  string s;
  for (size_t i = 0; i < idxs.size(); ++i) {
    if (i > 0)
      s += ",";
    if (covariates[idxs[i]].label == DesignNS::unit_label)
      s += "intercept";
    else
      s += covariates[idxs[i]].label + "="
           + covariates[idxs[i]].values[vals[i]];
  }
  if (idxs.size() == 0)
    s = "global";
  return s;
}
