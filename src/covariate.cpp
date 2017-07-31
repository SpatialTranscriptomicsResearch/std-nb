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

ostream &operator<<(ostream &os, const Covariate &covariate) {
  os << covariate.to_string();
  return os;
}

bool kind_included(Coefficient::Kind kind, Coefficient::Kind x) {
  return (kind & x) == x;
}

bool Coefficient::gene_dependent() const {
  return kind_included(kind, Coefficient::Kind::gene);
}

bool Coefficient::spot_dependent() const {
  return kind_included(kind, Coefficient::Kind::spot);
}

bool Coefficient::type_dependent() const {
  return kind_included(kind, Coefficient::Kind::type);
}

void Coefficient::store(const string &path, CompressionMode compression_mode,
                        const vector<string> &gene_names,
                        const vector<string> &spot_names,
                        const vector<string> &type_names,
                        vector<size_t> type_order) const {
  switch (kind) {
    case Coefficient::Kind::scalar:
      write_matrix(values, path, compression_mode, {}, {});
      break;
    case Coefficient::Kind::gene:
      write_matrix(values, path, compression_mode, gene_names, {});
      break;
    case Coefficient::Kind::spot:
      write_matrix(values, path, compression_mode, spot_names, {});
      break;
    case Coefficient::Kind::type:
      // TODO cov type order
      write_matrix(values, path, compression_mode, type_names, {}, {},
                   type_order);
      break;
    case Coefficient::Kind::gene_type:
      write_matrix(values, path, compression_mode, gene_names, type_names,
                   type_order);
      break;
    case Coefficient::Kind::spot_type:
      write_matrix(values, path, compression_mode, spot_names, type_names,
                   type_order);
      break;
  }
}

void Coefficient::restore(const string &path) {
  values = parse_file<Matrix>(path, read_matrix, "\t");
}

string to_string(const Coefficient::Kind &kind) {
  switch (kind) {
    case Coefficient::Kind::scalar:
      return "scalar";
    case Coefficient::Kind::gene:
      return "gene-dependent";
    case Coefficient::Kind::spot:
      return "spot-dependent";
    case Coefficient::Kind::type:
      return "type-dependent";
    case Coefficient::Kind::gene_type:
      return "gene- and type-dependent";
    case Coefficient::Kind::spot_type:
      return "spot- and type-dependent";
    default:
      throw std::runtime_error("Error: invalid Coefficient::Kind.");
  }
}

string to_string(const Coefficient::Variable &variable) {
  switch (variable) {
    case Coefficient::Variable::rate:
      return "rate";
    case Coefficient::Variable::variance:
      return "variance";
    case Coefficient::Variable::prior:
      return "prior";
    default:
      throw std::runtime_error("Error: invalid Coefficient::Variable.");
  }
}

string to_token(const Coefficient::Kind &kind) {
  switch (kind) {
    case Coefficient::Kind::scalar:
      return "scalar";
    case Coefficient::Kind::gene:
      return "gene";
    case Coefficient::Kind::spot:
      return "spot";
    case Coefficient::Kind::type:
      return "type";
    case Coefficient::Kind::gene_type:
      return "gene-type";
    case Coefficient::Kind::spot_type:
      return "spot-type";
    default:
      throw std::runtime_error("Error: invalid Coefficient::Kind.");
  }
}

Coefficient::Coefficient(size_t G, size_t T, size_t S, Variable variable_,
                         Kind kind_, CovariateInformation info_)
    : variable(variable_), kind(kind_), info(info_) {
  // TODO cov prior fill prior_idx
  switch (variable) {
    case Variable::rate:
      distribution = Distribution::gamma;
      break;
    case Variable::variance:
      distribution = Distribution::beta_prime;
      break;
    case Variable::prior:
      // TODO cov prior make variable
      distribution = Distribution::gamma;
      break;
    default:
      throw std::runtime_error(
          "Error: no distribution assigned for Variable type.");
      break;
  }
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

double &Coefficient::get(size_t g, size_t t, size_t s) {
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
      throw std::runtime_error("Error: invalid Coefficient::Kind.");
  }
}

double Coefficient::get(size_t g, size_t t, size_t s) const {
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
      throw std::runtime_error("Error: invalid Coefficient::Kind.");
  }
}

void Coefficient::compute_gradient(const vector<Coefficient> &coeffs,
                                   vector<Coefficient> &grad_coeffs,
                                   size_t idx) const {
  LOG(debug) << "Coefficient::compute_gradient " << to_string(kind) << " "
             << to_string(variable) << " " << idx;
  if (prior_idxs.size() < 2)
    return;
  size_t parent_a = prior_idxs[0];
  size_t parent_b = prior_idxs[1];
  bool parent_a_flexible
      = grad_coeffs[parent_a].distribution != Distribution::fixed;
  bool parent_b_flexible
      = grad_coeffs[parent_b].distribution != Distribution::fixed;
  switch (distribution) {
    case Distribution::gamma:
      visit([&](size_t g, size_t t, size_t s) {
        double a = coeffs[parent_a].get(g, t, s);
        double b = coeffs[parent_b].get(g, t, s);
        double x = get(g, t, s);

        grad_coeffs[idx].get(g, t, s) += (a - 1) - x * b;

        if (parent_a_flexible)
          grad_coeffs[parent_a].get(g, t, s)
              += a * (log(b) - digamma(a) + log(x));
        if (parent_b_flexible)
          grad_coeffs[parent_b].get(g, t, s) += a - b * x;
      });
      break;
    case Distribution::beta_prime:
      visit([&](size_t g, size_t t, size_t s) {
        double a = coeffs[parent_a].get(g, t, s);
        double b = coeffs[parent_b].get(g, t, s);
        double x = get(g, t, s);
        double p = neg_odds_to_prob(x);

        grad_coeffs[idx].get(g, t, s) += (a + b - 2) * p - a + 1;

        if (parent_a_flexible)
          grad_coeffs[parent_a].get(g, t, s)
              += log(x) - log(1 + x) + digamma_diff(a, b);
        if (parent_b_flexible)
          grad_coeffs[parent_b].get(g, t, s)
              += -log(1 + x) + digamma_diff(b, a);
      });
    case Distribution::fixed:
      return;
    default:
      throw std::runtime_error("Error: distribution not implemented.");
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

size_t Coefficient::size() const { return values.size(); }

STD::Vector Coefficient::vectorize() const {
  STD::Vector v(size());
  auto iter = begin(v);
  for (auto &x : values)
    *iter++ = x;
  return v;
}
