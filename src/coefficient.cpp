#include "coefficient.hpp"
#include "Experiment.hpp"
#include "aux.hpp"
#include "compression.hpp"
#include "design.hpp"
#include "entropy.hpp"
#include "io.hpp"
#include "pdist.hpp"

namespace Coefficient {

using namespace std;
using STD::Matrix;
using STD::Vector;

Kind determine_kind(const set<string> &term) {
  using namespace Design;
  static map<string, Kind> id2kind{
      {gene_label, Kind::gene},
      {spot_label, Kind::spot},
      {type_label, Kind::type},
  };
  Kind kind = Kind::scalar;
  for (auto &k : id2kind) {
    if (term.find(k.first) != term.end())
      kind = kind | k.second;
  }
  return kind;
}

Coefficient::Coefficient(size_t G, size_t T, size_t S, const Id &id,
                         const Parameters &params)
    : Id(id),
      parameters(params) {
  if (distribution == Type::gp and not spot_dependent())
    throw std::runtime_error(
        "Error: Gaussian processes only allowed for spot-dependent or "
        "spot- and type-dependent coefficients.");
  // TODO cov prior fill prior_idx
  switch (kind) {
    case Kind::scalar:
      values = Matrix::Zero(1, 1);
      break;
    case Kind::gene:
      values = Matrix::Zero(G, 1);
      break;
    case Kind::spot:
      values = Matrix::Zero(S, 1);
      break;
    case Kind::type:
      values = Matrix::Zero(T, 1);
      break;
    case Kind::gene_type:
      values = Matrix::Zero(G, T);
      break;
    case Kind::spot_type:
      values = Matrix::Zero(S, T);
      break;
    default:
      throw std::runtime_error(
          "Error: invalid Coefficient::Kind in Coefficient::Coefficient().");
  }

  if (distribution != Type::fixed)
    for (auto &x : values)
      x = parameters.variance
          * std::normal_distribution<double>()(EntropySource::rng);

  // TODO FIXUP coeffs
  // parent_a_flexible = grad_coeff->priors[0]->distribution != Type::fixed;
  // parent_b_flexible = grad_coeff->priors[1]->distribution != Type::fixed;
  parent_a_flexible = false;
  parent_b_flexible = false;

  LOG(debug) << "parent_a_flexible = " << parent_a_flexible;
  LOG(debug) << "parent_b_flexible = " << parent_b_flexible;

  LOG(debug) << *this;
}

Fixed::Fixed(size_t G, size_t T, size_t S, const Id &id,
             const Parameters &params)
    : Coefficient(G, T, S, id, params) {}
Distributions::Distributions(size_t G, size_t T, size_t S, const Id &id,
                             const Parameters &params)
    : Coefficient(G, T, S, id, params) {}
Gamma::Gamma(size_t G, size_t T, size_t S, const Id &id,
             const Parameters &params)
    : Distributions(G, T, S, id, params) {}
Beta::Beta(size_t G, size_t T, size_t S, const Id &id, const Parameters &params)
    : Distributions(G, T, S, id, params) {}
BetaPrime::BetaPrime(size_t G, size_t T, size_t S, const Id &id,
                     const Parameters &params)
    : Distributions(G, T, S, id, params) {}
Normal::Normal(size_t G, size_t T, size_t S, const Id &id,
               const Parameters &params)
    : Distributions(G, T, S, id, params) {}
namespace GP {
GP::GP(size_t G, size_t T, size_t S, const Id &id, const Parameters &params)
    : Coefficient(G, T, S, id, params) {}
Coord::Coord(size_t G, size_t T, size_t S, const Id &id,
             const Parameters &params)
    : Coefficient(G, T, S, id, params) {}
Points::Points(size_t G, size_t T, size_t S, const Id &id,
               const Parameters &params)
    : Coefficient(G, T, S, id, params) {}
}  // namespace GP

/** Calculates gradient with respect to the "natural" representation
 *
 * The natural representation always covers the entire real line
 * for normal x -> x
 * for gamma and beta prime x -> exp(x)
 * for beta x -> exp(x) / (exp(x) + 1)
 */

double Gamma::compute_gradient(CoefficientPtr grad_coeff) const {
  LOG(debug) << "Coefficient::Gamma::compute_gradient: " << *this;
  return visit([&](size_t g, size_t t, size_t s) {
    double a = priors[0]->get_actual(g, t, s);
    double b = priors[1]->get_actual(g, t, s);
    double x = get_actual(g, t, s);

    grad_coeff->get_raw(g, t, s) += (a - 1) - x * b;

    if (parent_a_flexible)
      grad_coeff->priors[0]->get_raw(g, t, s)
          += a * (log(b) - digamma(a) + log(x));
    if (parent_b_flexible)
      grad_coeff->priors[1]->get_raw(g, t, s) += a - b * x;

    return log_gamma_rate(x, a, b);
  });
}

double Beta::compute_gradient(CoefficientPtr grad_coeff) const {
  LOG(debug) << "Coefficient::Beta::compute_gradient: " << *this;
  return visit([&](size_t g, size_t t, size_t s) {
    double a = priors[0]->get_actual(g, t, s);
    double b = priors[1]->get_actual(g, t, s);
    double p = get_actual(g, t, s);

    grad_coeff->get_raw(g, t, s) += (a - 1) * (1 - p) - (b - 1) * p;

    if (parent_a_flexible)
      grad_coeff->priors[0]->get_raw(g, t, s)
          += a * (log(p) + digamma_diff(a, b));
    if (parent_b_flexible)
      grad_coeff->priors[1]->get_raw(g, t, s)
          += b * (log(1 - p) + digamma_diff(b, a));

    return log_beta(p, a, b);
  });
}

double BetaPrime::compute_gradient(CoefficientPtr grad_coeff) const {
  LOG(debug) << "Coefficient::BetaPrime::compute_gradient: " << *this;
  return visit([&](size_t g, size_t t, size_t s) {
    double a = priors[0]->get_actual(g, t, s);
    double b = priors[1]->get_actual(g, t, s);
    double log_odds = get_raw(g, t, s);
    double odds = exp(log_odds);
    double p = odds_to_prob(odds);

    grad_coeff->get_raw(g, t, s) += (a - 1) * (1 - p) - (b + 1) * p;

    if (parent_a_flexible)
      grad_coeff->priors[0]->get_raw(g, t, s)
          += a * (log(p) + digamma_diff(a, b));
    if (parent_b_flexible)
      grad_coeff->priors[1]->get_raw(g, t, s)
          += b * (log(1 - p) + digamma_diff(b, a));

    return log_beta(p, a, b);
  });
}

double Normal::compute_gradient(CoefficientPtr grad_coeff) const {
  LOG(debug) << "Coefficient::Normal::compute_gradient: " << *this;
  return visit([&](size_t g, size_t t, size_t s) {
    double mu = priors[0]->get_actual(g, t, s);
    double sigma = priors[1]->get_actual(g, t, s);
    double x = get_raw(g, t, s);

    grad_coeff->get_raw(g, t, s) += (mu - x) / (sigma * sigma);

    if (parent_a_flexible)
      grad_coeff->priors[0]->get_raw(g, t, s) += (x - mu) / (sigma * sigma);
    if (parent_b_flexible)
      grad_coeff->priors[1]->get_raw(g, t, s)
          += (x - mu - sigma) * (x - mu + sigma) / (sigma * sigma);

    return log_normal(x, mu, sigma);
  });
}

namespace GP{

double GP::compute_gradient(CoefficientPtr grad_coeff) const {
  LOG(verbose) << "Computing log Gaussian process gradient.";

  assert(distribution == Type::gp_proxy);

  ::GP::MeanTreatment mean_treatment = ::GP::MeanTreatment::zero;

  vector<double> deltas(values.size());
  for (size_t t = 0; t < deltas.size(); ++t)
    deltas[t] = exp(values(t));

  vector<const ::GP::GaussianProcess *> gps;
  /* TODO FIXUP coeffs
  for (auto prior : priors)
    gps.push_back(prior->gp.get());
  */

  vector<Matrix> formed_data;
  /* TODO FIXUP coeffs
  for (auto prior : priors)
    formed_data.push_back(prior->form_data());
  */

  vector<Matrix> mus = formed_data;
  for (auto &m : mus)
    m.setZero();

  vector<Matrix> vars = mus;

  Vector grad_delta = Vector::Zero(values.size());
  auto sv = predict_means_and_vars(gps, formed_data, deltas, mean_treatment,
                                   mus, vars, grad_delta);

  grad_coeff->values.array() += grad_delta.array();
  LOG(debug) << "    DELTA =        " << values.transpose();
  LOG(debug) << "GRADDELTA =        " << grad_delta.transpose();

  LOG(debug) << "spatial variance = " << sv.transpose();
  for (size_t idx = 0; idx < priors.size(); ++idx)
    if (sv[idx] > 0) {
      Matrix formed_gradient = (mus[idx] - formed_data[idx]).array()
                               / vars[idx].array() / vars[idx].array();

      /* TODO FIXUP coeffs
      grad_coeff->priors[idx]->add_formed_data(formed_gradient);
      */
    }

  double score = 0;
  for (size_t k = 0; k < formed_data.size(); ++k)
    if (sv[k] > 0)
      for (int i = 0; i < formed_data[k].rows(); ++i)
        for (int j = 0; j < formed_data[k].cols(); ++j)
          score
              += log_normal(formed_data[k](i, j), mus[k](i, j), vars[k](i, j));
  return score;
}
}

bool kind_included(Kind kind, Kind x) { return (kind & x) == x; }

bool Coefficient::gene_dependent() const {
  return kind_included(kind, Kind::gene);
}

bool Coefficient::spot_dependent() const {
  return kind_included(kind, Kind::spot);
}

bool Coefficient::type_dependent() const {
  return kind_included(kind, Kind::type);
}

void Coefficient::store(const string &path, CompressionMode compression_mode,
                        const vector<string> &gene_names,
                        const vector<string> &spot_names,
                        const vector<string> &type_names,
                        vector<size_t> type_order) const {
  switch (kind) {
    case Kind::scalar:
      write_matrix(values, path, compression_mode, {"scalar"}, {to_string()});
      break;
    case Kind::gene:
      write_matrix(values, path, compression_mode, gene_names, {to_string()});
      break;
    case Kind::spot:
      write_matrix(values, path, compression_mode, spot_names, {to_string()});
      break;
    case Kind::type:
      // TODO cov type order
      write_matrix(values, path, compression_mode, type_names, {to_string()},
                   {}, type_order);
      break;
    case Kind::gene_type:
      write_matrix(values, path, compression_mode, gene_names, type_names,
                   type_order);
      break;
    case Kind::spot_type:
      write_matrix(values, path, compression_mode, spot_names, type_names,
                   type_order);
      break;
  }
}

void Coefficient::restore(const string &path) {
  values = parse_file<Matrix>(path, read_matrix, "\t");
}

double &Coefficient::get_raw(size_t g, size_t t, size_t s) {
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
      throw std::runtime_error("Error: invalid Coefficient::Kind in get_raw().");
  }
}

double Coefficient::get_raw(size_t g, size_t t, size_t s) const {
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
      throw std::runtime_error(
          "Error: invalid Coefficient::Kind in get_raw() const.");
  }
}

double sigmoid(double x) { return 1 / (1 + exp(-x)); }

double Coefficient::get_actual(size_t g, size_t t, size_t s) const {
  double x = get_raw(g, t, s);
  switch (distribution) {
    case Type::beta:
      return sigmoid(x);
    case Type::beta_prime:
    case Type::gamma:
      return exp(x);
    default:
      return x;
  }
}

size_t Coefficient::size() const { return values.size(); }

size_t Coefficient::number_parameters() const {
  switch (distribution) {
    case Type::gp_coord:
    case Type::fixed:
      return 0;
    // case Type::fixed:
    // return 1;
    case Type::beta:
    case Type::beta_prime:
    case Type::gamma:
    case Type::normal:
      return 2;
    case Type::gp:
      return 3;
    default:
      throw std::logic_error("Not implemented.");
  }
}

Vector Coefficient::vectorize() const {
  Vector v(size());
  auto iter = begin(v);
  for (auto &x : values)
    *iter++ = x;
  return v;
}

string to_string(const Kind &kind) {
  if (kind == Kind::scalar) {
    return "scalar";
  }

  static vector<pair<Kind, string>> kinds = {
      {Kind::spot, Design::spot_label},
      {Kind::gene, Design::gene_label},
      {Kind::type, Design::type_label},
  };
  static auto all_kinds = accumulate(
      kinds.begin(), kinds.end(), static_cast<Kind>(0),
      [](Kind a, const pair<Kind, string> &x) { return a | x.first; });

  if ((kind & ~all_kinds) != static_cast<Kind>(0)) {
    stringstream ss;
    ss << "Error: encountered unknown kind " << static_cast<int>(kind)
       << " in to_string().";
    throw runtime_error(ss.str());
  }

  vector<string> dependence;
  for (auto &x : kinds) {
    if ((kind & x.first) != static_cast<Kind>(0)) {
      dependence.push_back(x.second + "-");
    }
  }

  return intercalate<vector<string>::iterator, string>(dependence.begin(),
                                                       dependence.end(), ", ")
         + "dependent";
}

string to_string(const Type &distribution) {
  switch (distribution) {
    case Type::fixed:
      return "fixed";
    case Type::gamma:
      return "gamma";
    case Type::beta:
      return "beta";
    case Type::beta_prime:
      return "beta'";
    case Type::normal:
      return "normal";
    case Type::gp:
      return "gaussian_process";
    case Type::gp_proxy:
      return "gaussian_process_proxy";
    case Type::gp_coord:
      return "gaussian_process_coord";
    default:
      throw std::runtime_error("Error: invalid Coefficient::Distribution.");
  }
}

string to_token(const Kind &kind) {
  switch (kind) {
    case Kind::scalar:
      return "scalar";
    case Kind::gene:
      return "gene";
    case Kind::spot:
      return "spot";
    case Kind::type:
      return "type";
    case Kind::gene_type:
      return "gene-type";
    case Kind::spot_type:
      return "spot-type";
    default:
      throw std::runtime_error(
          "Error: invalid Coefficient::Kind in to_token().");
  }
}

string storage_type(Kind kind) {
  switch (kind) {
    case Kind::scalar:
      return "scalar";
    case Kind::gene:
    case Kind::type:
    case Kind::spot:
      return "vector";
    case Kind::gene_type:
    case Kind::spot_type:
      return "matrix";
    default:
      throw std::runtime_error(
          "Error: invalid Coefficient::Kind in storage_type().");
  }
}

string Coefficient::to_string() const {
  string s = "Coefficient '" + name + "', " + ::Coefficient::to_string(distribution)
             + "-distributed " + ::Coefficient::to_string(kind);
  s += ": " + std::to_string(values.rows()) + "x"
       + std::to_string(values.cols()) + " (" + std::to_string(values.size())
       + ")";
  // TODO FIXUP coeffs improve
  s += " num_priors=" + std::to_string(priors.size());
  return s;
}

namespace GP {

Matrix Coord::form_data() const {
  if (distribution != Type::gp_coord)
    std::runtime_error(
        "Error: called form_data() on a coefficient that is not a Gaussian "
        "process coordinate system.");
  int n = 0;
  for (auto prior : priors)
    n += prior->values.rows();
  int ncol = 0;
  for (auto prior : priors)
    if (prior->values.cols() > ncol)
      ncol = prior->values.cols();
  Matrix m = Matrix::Zero(n, ncol);
  size_t row = 0;
  for (auto prior : priors) {
    for (int i = 0; i < prior->values.rows(); ++i) {
      for (int j = 0; j < prior->values.cols(); ++j)
        m(row + i, j) = prior->values(i, j);
    }
    row += prior->values.rows();
  }
  return m;
}

void Coord::add_formed_data(const Matrix &m) const {
  if (distribution != Type::gp_coord)
    std::runtime_error(
        "Error: called add_formed_data() on a coefficient that is not a "
        "Gaussian process coordinate system.");
  int n = 0;
  for (auto prior : priors)
    n += prior->values.rows();
  assert(m.rows() == n);
  size_t row = 0;
  for (auto prior : priors) {
    for (int i = 0; i < prior->values.rows(); ++i) {
      for (int j = 0; j < prior->values.cols(); ++j)
        prior->values(i, j) += m(row + i, j);
    }
    row += prior->values.rows();
  }
}
}

ostream &operator<<(ostream &os, const Coefficient &coeff) {
  os << coeff.to_string();
  return os;
}

CoefficientPtr make_shared(size_t G, size_t T, size_t S, const Id &cid,
                           const Parameters &params) {
  switch (cid.distribution) {
    case Type::fixed:
      return std::make_shared<Fixed>(G, T, S, cid, params);
    case Type::normal:
      return std::make_shared<Normal>(G, T, S, cid, params);
    case Type::beta:
      return std::make_shared<Beta>(G, T, S, cid, params);
    case Type::beta_prime:
      return std::make_shared<BetaPrime>(G, T, S, cid, params);
    case Type::gamma:
      return std::make_shared<Gamma>(G, T, S, cid, params);
    case Type::gp_proxy:
      return std::make_shared<GP::GP>(G, T, S, cid, params);
    case Type::gp_coord:
      return std::make_shared<GP::Coord>(G, T, S, cid, params);
    case Type::gp:
      return std::make_shared<GP::Points>(G, T, S, cid, params);
    default:
      throw std::runtime_error(
          "Error: invalid Coefficient::Type in make_shared().");
  }
}
}  // namespace Coefficient
