#include "coefficient.hpp"
#include "Experiment.hpp"
#include "aux.hpp"
#include "compression.hpp"
#include "design.hpp"
#include "entropy.hpp"
#include "gp.hpp"
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
                         const Parameters &params,
                         const std::vector<CoefficientPtr> &priors_)
    : Id(id), parameters(params), priors(priors_) {
  if (type == Type::gp_points and not spot_dependent())
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

  // TODO sample properly
  if (type != Type::fixed)
    for (auto &x : values)
      x = parameters.variance
          * std::normal_distribution<double>()(EntropySource::rng);

  parent_a_flexible = false;
  parent_b_flexible = false;
  if (priors.size() >= 2) {
    parent_a_flexible = priors[0]->type != Type::fixed;
    parent_b_flexible = priors[1]->type != Type::fixed;
  }

  LOG(debug) << "parent_a_flexible = " << parent_a_flexible;
  LOG(debug) << "parent_b_flexible = " << parent_b_flexible;

  LOG(debug) << *this;
}

Fixed::Fixed(size_t G, size_t T, size_t S, const Id &id,
             const Parameters &params)
    : Coefficient(G, T, S, id, params, {}) {}
Distributions::Distributions(size_t G, size_t T, size_t S, const Id &id,
                             const Parameters &params,
                             const std::vector<CoefficientPtr> &priors_)
    : Coefficient(G, T, S, id, params, priors_) {}
Gamma::Gamma(size_t G, size_t T, size_t S, const Id &id,
             const Parameters &params,
             const std::vector<CoefficientPtr> &priors_)
    : Distributions(G, T, S, id, params, priors_) {
  for (size_t i = 0; i < 2; ++i)
    if (priors[i]->type != Type::fixed and priors[i]->type != Type::gamma)
      throw std::runtime_error("Error: argument " + std::to_string(i) + " of "
                               + to_string() + " has invalid type "
                               + ::Coefficient::to_string(priors[i]->type));
}
Beta::Beta(size_t G, size_t T, size_t S, const Id &id, const Parameters &params,
           const std::vector<CoefficientPtr> &priors_)
    : Distributions(G, T, S, id, params, priors_) {
  for (size_t i = 0; i < 2; ++i)
    if (priors[i]->type != Type::fixed and priors[i]->type != Type::gamma)
      throw std::runtime_error("Error: argument " + std::to_string(i) + " of "
                               + to_string() + " has invalid type "
                               + ::Coefficient::to_string(priors[i]->type));
}
BetaPrime::BetaPrime(size_t G, size_t T, size_t S, const Id &id,
                     const Parameters &params,
                     const std::vector<CoefficientPtr> &priors_)
    : Distributions(G, T, S, id, params, priors_) {
  for (size_t i = 0; i < 2; ++i)
    if (priors[i]->type != Type::fixed and priors[i]->type != Type::gamma)
      throw std::runtime_error("Error: argument " + std::to_string(i) + " of "
                               + to_string() + " has invalid type "
                               + ::Coefficient::to_string(priors[i]->type));
}
Normal::Normal(size_t G, size_t T, size_t S, const Id &id,
               const Parameters &params,
               const std::vector<CoefficientPtr> &priors_)
    : Distributions(G, T, S, id, params, priors_) {
  if (priors[0]->type != Type::fixed and priors[0]->type != Type::normal
      and priors[0]->type != Type::gp_points)
    throw std::runtime_error("Error: argument " + std::to_string(0) + " of "
                             + to_string() + " has invalid type "
                             + ::Coefficient::to_string(priors[0]->type));
  if (priors[1]->type != Type::fixed and priors[1]->type != Type::gamma)
    throw std::runtime_error("Error: argument " + std::to_string(1) + " of "
                             + to_string() + " has invalid type "
                             + ::Coefficient::to_string(priors[1]->type));
}
namespace Spatial {
Coord::Coord(size_t G, size_t T, size_t S, const Id &id,
             const Parameters &params,
             const std::vector<CoefficientPtr> &priors_)
    : Coefficient(G, T, S, id, params, priors_),
      length_scale(priors[0]->get_actual(0, 0, 0)) {
  assert(not priors.empty());
  assert(priors[0]->type == Type::fixed);
  if (priors[0]->type != Type::fixed)
    throw std::runtime_error("Error: argument " + std::to_string(0) + " of "
                             + to_string() + " has invalid type "
                             + ::Coefficient::to_string(priors[0]->type));
  if (priors[1]->type != Type::fixed and priors[1]->type != Type::gamma)
    throw std::runtime_error("Error: argument " + std::to_string(1) + " of "
                             + to_string() + " has invalid type "
                             + ::Coefficient::to_string(priors[1]->type));
  if (priors[2]->type != Type::fixed and priors[2]->type != Type::gamma)
    throw std::runtime_error("Error: argument " + std::to_string(2) + " of "
                             + to_string() + " has invalid type "
                             + ::Coefficient::to_string(priors[2]->type));
}
Points::Points(size_t G, size_t T, size_t S, const Id &id,
               const Parameters &params,
               const std::vector<CoefficientPtr> &priors_)
    : Coefficient(G, T, S, id, params, priors_) {
  if (priors[0]->type != Type::fixed and priors[0]->type != Type::normal
      and priors[0]->type != Type::gp_points)
    throw std::runtime_error("Error: argument " + std::to_string(0) + " of "
                             + to_string() + " has invalid type "
                             + ::Coefficient::to_string(priors[0]->type));
}
}  // namespace Spatial

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

namespace Spatial {
double Coord::compute_gradient(CoefficientPtr grad_coeff) const {
  LOG(verbose) << "Computing Gaussian process gradient. length_scale = "
               << length_scale;

  assert(type == Type::gp_coord);
  if (grad_coeff->type != Type::gp_coord)
    throw std::runtime_error(
        "Error: mismatched argument type in Coord::compute_gradient().");

  Matrix formed_data = form_data();
  Matrix formed_mean = form_mean();

  Vector svs = Vector::Ones(formed_data.cols());
  Vector deltas = Vector::Ones(formed_data.cols());
  for (int t = 0; t < svs.size(); ++t) {
    svs(t) = priors[1]->get_actual(0, t, 0);
    deltas(t) = priors[2]->get_actual(0, t, 0);
  }

  Matrix mus = formed_data;
  mus.setZero();
  Matrix vars = mus;

  Vector grad_delta = Vector::Zero(values.size());
  auto grads = gp->predict_means_and_vars(formed_data, formed_mean, svs, deltas,
                                          mus, vars);

  Matrix formed_gradient
      = (mus - formed_data).array() / vars.array() / vars.array();

  dynamic_pointer_cast<Coord>(grad_coeff)->add_formed_data(formed_gradient);

  const size_t sv_idx = 1;
  if (priors[sv_idx]->type != Type::fixed)
    for (size_t t = 0; t < grads.size(); ++t)
      grad_coeff->priors[sv_idx]->get_raw(0, t, 0) += grads[t].sv;

  const size_t delta_idx = 2;
  if (priors[delta_idx]->type != Type::fixed)
    for (size_t t = 0; t < grads.size(); ++t)
      grad_coeff->priors[delta_idx]->get_raw(0, t, 0) += grads[t].delta;

  double score = 0;
  for (int i = 0; i < formed_data.rows(); ++i)
    for (int j = 0; j < formed_data.cols(); ++j)
      score += log_normal(formed_data(i, j), mus(i, j), vars(i, j));

  LOG(verbose) << "GP score: " << score;
  return score;
}
}  // namespace Spatial

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
      throw std::runtime_error(
          "Error: invalid Coefficient::Kind in get_raw().");
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
  switch (type) {
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

size_t Coefficient::number_variable() const {
  switch (type) {
    case Type::gp_coord:
    case Type::fixed:
      return 0;
    default:
      return size();
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

string to_string(const Type &type) {
  switch (type) {
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
    case Type::gp_points:
      return "gaussian_process_points";
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
  string s = "Coefficient '" + name + "', " + ::Coefficient::to_string(type)
             + "-distributed " + ::Coefficient::to_string(kind);
  s += ": " + std::to_string(values.rows()) + "x"
       + std::to_string(values.cols()) + " (" + std::to_string(values.size())
       + ")";
  // TODO FIXUP coeffs improve
  s += " num_priors=" + std::to_string(priors.size());
  size_t i = 0;
  for (auto &prior : priors)
    s += " prior" + std::to_string(i++) + "='" + prior->name + "'";
  return s;
}

namespace Spatial {

size_t Coord::size() const {
  int n = 0;
  for (auto pts : points)
    n += pts->values.rows();
  return n;
}

Matrix Coord::form_data() const {
  int ncol = 0;
  for (auto pts : points)
    if (pts->values.cols() > ncol)
      ncol = pts->values.cols();
  Matrix m = Matrix::Zero(size(), ncol);
  size_t row = 0;
  for (auto pts : points) {
    for (int i = 0; i < pts->values.rows(); ++i) {
      for (int j = 0; j < pts->values.cols(); ++j)
        m(row + i, j) = pts->values(i, j);
    }
    row += pts->values.rows();
  }
  return m;
}

Matrix Coord::form_mean() const {
  int ncol = 0;
  for (auto pts : points)
    if (pts->values.cols() > ncol)
      ncol = pts->values.cols();
  Matrix m = Matrix::Zero(size(), ncol);
  size_t row = 0;
  for (auto pts : points) {
    for (int s = 0; s < pts->values.rows(); ++s) {
      for (int t = 0; t < pts->values.cols(); ++t)
        m(row + s, t) = pts->priors[0]->get_actual(0, t, s);
    }
    row += pts->values.rows();
  }
  return m;
}

void Coord::add_formed_data(const Matrix &m) {
  if (type != Type::gp_coord)
    std::runtime_error(
        "Error: called add_formed_data() on a coefficient that is not a "
        "Gaussian process coordinate system.");
  int n = 0;
  for (auto pts : points)
    n += pts->values.rows();
  assert(m.rows() == n);
  size_t row = 0;
  for (auto pts : points) {
    for (int i = 0; i < pts->values.rows(); ++i) {
      for (int j = 0; j < pts->values.cols(); ++j)
        pts->values(i, j) += m(row + i, j);
    }
    if (pts->priors[0]->type != Type::fixed)
      for (int i = 0; i < pts->values.rows(); ++i) {
        for (int j = 0; j < pts->values.cols(); ++j)
          pts->values(i, j) += m(row + i, j);
      }

    row += pts->values.rows();
  }
}

void Coord::construct_gp() {
  size_t n = 0;
  for (auto &pts : points)
    n += pts->values.rows();
  size_t ncol = points.front()->values.cols();
  LOG(debug) << "n = " << n;
  Matrix m = Matrix::Zero(n, ncol);
  size_t i = 0;
  for (auto &pts : points) {
    for (int s = 0; s < pts->values.rows(); ++s)
      for (int j = 0; j < pts->values.cols(); ++j)
        m(i + s, j) = pts->values(s, j);
    i += pts->values.rows();
  }
  LOG(debug) << "coordinate dimensions = " << m.rows() << "x" << m.cols();
  gp = std::make_shared<GP::GaussianProcess>(
      GP::GaussianProcess(m, length_scale));
}
}  // namespace Spatial

ostream &operator<<(ostream &os, const Coefficient &coeff) {
  os << coeff.to_string();
  return os;
}

CoefficientPtr make_shared(size_t G, size_t T, size_t S, const Id &cid,
                           const Parameters &params,
                           const std::vector<CoefficientPtr> &priors) {
  switch (cid.type) {
    case Type::fixed:
      return std::make_shared<Fixed>(G, T, S, cid, params);
    case Type::normal:
      return std::make_shared<Normal>(G, T, S, cid, params, priors);
    case Type::beta:
      return std::make_shared<Beta>(G, T, S, cid, params, priors);
    case Type::beta_prime:
      return std::make_shared<BetaPrime>(G, T, S, cid, params, priors);
    case Type::gamma:
      return std::make_shared<Gamma>(G, T, S, cid, params, priors);
    case Type::gp_coord: {
      assert(priors.size() == 4);
      vector<CoefficientPtr> priors_;
      priors_.emplace_back(priors[0]);
      priors_.emplace_back(priors[2]);
      priors_.emplace_back(priors[3]);
      return std::make_shared<Spatial::Coord>(G, T, S, cid, params, priors_);
    }
    case Type::gp_points: {
      assert(priors.size() == 4);
      vector<CoefficientPtr> priors_;
      priors_.emplace_back(priors[1]);
      return std::make_shared<Spatial::Points>(G, T, S, cid, params, priors_);
    }
    default:
      throw std::runtime_error(
          "Error: invalid Coefficient::Type in make_shared().");
  }
}
}  // namespace Coefficient
