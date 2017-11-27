#include "coefficient.hpp"
#include "Experiment.hpp"
#include "aux.hpp"
#include "compression.hpp"
#include "design.hpp"
#include "entropy.hpp"
#include "io.hpp"
#include "pdist.hpp"

using namespace std;
using STD::Vector;
using STD::Matrix;

Coefficient::Kind determine_kind(const set<string> &term) {
  using namespace Design;
  static map<string, Coefficient::Kind> id2kind{
    { gene_label, Coefficient::Kind::gene },
    { spot_label, Coefficient::Kind::spot },
    { type_label, Coefficient::Kind::type },
  };
  Coefficient::Kind kind = Coefficient::Kind::scalar;
  for (auto& k : id2kind) {
    if (term.find(k.first) != term.end())
      kind = kind | k.second;
  }
  return kind;
}

Coefficient::Coefficient(size_t G, size_t T, size_t S, const Id& id)
    : Coefficient(G, T, S, id.name, id.kind, id.dist, id.info)
{
}

Coefficient::Coefficient(size_t G, size_t T, size_t S, const string &label_,
                         Kind kind_, Distribution dist,
                         CovariateInformation info_)
    : label(label_),
      kind(kind_),
      distribution(dist),
      info(info_) {
  if (distribution == Distribution::gp and not spot_dependent())
    throw std::runtime_error(
        "Error: Gaussian processes only allowed for spot-dependent or "
        "spot- and type-dependent coefficients.");
  // TODO cov prior fill prior_idx
  switch (kind) {
    case Kind::scalar:
      values = Matrix::Ones(1, 1);
      break;
    case Kind::gene:
      values = Matrix::Ones(G, 1);
      break;
    case Kind::spot:
      values = Matrix::Ones(S, 1);
      break;
    case Kind::type:
      values = Matrix::Ones(T, 1);
      break;
    case Kind::gene_type:
      values = Matrix::Ones(G, T);
      break;
    case Kind::spot_type:
      values = Matrix::Ones(S, T);
      break;
  }

  if (kind == Coefficient::Kind::gene_type
      or kind == Coefficient::Kind::spot_type)
    for (auto &x : values)
      x = exp(0.1 * std::normal_distribution<double>()(EntropySource::rng));

  LOG(verbose) << *this;
}

/** Calculates gradient with respect to the "natural" representation
 *
 * The natural representation always covers the entire real line
 * for normal x -> x
 * for gamma and beta prime x -> exp(x)
 * for beta x -> exp(x) / (exp(x) + 1)
 */
double Coefficient::compute_gradient(const vector<CoefficientPtr> &coeffs,
                                     vector<CoefficientPtr> &grad_coeffs,
                                     size_t coeff_idx) const {
  LOG(debug) << "Coefficient::compute_gradient " << coeff_idx << ":" << *this;
  if (distribution == Distribution::fixed
      or distribution == Distribution::gp
      or distribution == Distribution::gp_coord)
    return 0;
  if (distribution == Distribution::gp_proxy)
    return compute_gradient_gp(coeffs, grad_coeffs, coeff_idx);
  assert(prior_idxs.size() >= 2);
  size_t parent_a = prior_idxs[0];
  size_t parent_b = prior_idxs[1];
  bool parent_a_flexible
      = grad_coeffs[parent_a]->distribution != Distribution::fixed;
  bool parent_b_flexible
      = grad_coeffs[parent_b]->distribution != Distribution::fixed;
  LOG(debug) << "parent_a_flexible = " << parent_a_flexible;
  LOG(debug) << "parent_b_flexible = " << parent_b_flexible;
  switch (distribution) {
    case Distribution::gamma:
      LOG(debug) << "Computing gamma distribution gradient.";
      return visit([&](size_t g, size_t t, size_t s) {
        double a = coeffs[parent_a]->get_actual(g, t, s);
        double b = coeffs[parent_b]->get_actual(g, t, s);
        double x = get_actual(g, t, s);

        grad_coeffs[coeff_idx]->get_raw(g, t, s) += (a - 1) - x * b;

        if (parent_a_flexible)
          grad_coeffs[parent_a]->get_raw(g, t, s)
              += a * (log(b) - digamma(a) + log(x));
        if (parent_b_flexible)
          grad_coeffs[parent_b]->get_raw(g, t, s) += a - b * x;

        return log_gamma_rate(x, a, b);
      });
    case Distribution::beta:
      LOG(debug) << "Computing beta distribution gradient.";
      return visit([&](size_t g, size_t t, size_t s) {
        double a = coeffs[parent_a]->get_actual(g, t, s);
        double b = coeffs[parent_b]->get_actual(g, t, s);
        double p = get_actual(g, t, s);

        grad_coeffs[coeff_idx]->get_raw(g, t, s)
            += (a - 1) * (1 - p) - (b - 1) * p;

        if (parent_a_flexible)
          grad_coeffs[parent_a]->get_raw(g, t, s)
              += a * (log(p) + digamma_diff(a, b));
        if (parent_b_flexible)
          grad_coeffs[parent_b]->get_raw(g, t, s)
              += b * (log(1 - p) + digamma_diff(b, a));

        return log_beta(p, a, b);
      });
    case Distribution::beta_prime:
      LOG(debug) << "Computing beta' distribution gradient.";
      return visit([&](size_t g, size_t t, size_t s) {
        double a = coeffs[parent_a]->get_actual(g, t, s);
        double b = coeffs[parent_b]->get_actual(g, t, s);
        double log_odds = get_raw(g, t, s);
        double x = exp(log_odds);
        double p = odds_to_prob(x);

        grad_coeffs[coeff_idx]->get_raw(g, t, s)
            += (a - 1) * (1 - p) - (b + 1) * p;

        if (parent_a_flexible)
          grad_coeffs[parent_a]->get_raw(g, t, s)
              += a * (log(p) + digamma_diff(a, b));
        if (parent_b_flexible)
          grad_coeffs[parent_b]->get_raw(g, t, s)
              += b * (log(1 - p) + digamma_diff(b, a));

        return log_beta(p, a, b);
      });
    case Distribution::normal:
      LOG(debug) << "Computing log normal distribution gradient.";
      return visit([&](size_t g, size_t t, size_t s) {
        double mu = coeffs[parent_a]->get_actual(g, t, s);
        double sigma = coeffs[parent_b]->get_actual(g, t, s);
        double x = get_raw(g, t, s);

        grad_coeffs[coeff_idx]->get_raw(g, t, s) += (mu - x) / (sigma * sigma);

        if (parent_a_flexible)
          grad_coeffs[parent_a]->get_raw(g, t, s) += (x - mu) / (sigma * sigma);
        if (parent_b_flexible)
          grad_coeffs[parent_b]->get_raw(g, t, s)
              += (x - mu - sigma) * (x - mu + sigma) / (sigma * sigma);

        return log_normal(x, mu, sigma);
      });
    default:
      throw std::runtime_error(
          "Error: coefficient gradient not implemented for distribution '"
          + ::to_string(distribution) + "'.");
  }
}

double Coefficient::compute_gradient_gp(const vector<CoefficientPtr> &coeffs,
                                        vector<CoefficientPtr> &grad_coeffs,
                                        size_t coeff_idx) const {
  LOG(verbose) << "Computing log Gaussian process gradient.";

  GP::MeanTreatment mean_treatment = GP::MeanTreatment::zero;

  vector<double> deltas(values.size());
  for (size_t t = 0; t < deltas.size(); ++t)
    deltas[t] = values(t);

  vector<const GP::GaussianProcess *> gps;
  for (auto idx : prior_idxs)
    gps.push_back(coeffs[idx]->gp.get());

  vector<Matrix> formed_data;
  for (auto idx : prior_idxs)
    formed_data.push_back(coeffs[idx]->form_data(coeffs));

  vector<Matrix> mus = formed_data;
  for (auto &m : mus)
    m.setZero();

  vector<Matrix> vars = mus;

  Vector grad_delta = Vector::Zero(values.size());
  auto sv = predict_means_and_vars(gps, formed_data, deltas, mean_treatment,
                                   mus, vars, grad_delta);

  grad_coeffs[coeff_idx]->values.array() += grad_delta.array();
  LOG(debug) << "    DELTA =        " << coeffs[coeff_idx]->values.transpose();
  LOG(debug) << "GRADDELTA =        " << grad_delta.transpose();

  LOG(debug) << "spatial variance = " << sv.transpose();
  for (size_t idx = 0; idx < prior_idxs.size(); ++idx)
    if (sv[idx] > 0) {
      Matrix formed_gradient = (mus[idx] - formed_data[idx]).array()
                               / vars[idx].array() / vars[idx].array();

      grad_coeffs[prior_idxs[idx]]->add_formed_data(formed_gradient,
                                                   grad_coeffs);
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

double sigmoid(double x) {
  return 1 / (1 + exp(-x));
}

double Coefficient::get_actual(size_t g, size_t t, size_t s) const {
  double x = get_raw(g,t,s);
  switch (distribution) {
    case Distribution::beta:
      return sigmoid(x);
    case Distribution::beta_prime:
    case Distribution::gamma:
      return exp(x);
    default:
      return x;
  }
}


size_t Coefficient::size() const { return values.size(); }

size_t Coefficient::number_parameters() const {
  switch (distribution) {
    case Distribution::fixed:
    case Distribution::gp_coord:
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

string to_string(const Coefficient::Kind &kind) {
  if (kind == Coefficient::Kind::scalar) {
    return "scalar";
  }

  static vector<pair<Coefficient::Kind, string>> kinds = {
      {Coefficient::Kind::spot, Design::spot_label},
      {Coefficient::Kind::gene, Design::gene_label},
      {Coefficient::Kind::type, Design::type_label},
  };
  static auto all_kinds = accumulate(
      kinds.begin(), kinds.end(), static_cast<Coefficient::Kind>(0),
      [](Coefficient::Kind a, const pair<Coefficient::Kind, string> &x) {
        return a | x.first;
      });

  if ((kind & ~all_kinds) != static_cast<Coefficient::Kind>(0)) {
    stringstream ss;
    ss << "Error: encountered unknown kind " << static_cast<int>(kind)
       << " in to_string().";
    throw runtime_error(ss.str());
  }

  vector<string> dependence;
  for (auto &x : kinds) {
    if ((kind & x.first) != static_cast<Coefficient::Kind>(0)) {
      dependence.push_back(x.second + "-");
    }
  }

  return intercalate<vector<string>::iterator, string>(
      dependence.begin(), dependence.end(), ", ") + "dependent";
}

string to_string(const Coefficient::Distribution &distribution) {
  switch (distribution) {
    case Coefficient::Distribution::fixed:
      return "fixed";
    case Coefficient::Distribution::gamma:
      return "gamma";
    case Coefficient::Distribution::beta:
      return "beta";
    case Coefficient::Distribution::beta_prime:
      return "beta'";
    case Coefficient::Distribution::normal:
      return "normal";
    case Coefficient::Distribution::gp:
      return "gaussian_process";
    case Coefficient::Distribution::gp_proxy:
      return "gaussian_process_proxy";
    case Coefficient::Distribution::gp_coord:
      return "gaussian_process_coord";
    default:
      throw std::runtime_error("Error: invalid Coefficient::Distribution.");
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
      throw std::runtime_error(
          "Error: invalid Coefficient::Kind in to_token().");
  }
}

string storage_type(Coefficient::Kind kind) {
  using Kind = Coefficient::Kind;
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
      throw std::runtime_error("Error: invalid Coefficient::Kind in storage_type().");
  }
}


string Coefficient::to_string() const {
  string s = "Coefficient '" + label + "', "
             + ::to_string(distribution) + "-distributed " + ::to_string(kind);
  s += ": " + std::to_string(values.rows()) + "x"
       + std::to_string(values.cols()) + " (" + std::to_string(values.size())
       + ")";
  for (auto &prior_idx : prior_idxs)
    s += " prior=" + std::to_string(prior_idx);
  return s;
}

Matrix Coefficient::form_data(const vector<CoefficientPtr> &coeffs) const {
  if (distribution != Distribution::gp_coord)
    std::runtime_error(
        "Error: called for_data() on a coefficient that is not a Gaussian "
        "process coordinate system.");
  int n = 0;
  for (auto idx : prior_idxs)
    n += coeffs[idx]->values.rows();
  int ncol = 0;
  for (auto idx : prior_idxs)
    if (coeffs[idx]->values.cols() > ncol)
      ncol = coeffs[idx]->values.cols();
  Matrix m = Matrix::Zero(n, ncol);
  size_t row = 0;
  for (auto idx : prior_idxs) {
    for (int i = 0; i < coeffs[idx]->values.rows(); ++i) {
      for (int j = 0; j < coeffs[idx]->values.cols(); ++j)
        m(row + i, j) = coeffs[idx]->values(i, j);
    }
    row += coeffs[idx]->values.rows();
  }
  return m;
}

void Coefficient::add_formed_data(const Matrix &m,
                                  vector<CoefficientPtr> &coeffs) const {
  if (distribution != Distribution::gp_coord)
    std::runtime_error(
        "Error: called add_formed_data() on a coefficient that is not a "
        "Gaussian process coordinate system.");
  int n = 0;
  for (auto idx : prior_idxs)
    n += coeffs[idx]->values.rows();
  assert(m.rows() == n);
  size_t row = 0;
  for (auto idx : prior_idxs) {
    for (int i = 0; i < coeffs[idx]->values.rows(); ++i) {
      for (int j = 0; j < coeffs[idx]->values.cols(); ++j)
        coeffs[idx]->values(i, j) += m(row + i, j);
    }
    row += coeffs[idx]->values.rows();
  }
}

ostream &operator<<(ostream &os, const Coefficient &coeff) {
  os << coeff.to_string();
  return os;
}
