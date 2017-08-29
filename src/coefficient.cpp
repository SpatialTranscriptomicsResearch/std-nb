#include "coefficient.hpp"
#include "aux.hpp"
#include "compression.hpp"
#include "design.hpp"
#include "entropy.hpp"
#include "io.hpp"
#include "pdist.hpp"

using namespace std;
using STD::Matrix;

Coefficient::Kind determine_kind(const vector<string> &term) {
  Coefficient::Kind kind = Coefficient::Kind::scalar;
  for (auto &covariate_label : term) {
    string label = to_lower(covariate_label);
    if (label == "gene")
      kind = kind | Coefficient::Kind::gene;
    else if (label == "spot")
      kind = kind | Coefficient::Kind::spot;
    else if (label == "type")
      kind = kind | Coefficient::Kind::type;
  }
  return kind;
}

std::string to_string(DistributionMode mode) {
  switch (mode) {
    case DistributionMode::log_normal:
      return "log_normal";
    case DistributionMode::gamma_odds:
      return "gamma_odds";
    case DistributionMode::gamma_odds_log_normal:
      return "gamma_odds_log_normal";
  }
  throw std::runtime_error("Error in to_string(DistributionMode).");
}

DistributionMode distribution_from_string(const std::string &s_) {
  string s = to_lower(s_);
  if (s == "log_normal")
    return DistributionMode::log_normal;
  else if (s == "gamma_odds")
    return DistributionMode::gamma_odds;
  else if (s == "gamma_odds_log_normal")
    return DistributionMode::gamma_odds_log_normal;
  else
    throw std::runtime_error("Couldn't parse DistributionMode: '" + s + "'.");
}

std::ostream &operator<<(std::ostream &os, DistributionMode mode) {
  os << to_string(mode);
  return os;
}

std::istream &operator>>(std::istream &is, DistributionMode &mode) {
  string token;
  is >> token;
  mode = distribution_from_string(token);
  return is;
}

Coefficient::Distribution choose_distribution(Coefficient::Variable variable,
                                              Coefficient::Kind kind,
                                              DistributionMode mode,
                                              bool use_gp) {
  if (use_gp and variable == Coefficient::Variable::rate
      and (kind == Coefficient::Kind::spot_type
           or kind == Coefficient::Kind::spot))
    return Coefficient::Distribution::log_gp;
  switch (mode) {
    case DistributionMode::log_normal:
      return Coefficient::Distribution::log_normal;
    case DistributionMode::gamma_odds:
    case DistributionMode::gamma_odds_log_normal:
      switch (variable) {
        case Coefficient::Variable::rate:
          return Coefficient::Distribution::gamma;
        case Coefficient::Variable::odds:
          return Coefficient::Distribution::beta_prime;
        case Coefficient::Variable::prior:
          if (mode == DistributionMode::gamma_odds)
            return Coefficient::Distribution::gamma;
          else if (mode == DistributionMode::gamma_odds_log_normal)
            return Coefficient::Distribution::log_normal;
          else
            throw std::runtime_error("Error: this should not happen!");
      }
  }
  throw std::runtime_error("Error in choose_distribution().");
}

Coefficient::Coefficient(size_t G, size_t T, size_t S, Variable variable_,
                         Kind kind_, Distribution dist,
                         std::shared_ptr<GP::GaussianProcess> gp_,
                         CovariateInformation info_)
    : variable(variable_),
      kind(kind_),
      distribution(dist),
      gp(gp_),
      info(info_) {
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

double Coefficient::compute_gradient(const vector<Coefficient> &coeffs,
                                     vector<Coefficient> &grad_coeffs,
                                     size_t idx) const {
  LOG(debug) << "Coefficient::compute_gradient " << idx << ":" << *this;
  if (distribution == Distribution::fixed or prior_idxs.size() < 2)
    return 0;
  size_t parent_a = prior_idxs[0];
  size_t parent_b = prior_idxs[1];
  bool parent_a_flexible
      = grad_coeffs[parent_a].distribution != Distribution::fixed;
  bool parent_b_flexible
      = grad_coeffs[parent_b].distribution != Distribution::fixed;
  LOG(debug) << "parent_a_flexible = " << parent_a_flexible;
  LOG(debug) << "parent_b_flexible = " << parent_b_flexible;
  switch (distribution) {
    case Distribution::gamma:
      LOG(debug) << "Computing gamma distribution gradient.";
      return visit([&](size_t g, size_t t, size_t s) {
        double a = coeffs[parent_a].get(g, t, s);
        double b = coeffs[parent_b].get(g, t, s);
        double x = get(g, t, s);

        grad_coeffs[idx].get(g, t, s) += (a - 1) - x * b;

        if (parent_a_flexible)
          grad_coeffs[parent_a].get(g, t, s)
              += a * (log(b) - digamma(a) + log(x));
        if (parent_b_flexible)
          grad_coeffs[parent_b].get(g, t, s) += a - b * x;

        return log_gamma_rate(x, a, b);
      });
    case Distribution::beta_prime:
      LOG(debug) << "Computing beta prime distribution gradient.";
      return visit([&](size_t g, size_t t, size_t s) {
        double a = coeffs[parent_a].get(g, t, s);
        double b = coeffs[parent_b].get(g, t, s);
        double x = get(g, t, s);
        double p = odds_to_prob(x);

        grad_coeffs[idx].get(g, t, s) += a - 1 - (a + b - 2) * p;

        if (parent_a_flexible)
          grad_coeffs[parent_a].get(g, t, s)
              += log(x) - log(1 + x) + digamma_diff(a, b);
        if (parent_b_flexible)
          grad_coeffs[parent_b].get(g, t, s)
              += -log(1 + x) + digamma_diff(b, a);
        return log_beta_odds(x, a, b);
      });
    case Distribution::log_normal:
      LOG(debug) << "Computing log normal distribution gradient.";
      return visit([&](size_t g, size_t t, size_t s) {
        double exp_mu = coeffs[parent_a].get(g, t, s);
        double sigma = coeffs[parent_b].get(g, t, s);
        double exp_x = get(g, t, s);
        double x = log(exp_x);
        double mu = log(exp_mu);

        grad_coeffs[idx].get(g, t, s) += (mu - x) / (sigma * sigma);

        if (parent_a_flexible)
          grad_coeffs[parent_a].get(g, t, s) += (x - mu) / (sigma * sigma);
        if (parent_b_flexible)
          grad_coeffs[parent_b].get(g, t, s)
              += (x - mu - sigma) * (x - mu + sigma) / (sigma * sigma);
        return log_normal(x, mu, sigma);
      });
    case Distribution::log_gp: {
      size_t n = values.rows();
      LOG(debug) << "Computing log Gaussian process gradient.";
      Coefficient posterior_means = *this;
      Coefficient posterior_vars = *this;
      for (size_t i = 0; i < static_cast<size_t>(values.cols()); ++i) {
        STD::Vector mu(n);
        STD::Vector var(n);
        double delta = 1;
        gp->predict_means_and_vars(values.col(i).array().log(), delta, mu, var);
        posterior_means.values.col(i) = mu;
        posterior_vars.values.col(i) = var;
      }
      return visit([&](size_t g, size_t t, size_t s) {
        double mu = posterior_means.get(g, t, s);
        double var = posterior_vars.get(g, t, s);
        double exp_x = get(g, t, s);
        double x = log(exp_x);

        grad_coeffs[idx].get(g, t, s) += (mu - x) / var;
        // TODO log_gp return value
        return 0;
      });
    }
    case Distribution::fixed:
      return 0;
    default:
      throw std::runtime_error("Error: distribution not implemented.");
  }
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
      throw std::runtime_error("Error: invalid Coefficient::Kind in get().");
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
      throw std::runtime_error("Error: invalid Coefficient::Kind in get() const.");
  }
}

size_t Coefficient::size() const { return values.size(); }

size_t Coefficient::number_parameters() const {
  switch (distribution) {
    case Distribution::fixed:
      return 0;
    default:
      return size();
  }
}

STD::Vector Coefficient::vectorize() const {
  STD::Vector v(size());
  auto iter = begin(v);
  for (auto &x : values)
    *iter++ = x;
  return v;
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
      throw std::runtime_error("Error: invalid Coefficient::Kind in to_string().");
  }
}

string to_string(const Coefficient::Variable &variable) {
  switch (variable) {
    case Coefficient::Variable::rate:
      return "rate";
    case Coefficient::Variable::odds:
      return "odds";
    case Coefficient::Variable::prior:
      return "prior";
    default:
      throw std::runtime_error("Error: invalid Coefficient::Variable.");
  }
}

string to_string(const Coefficient::Distribution &distribution) {
  switch (distribution) {
    case Coefficient::Distribution::fixed:
      return "fixed";
    case Coefficient::Distribution::gamma:
      return "gamma";
    case Coefficient::Distribution::beta_prime:
      return "beta_prime";
    case Coefficient::Distribution::log_normal:
      return "log_normal";
    case Coefficient::Distribution::log_gp:
      return "log_gaussian_process";
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
      throw std::runtime_error("Error: invalid Coefficient::Kind in to_token().");
  }
}

string Coefficient::to_string() const {
  string s = "Coefficient, " + ::to_string(variable) + " "
             + ::to_string(distribution) + "-distributed " + ::to_string(kind);
  for (auto &prior_idx : prior_idxs)
    s += " prior=" + std::to_string(prior_idx);
  return s;
}

ostream &operator<<(ostream &os, const Coefficient &coeff) {
  os << coeff.to_string();
  return os;
}
