#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>
#include <vector>
#include <boost/filesystem.hpp>
#include "aux.hpp"
#include "cli.hpp"
#include "counts.hpp"
#include "io.hpp"
#include "Model.hpp"

using namespace std;
namespace PF = PoissonFactorization;

const string default_output_string = "THIS PATH SHOULD NOT EXIST";
const bool activate_quantiles = false;

vector<string> gen_alpha_labels() {
  vector<string> v;
  for (char y = 'A'; y <= 'Z'; ++y) v.push_back(string() + y);
  for (char y = 'a'; y <= 'z'; ++y) v.push_back(string() + y);
  for (char y = '0'; y <= '9'; ++y) v.push_back(string() + y);
  return v;
}

const vector<string> alphabetic_labels = gen_alpha_labels();

struct Options {
  enum class Labeling { Auto, None, Path, Alpha };
  vector<string> tsv_paths;
  size_t num_factors = 20;
  size_t num_burn_in = 200;
  size_t num_steps = 2000;
  size_t report_interval = 20;
  string output = default_output_string;
  bool intersect = false;
  vector<double> quantiles;
  Labeling labeling = Labeling::Auto;
  bool compute_likelihood = false;
  bool perform_splitmerge = false;
  bool timing = true;
  size_t top = 0;
  PF::Target sample_these = PF::DefaultTarget();
  PF::Partial::Kind feature_type = PF::Partial::Kind::Gamma;
  PF::Partial::Kind mixing_type = PF::Partial::Kind::HierGamma;
};

istream &operator>>(istream &is, Options::Labeling &label) {
  string token;
  is >> token;
  token = to_lower(token);
  if (token == "auto")
    label = Options::Labeling::Auto;
  else if (token == "none")
    label = Options::Labeling::None;
  else if (token == "path")
    label = Options::Labeling::Path;
  else if (token == "alpha")
    label = Options::Labeling::Alpha;
  else
    throw std::runtime_error("Error: could not parse labeling '" + token +
                             "'.");
  return is;
}

ostream &operator<<(ostream &os, const Options::Labeling &label) {
  switch (label) {
    case Options::Labeling::Auto:
      os << "auto";
      break;
    case Options::Labeling::None:
      os << "none";
      break;
    case Options::Labeling::Path:
      os << "path";
      break;
    case Options::Labeling::Alpha:
      os << "alpha";
      break;
  }
  return os;
}

// TODO automatically determine return type
template <typename Iter>
vector<double> get_quantiles(const Iter begin, const Iter end, const vector<double> &quantiles) {
  vector<double> res;
  sort(begin, end);
  const size_t N = std::distance(begin, end);
  for(auto quantile: quantiles)
    res.push_back(*(begin + size_t((N-1) * quantile)));
  return res;
}

// TODO implement for features and weights
template <typename T_>
vector<T_> mcmc_quantiles(const vector<T_> &models, const vector<double> &quantiles) {
  const size_t M = models.size();
  const size_t Q = quantiles.size();

  const size_t G = models.begin()->G;
  const size_t S = models.begin()->S;
  const size_t T = models.begin()->T;
  const size_t E = models.begin()->E;

  const size_t GT = G * T;
  const size_t ST = S * T;

  vector<T_> quantile_models(Q, *models.begin());

  // contributions_gene_type
  {
    PF::IMatrix v(M, GT, arma::fill::zeros);
    for (size_t m = 0; m < M; ++m)
      for (size_t gt = 0; gt < GT; ++gt)
        v(m, gt) = models[m].contributions_gene_type(gt);

    for (size_t gt = 0; gt < GT; ++gt) {
      auto percentiles = get_quantiles(v.begin_col(gt), v.end_col(gt), quantiles);
      for (size_t q = 0; q < Q; ++q)
        quantile_models[q].contributions_gene_type(gt) = percentiles[q];
    }
  }

  // contributions_spot_type
  {
    PF::IMatrix v(M, ST, arma::fill::zeros);
    for (size_t m = 0; m < M; ++m)
      for (size_t st = 0; st < ST; ++st)
        v(m, st) = models[m].contributions_spot_type(st);

    for (size_t st = 0; st < ST; ++st) {
      auto percentiles = get_quantiles(v.begin_col(st), v.end_col(st), quantiles);
      for (size_t q = 0; q < Q; ++q)
        quantile_models[q].contributions_spot_type(st) = percentiles[q];
    }
  }

  // contributions_gene
  {
    PF::IMatrix v(M, G, arma::fill::zeros);
    for (size_t m = 0; m < M; ++m)
      for (size_t g = 0; g < G; ++g)
        v(m, g) = models[m].contributions_gene(g);

    for (size_t g = 0; g < G; ++g) {
      auto percentiles = get_quantiles(v.begin_col(g), v.end_col(g), quantiles);
      for (size_t q = 0; q < Q; ++q)
        quantile_models[q].contributions_gene(g) = percentiles[q];
    }
  }

  // contributions_spot
  {
    PF::IMatrix v(M, S, arma::fill::zeros);
    for (size_t m = 0; m < M; ++m)
      for (size_t s = 0; s < S; ++s)
        v(m, s) = models[m].contributions_spot(s);

    for (size_t s = 0; s < S; ++s) {
      auto percentiles = get_quantiles(v.begin_col(s), v.end_col(s), quantiles);
      for (size_t q = 0; q < Q; ++q)
        quantile_models[q].contributions_spot(s) = percentiles[q];
    }
  }

  // contributions_experiment
  {
    PF::IMatrix v(M, E, arma::fill::zeros);
    for (size_t m = 0; m < M; ++m)
      for (size_t e = 0; e < E; ++e)
        v(m, e) = models[m].contributions_experiment(e);

    for (size_t e = 0; e < E; ++e) {
      auto percentiles = get_quantiles(v.begin_col(e), v.end_col(e), quantiles);
      for (size_t q = 0; q < Q; ++q)
        quantile_models[q].contributions_experiment(e) = percentiles[q];
    }
  }

  // features
  {
    PF::IMatrix v(M, GT, arma::fill::zeros);
    for (size_t m = 0; m < M; ++m)
      for (size_t gt = 0; gt < GT; ++gt)
        v(m, gt) = models[m].features.matrix(gt);

    for (size_t gt = 0; gt < GT; ++gt) {
      auto percentiles = get_quantiles(v.begin_col(gt), v.end_col(gt), quantiles);
      for (size_t q = 0; q < Q; ++q)
        quantile_models[q].features.matrix(gt) = percentiles[q];
    }
  }

  // mixing weights
  {
    PF::IMatrix v(M, ST, arma::fill::zeros);
    for (size_t m = 0; m < M; ++m)
      for (size_t st = 0; st < ST; ++st)
        v(m, st) = models[m].weights.matrix(st);

    for (size_t st = 0; st < ST; ++st) {
      auto percentiles = get_quantiles(v.begin_col(st), v.end_col(st), quantiles);
      for (size_t q = 0; q < Q; ++q)
        quantile_models[q].weights.matrix(st) = percentiles[q];
    }
  }

  return quantile_models;
}

template <typename T>
void perform_gibbs_sampling(const Counts &data, T &pfa,
                            const Options &options) {
  LOG(info) << "Initial model" << endl << pfa;
  vector<T> models;
  models.push_back(pfa);
  for (size_t iteration = 1; iteration <= options.num_steps; ++iteration) {
    if (iteration > pfa.parameters.enforce_iter)
      pfa.parameters.enforce_mean = PF::ForceMean::None;
    LOG(info) << "Performing iteration " << iteration;
    pfa.gibbs_sample(data, options.sample_these, options.timing);
    LOG(info) << "Current model" << endl << pfa;
    if (iteration % options.report_interval == 0)
      pfa.store(data, options.output + "iter" + to_string(iteration) + "_");
    if (options.compute_likelihood)
      LOG(info) << "Log-likelihood = "
                << pfa.log_likelihood_poisson_counts(data.counts);
    if (activate_quantiles)
      if (iteration > options.num_burn_in)
        models.push_back(pfa);
  }
  if (activate_quantiles) {
    auto quantile_models = mcmc_quantiles(models, options.quantiles);
    for (size_t q = 0; q < options.quantiles.size(); ++q)
      quantile_models[q].store(
          data,
          options.output + "quantile" + to_string(options.quantiles[q]) + "_");
  }
  if (options.compute_likelihood)
    LOG(info) << "Final log-likelihood = "
              << pfa.log_likelihood_poisson_counts(data.counts);
  pfa.store(data, options.output, true);
}

int main(int argc, char **argv) {
  EntropySource::seed();

  Options options;

  PF::Parameters parameters;

  string config_path;
  string usage_info = "This software implements the βγΓ-Poisson Factor Analysis model of\n"
    "Beta-Negative Binomial Process and Poisson Factor Analysis\n"
    "by Mingyuan Zhou, Lauren A. Hannah, David B. Dunson, and Lawrence Carin\n"
    "Proceedings of the 15th International Conference on Artificial Intelligence and\n"
    "Statistics (AISTATS) 2012, La Palma, Canary Islands. Volume XX of JMLR: W&CP XX.\n"
    "\n"
    "Please provide one or more count matrices as argument to the --file switch or as\n"
    "free arguments.";

  size_t num_cols = 80;
  namespace po = boost::program_options;
  po::options_description cli_options;
  po::options_description generic_options =
      gen_generic_options(config_path, num_cols);

  po::options_description required_options("Required options", num_cols);
  po::options_description basic_options("Basic options", num_cols);
  po::options_description hyperparameter_options("Hyper-parameter options", num_cols);
  po::options_description inference_options("MCMC inference options", num_cols);

  required_options.add_options()
    ("file", po::value(&options.tsv_paths)->required(),
     "Path to a count matrix file, can be given multiple times. "
     "Format: tab-separated, including a header line, "
     "and row names in the first column of each row.");

  basic_options.add_options()
    ("feature,f", po::value(&options.feature_type)->default_value(options.feature_type),
     "Which type of distribution to use for the features. "
     "Can be one of 'gamma' or 'dirichlet'.")
    ("mix,m", po::value(&options.mixing_type)->default_value(options.mixing_type),
     "Which type of distribution to use for the mixing weights. "
     "Can be one of 'gamma' or 'dirichlet'.")
    ("types,t", po::value(&options.num_factors)->default_value(options.num_factors),
     "Maximal number of cell types to look for.")
    ("iter,i", po::value(&options.num_steps)->default_value(options.num_steps),
     "Number of iterations to perform.")
    ("burn,b", po::value(&options.num_burn_in)->default_value(options.num_burn_in),
     "Length of burn-in period: number of iterations to discard before integrating parameter samples.")
    ("report,r", po::value(&options.report_interval)->default_value(options.report_interval),
     "Interval for reporting the parameters.")
    ("nolikel", po::bool_switch(&options.compute_likelihood),
     "Do not compute and print the likelihood every iteration.")
    ("var,V", po::bool_switch(&parameters.variational),
     "Sample contribution marginals. This is faster but less accurate.")
    ("split,s", po::bool_switch(&options.perform_splitmerge),
     "Perform split/merge steps.")
    ("output,o", po::value(&options.output),
     "Prefix for generated output files.")
    ("top", po::value(&options.top)->default_value(options.top),
     "Use only those genes with the highest read count across all spots. Zero indicates all genes.")
    ("intersect", po::bool_switch(&options.intersect),
     "When using multiple count matrices, use the intersection of rows, rather than their union.")
    ("timing", po::bool_switch(&options.timing),
     "Print out timing information.")
    ("forcemean", po::value(&parameters.enforce_mean)->default_value(parameters.enforce_mean),
     "Enforce means / sums of random variables. Can be any comma-separated combination of 'theta', 'phi', 'spot', 'experiment'.")
    ("forceiter", po::value(&parameters.enforce_iter)->default_value(parameters.enforce_iter),
     "How long to enforce means / sums of random variables. 0 means forever, anything else the given number of iterations.")
    ("expscale", po::bool_switch(&parameters.activate_experiment_scaling),
     "Activate usage of the experiment scaling variables.")
    ("sample", po::value(&options.sample_these)->default_value(options.sample_these),
     "Which sampling steps to perform.")
    ("quant,q", po::value<vector<double>>(&options.quantiles)->default_value({0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0}, "0,0.05,0.25,0.5,0.75,0.95,1.0"),
     "Which quantiles to report for each parameter.")
    ("label", po::value(&options.labeling),
     "How to label the spots. Can be one of 'alpha', 'path', 'none'. If only one count table is given, the default is to use 'none'. If more than one is given, the default is 'alpha'.");

  hyperparameter_options.add_options()
    ("alpha", po::value(&parameters.hyperparameters.alpha)->default_value(parameters.hyperparameters.alpha),
     "Dirichlet prior alpha of the factor loading matrix.")
    ("phi_r_1", po::value(&parameters.hyperparameters.phi_r_1)->default_value(parameters.hyperparameters.phi_r_1),
     "Gamma prior 1 of r[g][t].")
    ("phi_r_2", po::value(&parameters.hyperparameters.phi_r_2)->default_value(parameters.hyperparameters.phi_r_2),
     "Gamma prior 2 of r[g][t].")
    ("phi_p_1", po::value(&parameters.hyperparameters.phi_p_1)->default_value(parameters.hyperparameters.phi_p_1),
     "Beta prior 1 of p[g][t].")
    ("phi_p_2", po::value(&parameters.hyperparameters.phi_p_2)->default_value(parameters.hyperparameters.phi_p_2),
     "Beta prior 2 of p[g][t].")
    ("theta_r_1", po::value(&parameters.hyperparameters.theta_r_1)->default_value(parameters.hyperparameters.theta_r_1),
     "Gamma prior 1 of r[t].")
    ("theta_r_2", po::value(&parameters.hyperparameters.theta_r_2)->default_value(parameters.hyperparameters.theta_r_2),
     "Gamma prior 2 of r[t].")
    ("theta_p_1", po::value(&parameters.hyperparameters.theta_p_1)->default_value(parameters.hyperparameters.theta_p_1, to_string(round(parameters.hyperparameters.theta_p_1 * 100) / 100)),
     "Beta prior 1 of p[t].")
    ("theta_p_2", po::value(&parameters.hyperparameters.theta_p_2)->default_value(parameters.hyperparameters.theta_p_2, to_string(round(parameters.hyperparameters.theta_p_2 * 100) / 100)),
     "Beta prior 2 of p[t].")
    ("spot_1", po::value(&parameters.hyperparameters.spot_a)->default_value(parameters.hyperparameters.spot_a),
     "Gamma prior 1 of the spot scaling parameter.")
    ("spot_2", po::value(&parameters.hyperparameters.spot_b)->default_value(parameters.hyperparameters.spot_b),
     "Gamma prior 2 of the spot scaling parameter.")
    ("exp_1", po::value(&parameters.hyperparameters.experiment_a)->default_value(parameters.hyperparameters.experiment_a),
     "Gamma prior 1 of the experiment scaling parameter.")
    ("exp_2", po::value(&parameters.hyperparameters.experiment_b)->default_value(parameters.hyperparameters.experiment_b),
     "Gamma prior 2 of the experiment scaling parameter.");

  inference_options.add_options()
    ("MHiter", po::value(&parameters.n_iter)->default_value(parameters.n_iter),
     "Maximal number of propositions for Metropolis-Hastings sampling of r")
    ("MHtemp", po::value(&parameters.temperature)->default_value(parameters.temperature),
     "Temperature for Metropolis-Hastings sampling of R.")
    ("MHsd", po::value(&parameters.prop_sd)->default_value(parameters.prop_sd),
     "Standard deviation for log-normal proposition scaling in Metropolis-Hastings sampling of r[g][t]");

  cli_options.add(generic_options)
      .add(required_options)
      .add(basic_options)
      .add(hyperparameter_options)
      .add(inference_options);

  po::positional_options_description positional_options;
  positional_options.add("file", -1);

  ExecutionInformation exec_info;
  int ret_val
      = process_cli_options(argc, const_cast<const char **>(argv), exec_info,
                            usage_info, cli_options, true, positional_options);
  if (ret_val != PROCESSING_SUCCESSFUL)
    return ret_val;

  // invert the negative CLI switch value
  options.compute_likelihood = !options.compute_likelihood;

  if (options.output == default_output_string) {
    options.output
        = generate_random_label(exec_info.program_name, 0)
          + "/";
  }
  string log_file_path = options.output;
  if (options.output.empty()) {
    log_file_path = "log.txt";
  } else if (*options.output.rbegin() == '/') {
    if (not boost::filesystem::create_directory(options.output)) {
      LOG(fatal) << "Error creating output directory " << options.output;
      return EXIT_FAILURE;
    } else {
      log_file_path += "log.txt";
    }
  } else {
    log_file_path += ".log.txt";
  }

  init_logging(log_file_path);

  LOG(info) << exec_info.name_and_version();
  LOG(info) << exec_info.datetime;
  LOG(info) << "Working directory = " << exec_info.directory;
  LOG(info) << "Command = " << exec_info.cmdline << endl;

  vector<string> labels;
  switch (options.labeling) {
    case Options::Labeling::None:
      labels = vector<string>(options.tsv_paths.size(), "");
      break;
    case Options::Labeling::Path:
      labels = options.tsv_paths;
      break;
    case Options::Labeling::Alpha:
      labels = alphabetic_labels;
      break;
    case Options::Labeling::Auto:
      if (options.tsv_paths.size() > 1)
        labels = alphabetic_labels;
      else
        labels = vector<string>(options.tsv_paths.size(), "");
      break;
  }

  if (labels.size() < options.tsv_paths.size()) {
    LOG(warning) << "Warning: too few labels available! Using paths as labels.";
    labels = options.tsv_paths;
  }

  if (options.perform_splitmerge)
    options.sample_these
        = options.sample_these | PoissonFactorization::Target::merge_split;

  Counts data(options.tsv_paths[0], labels[0]);
  for (size_t i = 1; i < options.tsv_paths.size(); ++i)
    if (options.intersect)
      data = data * Counts(options.tsv_paths[i], labels[i]);
    else
      data = data + Counts(options.tsv_paths[i], labels[i]);

  if (options.top > 0)
    data.select_top(options.top);

  LOG(info) << "Using " << options.feature_type
            << " distribution for the features.";
  LOG(info) << "Using " << options.mixing_type
            << " distribution for the mixing weights.";

  using Kind = PF::Partial::Kind;
  switch (options.feature_type) {
    case Kind::Dirichlet: {
      switch (options.mixing_type) {
        case Kind::Dirichlet: {
          PF::Model<Kind::Dirichlet, Kind::Dirichlet> pfa(
              data, options.num_factors, parameters);
          perform_gibbs_sampling(data, pfa, options);
        } break;
        case Kind::HierGamma: {
          PF::Model<Kind::Dirichlet, Kind::HierGamma> pfa(
              data, options.num_factors, parameters);
          perform_gibbs_sampling(data, pfa, options);
        } break;
        default:
          throw std::runtime_error("Error: Mixing type '"
                                   + to_string(options.mixing_type)
                                   + "' not implemented.");
          break;
      }
    } break;
    case Kind::Gamma: {
      switch (options.mixing_type) {
        case Kind::Dirichlet: {
          PF::Model<Kind::Gamma, Kind::Dirichlet> pfa(data, options.num_factors,
                                                      parameters);
          perform_gibbs_sampling(data, pfa, options);
        } break;
        case Kind::HierGamma: {
          PF::Model<Kind::Gamma, Kind::HierGamma> pfa(data, options.num_factors,
                                                      parameters);
          perform_gibbs_sampling(data, pfa, options);
        } break;
        default:
          throw std::runtime_error("Error: Mixing type '"
                                   + to_string(options.mixing_type)
                                   + "' not implemented.");
          break;
      }
      break;
      default:
        throw std::runtime_error("Error: feature type '"
                                 + to_string(options.feature_type)
                                 + "' not implemented.");
        break;
    }
  }

  return EXIT_SUCCESS;
}
