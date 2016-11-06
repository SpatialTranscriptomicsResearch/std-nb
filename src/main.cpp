#include <cstdlib>
#include <exception>
#include <fenv.h>
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

struct Options {
  enum class Labeling { Auto, None, Path, Alpha };
  vector<string> tsv_paths;
  size_t num_factors = 20;
  long num_warm_up = -1;
  size_t num_steps = 2000;
  size_t report_interval = 200;
  string output = default_output_string;
  bool intersect = false;
  Labeling labeling = Labeling::Auto;
  bool compute_likelihood = false;
  bool no_local_gene_expression = false;
  bool sample_local_phi_priors = false;
  bool share_coord_sys = false;
  bool perform_pairwise_dge = false;
  bool stochastic_gibbs = false;
  size_t top = 0;
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

template <typename T>
struct Moments {
  long warm_up;
  size_t n;
  T sum;
  T sumsq;
  Moments(long warm_up_, const T &m) : warm_up(warm_up_), n(0), sum(m * 0), sumsq(m * 0) {
  }
  void update(long iteration, const T &m) {
    if (warm_up >= 0 and iteration >= warm_up) {
      sum = sum + m;
      sumsq = sumsq + m * m;
      n++;
    }
  }
  void evaluate(const std::string &prefix) {
    if (n > 1) {
      T var = (sumsq - sum * sum / n) / (n - 1);
      sum = sum / n;
      sum.store(prefix + "mean_");
      var.store(prefix + "variance_");
    }
  }
};

template <typename T>
void perform_gibbs_sampling(const vector<Counts> &data, T &pfa,
                            const Options &options) {
  LOG(info) << "Initial model" << endl << pfa;
  Moments<T> moments(options.num_warm_up,
                     options.num_warm_up >= 0
                         ? pfa
                         : T({}, 0, pfa.parameters, options.share_coord_sys));
  std::vector<size_t> which_experiments;

  const size_t iteration_num_digits
      = 1 + ceil(log(options.num_steps) / log(10));

  for (size_t iteration = 1; iteration <= options.num_steps; ++iteration) {
    LOG(info) << "Performing iteration " << iteration;
    if (iteration == 1 or not options.stochastic_gibbs) {
      which_experiments.resize(data.size());
      std::iota(begin(which_experiments), end(which_experiments), 0);
    } else {
      which_experiments.resize(1);
      which_experiments[0] = std::uniform_int_distribution<size_t>(
          0, data.size() - 1)(EntropySource::rng);
    }
    pfa.gibbs_sample(which_experiments);
    LOG(verbose) << "Current model" << endl << pfa;
    if (iteration % options.report_interval == 0) {
      const string prefix
          = options.output + "iter"
            + to_string_embedded(iteration, iteration_num_digits) + "/";
      if (boost::filesystem::create_directory(prefix)) {
        pfa.store(prefix);
        if (options.perform_pairwise_dge) {
          pfa.perform_local_dge(prefix);
          pfa.perform_pairwise_dge(prefix);
        }
      } else {
        throw(std::runtime_error("Couldn't create directory " + prefix));
      }
    }

    if (options.compute_likelihood) {
      if (verbosity >= Verbosity::verbose) {
        LOG(info) << "Log-likelihood = " << pfa.log_likelihood();
      } else {
        LOG(info) << "Observed Log-likelihood = " << pfa.log_likelihood_poisson_counts();
      }
    }

    moments.update(iteration, pfa);
  }
  moments.evaluate(options.output);
  if (options.compute_likelihood)
    LOG(info) << "Final log-likelihood = " << pfa.log_likelihood();
  pfa.store(options.output);
  pfa.perform_local_dge(options.output);
  if (options.perform_pairwise_dge)
    pfa.perform_pairwise_dge(options.output);
}

int main(int argc, char **argv) {
  feenableexcept(FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW);

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
    ("warm,w", po::value(&options.num_warm_up)->default_value(options.num_warm_up),
     "Length of warm-up period: number of iterations to discard before integrating parameter samples. Negative numbers deactivate MCMC integration.")
    ("report,r", po::value(&options.report_interval)->default_value(options.report_interval),
     "Interval for reporting the parameters.")
    ("nolikel", po::bool_switch(&options.compute_likelihood),
     "Do not compute and print the likelihood every iteration.")
    ("overrelax", po::bool_switch(&parameters.over_relax),
     "Perform overrelaxation. See arXiv:bayes-an/9506004.")
    ("pairwisedge", po::bool_switch(&options.perform_pairwise_dge),
     "Perform pairwise comparisons between all factors in each experiment.")
    ("stochastic", po::bool_switch(&options.stochastic_gibbs),
     "Perform stochastic Gibbs sampling by sampling contributions only for one randomly selected experiement per iteration.")
    ("localthetapriors", po::bool_switch(&parameters.theta_local_priors),
     "Use local priors for the mixing weights.")
    ("nolocal", po::bool_switch(&options.no_local_gene_expression),
     "Deactivate local gene expression profiles.")
    ("phi_ml", po::bool_switch(&parameters.phi_prior_maximum_likelihood),
     "Use maximum likelihood instead of Metropolis-Hastings for the first prior of Φ.")
    ("phi_likel", po::bool_switch(&parameters.respect_phi_prior_likelihood),
     "Respect the likelihood contributions of the feature priors.")
    ("theta_likel", po::bool_switch(&parameters.respect_theta_prior_likelihood),
     "Respect the likelihood contributions of the mixture priors.")
    ("expcont", po::bool_switch(&parameters.expected_contributions),
     "Dont sample x_{gst} contributions, but use expected values instead.")
    ("sharecoords", po::bool_switch(&options.share_coord_sys),
     "Assume that the samples lie in the same coordinate system.")
    ("localphi", po::bool_switch(&options.sample_local_phi_priors),
     "Sample the local feature priors.")
    ("lambda", po::bool_switch(&parameters.store_lambda),
     "Store to disk the lambda matrix for genes and types every time parameters are written. "
     "(This file is about the same size as the input files, so in order to limit storage usage you may not want to store it.)")
    ("output,o", po::value(&options.output),
     "Prefix for generated output files.")
    ("top", po::value(&options.top)->default_value(options.top),
     "Use only those genes with the highest read count across all spots. Zero indicates all genes.")
    ("intersect", po::bool_switch(&options.intersect),
     "When using multiple count matrices, use the intersection of rows, rather than their union.")
    ("forceiter", po::value(&parameters.enforce_iter)->default_value(parameters.enforce_iter),
     "How long to enforce means / sums of random variables. 0 means forever, anything else the given number of iterations.")
    ("sample", po::value(&parameters.targets)->default_value(parameters.targets),
     "Which sampling steps to perform.")
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
    ("sigma", po::value(&parameters.hyperparameters.sigma)->default_value(parameters.hyperparameters.sigma),
     "Sigma parameter for field characteristic length scale.");

  inference_options.add_options()
    ("MHiter", po::value(&parameters.n_iter)->default_value(parameters.n_iter),
     "Maximal number of propositions for Metropolis-Hastings sampling of r")
    ("MHtemp", po::value(&parameters.temperature)->default_value(parameters.temperature),
     "Temperature for Metropolis-Hastings sampling of R.");

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

  auto data_sets = load_data(options.tsv_paths, options.intersect, options.top);

  LOG(info) << "Using " << options.feature_type
            << " distribution for the features.";
  LOG(info) << "Using " << options.mixing_type
            << " distribution for the mixing weights.";

  using Kind = PF::Partial::Kind;

  if (options.sample_local_phi_priors)
    parameters.targets = parameters.targets | PF::Target::phi_prior_local;

  if (data_sets.size() < 2) {
    LOG(info) << "Deactivating local features.";
    options.no_local_gene_expression = true;
  }

  if (options.no_local_gene_expression)
    parameters.targets
        = parameters.targets
          & (~(PF::Target::phi_local | PF::Target::phi_prior_local));

  switch (options.feature_type) {
    case Kind::Dirichlet: {
      /* TODO reactivate
      switch (options.mixing_type) {
        case Kind::Dirichlet: {
          PF::Model<Kind::Dirichlet, Kind::Dirichlet> pfa(
              data, options.num_factors, parameters);
          perform_gibbs_sampling(data, pfa, options);
        } break;
        case Kind::HierGamma: {
          PF::Model<Kind::Dirichlet, Kind::HierGamma> pfa(
              data_sets, options.num_factors, parameters);
          perform_gibbs_sampling(data_sets, pfa, options);
        } break;
        default:
          throw std::runtime_error("Error: Mixing type '"
                                   + to_string(options.mixing_type)
                                   + "' not implemented.");
          break;
      }
      */
    } break;
    case Kind::Gamma: {
      switch (options.mixing_type) {
        /*
        case Kind::Dirichlet: {
          PF::Model<PF::ModelType<Kind::Gamma, Kind::Dirichlet>> pfa(
              data_sets, options.num_factors, parameters,
              options.share_coord_sys);
          perform_gibbs_sampling(data_sets, pfa, options);
        } break;
        */
        case Kind::HierGamma: {
          PF::Model<PF::ModelType<Kind::Gamma, Kind::HierGamma>> pfa(
              data_sets, options.num_factors, parameters,
              options.share_coord_sys);
          perform_gibbs_sampling(data_sets, pfa, options);
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
