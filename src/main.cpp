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

struct Options {
  vector<string> tsv_paths;
  size_t num_factors = 20;
  long num_warm_up = -1;
  bool intersect = false;
  string load_prefix = "";
  bool compute_likelihood = false;
  bool share_coord_sys = false;
  bool fields = false;
  bool perform_dge = false;
  bool keep_empty = false;
  bool transpose = false;
  bool learn_priors = false;
  size_t top = 0;
  size_t bottom = 0;
};

template <typename T>
struct Moments {
  long warm_up;
  size_t n;
  T sum;
  T sumsq;
  Moments(long warm_up_, const T &m)
      : warm_up(warm_up_), n(0), sum(m * 0), sumsq(m * 0) {}
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

void run(const std::vector<Counts> &data_sets, const Options &options,
         const STD::Parameters &parameters) {
  STD::Model pfa(data_sets, options.num_factors, parameters,
                 options.share_coord_sys);
  if (options.load_prefix != "")
    pfa.restore(options.load_prefix);
  LOG(info) << "Initial model" << endl << pfa;
  pfa.gradient_update();
  pfa.store("");
  if (options.compute_likelihood)
    LOG(info) << "Final log-likelihood = "
              << pfa.log_likelihood(pfa.parameters.output_directory);

}

int main(int argc, char **argv) {
  // available:
  // FE_DIVBYZERO is triggered by log(0)
  // FE_INEXACT
  // FE_INVALID
  // FE_OVERFLOW
  // FE_UNDERFLOW
  feenableexcept(FE_INVALID | FE_OVERFLOW);

  EntropySource::seed();

  Options options;

  STD::Parameters parameters;

  string config_path;
  string usage_info = "Spatial Transcriptome Deconvolution\n"
    "\n"
    "This software implements the Spatial Transcriptome Deconvolution model of\n"
    "Jonas Maaskola et al. 2016.\n"
    "Among others it is related to the βγΓ-Poisson Factor Analysis model in\n"
    "\"Beta-Negative Binomial Process and Poisson Factor Analysis\"\n"
    "by Mingyuan Zhou, Lauren A. Hannah, David B. Dunson, and Lawrence Carin\n"
    "Proceedings of the 15th International Conference on Artificial Intelligence and\n"
    "Statistics (AISTATS) 2012, La Palma, Canary Islands. Volume XX of JMLR: W&CP XX.\n"
    "\n"
    "Please provide one or more count matrices as argument to the --file switch or as\n"
    "free arguments. These need to have genes in the rows and spots in the columns.";

  size_t num_cols = 80;
  namespace po = boost::program_options;
  po::options_description cli_options;
  po::options_description generic_options =
      gen_generic_options(config_path, num_cols);

  po::options_description required_options("Required options", num_cols);
  po::options_description basic_options("Basic options", num_cols);
  po::options_description advanced_options("Advanced options", num_cols);
  po::options_description hyperparameter_options("Hyper-parameter options", num_cols);
  po::options_description inference_options("MCMC inference options", num_cols);

  required_options.add_options()
    ("file", po::value(&options.tsv_paths)->required(),
     "Path to a count matrix file, can be given multiple times. "
     "Format: tab-separated, genes in rows, spots in columns; including a "
     "header line, and row names in the first column of each row.");

  basic_options.add_options()
    ("types,t", po::value(&options.num_factors)->default_value(options.num_factors),
     "Maximal number of cell types to look for.")
    ("iter,i", po::value(&parameters.grad_iterations)->default_value(parameters.grad_iterations),
     "Number of iterations to perform.")
    ("report,r", po::value(&parameters.report_interval)->default_value(parameters.report_interval),
     "Interval for reporting the parameters.")
    ("load,l", po::value(&options.load_prefix),
     "Load previous run results with the given path prefix.")
    ("sharecoords", po::bool_switch(&options.share_coord_sys),
     "Assume that the samples lie in the same coordinate system.")
    ("output,o", po::value(&parameters.output_directory),
     "Prefix for generated output files.")
    ("fields", po::bool_switch(&options.fields),
     "Activate fields.")
    ("top", po::value(&options.top)->default_value(options.top),
     "Use only those genes with the highest read count across all spots. Zero indicates all genes.")
    ("bot", po::value(&options.bottom)->default_value(options.bottom),
     "Use only those genes with the lowest read count across all spots. Zero indicates all genes.")
    ("transpose", po::bool_switch(&options.transpose),
     "Count matrices have spots in columns and genes in columns. Default is genes in rows and spots in columns.");

  advanced_options.add_options()
    ("intersect", po::bool_switch(&options.intersect),
     "When using multiple count matrices, use the intersection of rows, rather than their union.")
    ("learnprior", po::bool_switch(&options.learn_priors),
     "Learn priors for gamma and rho.")
    ("normalized_est", po::bool_switch(&parameters.normalize_spot_stats),
     "When sampling theta priors normalize spot statistics.")
    ("keep_empty", po::bool_switch(&options.keep_empty),
     "Do not discard genes or spots with zero counts.")
    ("nolikel", po::bool_switch(&options.compute_likelihood),
     "Do not compute and print the likelihood every iteration.")
    ("nopriors", po::bool_switch(&parameters.ignore_priors),
     "Do not use priors for r and p, i.e. perform (conditional) maximum-likelihood for them, rather than maximum-a-posteriori.")
    ("p_map", po::bool_switch(&parameters.p_empty_map),
     "Choose p(gt) by maximum-a-posteriori rather than by Gibbs sampling when no data is available.")
    ("cont_map", po::bool_switch(&parameters.contributions_map),
     "Sample contributions by maximum-a-posteriori.")
    ("hmc_epsilon", po::value(&parameters.hmc_epsilon)->default_value(parameters.hmc_epsilon),
     "Epsilon parameter for the leapfrog algorithm.")
    ("hmc_l", po::value(&parameters.hmc_L)->default_value(parameters.hmc_L),
     "Number of micro-steps to take in the leapfrog algorithm.")
    ("hmc_n", po::value(&parameters.hmc_N)->default_value(parameters.hmc_N),
     "Number of leapfrog steps to take per iteration.")
    ("dropout", po::value(&parameters.dropout_gene_spot)->default_value(parameters.dropout_gene_spot),
     "Randomly discard a fraction of the gene-spot pairs during sampling.")
    ("compression", po::value(&parameters.compression_mode)->default_value(parameters.compression_mode, "gzip"),
     "Compression method to use. Can be one of 'gzip', 'bzip2', 'none'.")
    ("mesh_dist", po::value(&parameters.mesh_hull_distance)->default_value(parameters.mesh_hull_distance),
     "Maximal distance from the closest given point in which to insert additional mesh points.")
    ("mesh_enlarge", po::value(&parameters.mesh_hull_enlarge)->default_value(parameters.mesh_hull_enlarge),
     "Additional mesh points are sampeled from the bounding box enlarged by this factor (only used if --mesh_dist is zero).")
    ("mesh_add", po::value(&parameters.mesh_additional)->default_value(parameters.mesh_additional),
     "Add additional mesh points uniformly distributed in the bounding box.")
    ("lbfgs_iter", po::value(&parameters.lbfgs_iter)->default_value(parameters.lbfgs_iter),
     "Maximal number of iterations to perform per lBFGS optimization of the field.")
    ("lbfgs_eps", po::value(&parameters.lbfgs_epsilon)->default_value(parameters.lbfgs_epsilon),
     "Epsilon parameter for lBFGS optimization of the field.")
    ("field_lambda_dir", po::value(&parameters.field_lambda_dirichlet)->default_value(parameters.field_lambda_dirichlet),
     "Lambda value for Dirichlet energy in field calculations.")
    ("field_lambda_lap", po::value(&parameters.field_lambda_laplace)->default_value(parameters.field_lambda_laplace),
     "Lambda value for squared Laplace operator in field calculations.")
    ("warm,w", po::value(&options.num_warm_up)->default_value(options.num_warm_up),
     "Length of warm-up period: number of iterations to discard before integrating parameter samples. Negative numbers deactivate MCMC integration.")
    ("expcont", po::bool_switch(&parameters.expected_contributions),
     "Dont sample x_{gst} contributions, but use expected values instead.")
    ("dge", po::bool_switch(&options.perform_dge),
     "Perform differential gene expression analysis.\n"
     "- \tFor all factors, compare each experiment's local profile against the global one.\n"
     "- \tPairwise comparisons between all factors in each experiment.")
    ("sample", po::value(&parameters.targets)->default_value(parameters.targets),
     "Which sampling steps to perform.")
    ("optim", po::value(&parameters.optim_method)->default_value(parameters.optim_method),
     "Which optimization method to use. Available are: Gradient, RPROP, lBFGS.")
    ("grad_alpha", po::value(&parameters.grad_alpha)->default_value(parameters.grad_alpha),
     "Initial learning rate for gradient learning.")
    ("grad_anneal", po::value(&parameters.grad_anneal)->default_value(parameters.grad_anneal),
     "Anneal learning rate by this factor every iteration.");

  hyperparameter_options.add_options()
    ("gamma_1", po::value(&parameters.hyperparameters.gamma_1)->default_value(parameters.hyperparameters.gamma_1),
     "Gamma prior 1 of r[g][t].")
    ("gamma_2", po::value(&parameters.hyperparameters.gamma_2)->default_value(parameters.hyperparameters.gamma_2),
     "Gamma prior 2 of r[g][t].")
    ("lambda_1", po::value(&parameters.hyperparameters.lambda_1)->default_value(parameters.hyperparameters.lambda_1),
     "Gamma prior 1 of local r[g][t].")
    ("lambda_2", po::value(&parameters.hyperparameters.lambda_2)->default_value(parameters.hyperparameters.lambda_2),
     "Gamma prior 2 of local r[g][t].")
    ("rho_1", po::value(&parameters.hyperparameters.rho_1)->default_value(parameters.hyperparameters.rho_1),
     "Beta prior 1 of p[g][t].")
    ("rho_2", po::value(&parameters.hyperparameters.rho_2)->default_value(parameters.hyperparameters.rho_2),
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
    ("bline1", po::value(&parameters.hyperparameters.beta_1)->default_value(parameters.hyperparameters.beta_1),
     "First prior for the baseline features.")
    ("bline2", po::value(&parameters.hyperparameters.beta_2)->default_value(parameters.hyperparameters.beta_2),
     "Second prior for the baseline features.");

  inference_options.add_options()
    ("MHiter", po::value(&parameters.n_iter)->default_value(parameters.n_iter),
     "Maximal number of propositions for Metropolis-Hastings sampling of r")
    ("MHtemp", po::value(&parameters.temperature)->default_value(parameters.temperature),
     "Temperature for Metropolis-Hastings sampling of R.");

  cli_options.add(generic_options)
      .add(required_options)
      .add(basic_options)
      .add(advanced_options)
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

  if (parameters.output_directory == STD::default_output_string) {
    parameters.output_directory
        = generate_random_label(exec_info.program_name, 0) + "/";
  }
  string log_file_path = parameters.output_directory;
  if (parameters.output_directory.empty()) {
    log_file_path = "log.txt";
  } else if (*parameters.output_directory.rbegin() == '/') {
    if (not boost::filesystem::create_directory(parameters.output_directory)) {
      LOG(fatal) << "Error creating output directory "
                 << parameters.output_directory;
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

  auto data_sets
      = load_data(options.tsv_paths, options.intersect, options.top,
                  options.bottom, not options.keep_empty, options.transpose);

  if (options.learn_priors)
    parameters.targets = parameters.targets | STD::Target::gamma_prior
                         | STD::Target::rho_prior;

  if (data_sets.size() < 2) {
    LOG(info)
        << "Less than 2 data sets; deactivating local baseline and features.";
    parameters.targets
        = parameters.targets & ~(STD::Target::lambda | STD::Target::beta);
  }

  if (options.fields)
    parameters.targets = parameters.targets | STD::Target::field;

  LOG(verbose) << "Inference targets = " << parameters.targets;

  try {
    run(data_sets, options, parameters);
  } catch (std::exception &e) {
    LOG(fatal) << "An error occurred during program execution.";
    LOG(fatal) << e.what();
    LOG(fatal) << "Please consult the command line help with -h and the "
                  "documentation.";
    LOG(fatal) << "If errors persist please get in touch with the developers.";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
