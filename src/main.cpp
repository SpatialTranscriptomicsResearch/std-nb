#include <fenv.h>
#include <boost/filesystem.hpp>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>
#include <vector>
#include "Model.hpp"
#include "aux.hpp"
#include "cli.hpp"
#include "counts.hpp"
#include "io.hpp"

using namespace std;

struct Options {
  vector<string> tsv_paths;
  string design_path;
  Design design;
  size_t num_factors = 20;
  bool intersect = false;
  string load_prefix = "";
  // TODO covariates reactivate likelihood
  // bool compute_likelihood = false;
  bool share_coord_sys = false;
  bool keep_empty = false;
  bool transpose = false;
  size_t top = 0;
  size_t bottom = 0;
};

void run(const std::vector<Counts> &data_sets, const Options &options,
         const STD::Parameters &parameters) {
  STD::Model pfa(data_sets, options.num_factors,
                 options.design, parameters, options.share_coord_sys);
  if (options.load_prefix != "")
    pfa.restore(options.load_prefix);
  LOG(info) << "Initial model" << endl << pfa;
  pfa.gradient_update();
  pfa.store("", true);
  /* TODO covariates reactivate likelihood
  if (options.compute_likelihood)
    LOG(info) << "Final log-likelihood = "
              << pfa.log_likelihood(pfa.parameters.output_directory);
  */
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
  po::options_description generic_options
      = gen_generic_options(config_path, num_cols);

  po::options_description required_options("Required options", num_cols);
  po::options_description basic_options("Basic options", num_cols);
  po::options_description gaussian_process_options("Gaussian process options", num_cols);
  po::options_description advanced_options("Advanced options", num_cols);
  po::options_description grad_options("Gradient descent options", num_cols);
  po::options_description hmc_options("Hybrid Monte-Carlo options", num_cols);
  po::options_description lbfgs_options("lBFGS options", num_cols);
  po::options_description rprop_options("RPROP options", num_cols);
  po::options_description inference_options("MCMC inference options", num_cols);
  po::options_description hyperparameter_options("Hyper-parameter options", num_cols);

  required_options.add_options()
    ("file", po::value(&options.tsv_paths),
     "Path to a count matrix file, can be given multiple times. "
     "Format: tab-separated, genes in rows, spots in columns; including a "
     "header line, and row names in the first column of each row.")
    ("design,d", po::value(&options.design_path),
     "Path to a design matrix file. "
     "Format: tab-separated.");

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
    ("formula,f", po::value(&parameters.rate_formula)->default_value(parameters.rate_formula),
     "Regression formula for the rate parameter of negative binomial rate. "
     "Arbitrary covariates annotated to the input files in the design matrix file. "
     "Additionally, the following covariates are predefined by default:\n"
     "gene    \tcovariate with values for every gene\n"
     "type    \tcovariate with values for every type\n"
     "section \tcovariate with values for every input file. "
     "Note that in case you have split one section's spots into two different count matrix input files, you can specify your own 'section' column in the design matrix file and give the correct input file to section mapping there.\n"
     "1       \tcovariate serving as an intercept term to set overall baseline")
    ("varformula", po::value(&parameters.variance_formula)->default_value(parameters.variance_formula),
     "Regression formula for the variance parameter of negative binomial rate. "
     "See information on --formula for syntax.")
    ("distmode", po::value(&parameters.distribution_mode)->default_value(parameters.distribution_mode),
     "Which probability distributions to use as default:\n"
     "gamma_odds            \tUse gamma distributions for rate and prior parameters, and beta prime distributions for the odds parameters\n"
     "gamma_odds_log_normal \tUse gamma distributions for rate parameters, beta prime distributions for the odds parameters, and log normal for the prior parameters\n"
     "log_normal \tUse log normal for all parameters")
    ("top", po::value(&options.top)->default_value(options.top),
     "Use only those genes with the highest read count across all spots. Zero indicates all genes.")
    ("bot", po::value(&options.bottom)->default_value(options.bottom),
     "Use only those genes with the lowest read count across all spots. Zero indicates all genes.")
    ("transpose", po::bool_switch(&options.transpose),
     "Count matrices have spots in columns and genes in columns. Default is genes in rows and spots in columns.");

  gaussian_process_options.add_options()
    ("gp", po::bool_switch(&parameters.gp.use),
     "Model spatial factor activities as a Gaussian process.")
    ("gp_length", po::value(&parameters.gp.length_scale)->default_value(parameters.gp.length_scale),
     "Length scale to use for Gaussian process.")
    ("gp_var_spatial", po::value(&parameters.gp.spatial_variance)->default_value(parameters.gp.spatial_variance),
     "Spatial variance to use for Gaussian process.")
    ("gp_var_indep", po::value(&parameters.gp.independent_variance)->default_value(parameters.gp.independent_variance),
     "Independent variance to use for Gaussian process.");

  advanced_options.add_options()
    ("intersect", po::bool_switch(&options.intersect),
     "When using multiple count matrices, use the intersection of rows, rather than their union.")
    ("minval", po::value(&parameters.min_value)->default_value(parameters.min_value),
     "Minimal value to enforce for parameters.")
    ("warn", po::bool_switch(&parameters.warn_lower_limit),
     "Warn when parameter values reach the lower limit specified by --minval.")
    ("keep_empty", po::bool_switch(&options.keep_empty),
     "Do not discard genes or spots with zero counts.")
    // TODO covariates reactivate likelihood
    // ("likel", po::bool_switch(&options.compute_likelihood),
    //  "Compute and print the likelihood after finishing.")
    ("dropout", po::value(&parameters.dropout_gene_spot)->default_value(parameters.dropout_gene_spot),
     "Randomly discard a fraction of the gene-spot pairs during sampling.")
    ("compression", po::value(&parameters.compression_mode)->default_value(parameters.compression_mode, "gzip"),
     "Compression method to use. Can be one of 'gzip', 'bzip2', 'none'.")
    ("optim", po::value(&parameters.optim_method)->default_value(parameters.optim_method),
     "Which optimization method to use. Available are: Gradient, RPROP, lBFGS.")
    ("contrib", po::value(&parameters.sample_method)->default_value(parameters.sample_method),
     "How to sample the contributions. Available are: Mean, Multinomial, Trial, TrialMean, MH, HMC, RPROP, lBFGS.")
    ("sample_iter", po::value(&parameters.sample_iterations)->default_value(parameters.sample_iterations),
     "Number of iterations to perform for iterative sampling methods.");

  lbfgs_options.add_options()
    ("lbfgs_iter", po::value(&parameters.lbfgs_iter)->default_value(parameters.lbfgs_iter),
     "Maximal number of iterations to perform per lBFGS optimization.")
    ("lbfgs_eps", po::value(&parameters.lbfgs_epsilon)->default_value(parameters.lbfgs_epsilon),
     "Epsilon parameter for lBFGS optimization.");

  hmc_options.add_options()
    ("hmc_epsilon", po::value(&parameters.hmc_epsilon)->default_value(parameters.hmc_epsilon),
     "Epsilon parameter for the leapfrog algorithm.")
    ("hmc_l", po::value(&parameters.hmc_L)->default_value(parameters.hmc_L),
     "Number of micro-steps to take in the leapfrog algorithm.")
    ("hmc_n", po::value(&parameters.hmc_N)->default_value(parameters.hmc_N),
     "Number of leapfrog steps to take per iteration.");

  grad_options.add_options()
    ("grad_alpha", po::value(&parameters.grad_alpha)->default_value(parameters.grad_alpha),
     "Initial learning rate for gradient learning.")
    ("grad_anneal", po::value(&parameters.grad_anneal)->default_value(parameters.grad_anneal),
     "Anneal learning rate by this factor every iteration.");

  rprop_options.add_options()
    ("rprop_etap", po::value(&parameters.rprop.eta_plus)->default_value(parameters.rprop.eta_plus),
     "Multiplier for parameter-specific learning rates in case of successive gradients with identical sign.")
    ("rprop_etam", po::value(&parameters.rprop.eta_minus)->default_value(parameters.rprop.eta_minus),
     "Multiplier for parameter-specific learning rates in case of successive gradients with opposite sign.")
    ("rprop_min", po::value(&parameters.rprop.min_change)->default_value(parameters.rprop.min_change),
     "Minimal parameter-specific learning rate.")
    ("rprop_max", po::value(&parameters.rprop.max_change)->default_value(parameters.rprop.max_change),
     "Maximal parameter-specific learning rate.");

  hyperparameter_options.add_options()
    ("gamma_1", po::value(&parameters.hyperparameters.gamma_1)->default_value(parameters.hyperparameters.gamma_1),
     "Prior 1 of gamma.")
    ("gamma_2", po::value(&parameters.hyperparameters.gamma_2)->default_value(parameters.hyperparameters.gamma_2),
     "Prior 2 of gamma.")
    ("lambda_1", po::value(&parameters.hyperparameters.lambda_1)->default_value(parameters.hyperparameters.lambda_1),
     "Prior 1 of lambda.")
    ("lambda_2", po::value(&parameters.hyperparameters.lambda_2)->default_value(parameters.hyperparameters.lambda_2),
     "Prior 2 of lambda.")
    ("rho_1", po::value(&parameters.hyperparameters.rho_1)->default_value(parameters.hyperparameters.rho_1),
     "Prior 1 of rho.")
    ("rho_2", po::value(&parameters.hyperparameters.rho_2)->default_value(parameters.hyperparameters.rho_2),
     "Prior 2 of rho.")
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
     "Second prior for the baseline features.")
    ("normal_1", po::value(&parameters.hyperparameters.normal_1)->default_value(parameters.hyperparameters.normal_1),
     "Exponential of the mean of log normal distributions.")
    ("normalrho_2", po::value(&parameters.hyperparameters.normal_2)->default_value(parameters.hyperparameters.normal_2),
     "Variance of the log normal distribution.");

  inference_options.add_options()
    ("MHiter", po::value(&parameters.n_iter)->default_value(parameters.n_iter),
     "Maximal number of propositions for Metropolis-Hastings sampling of r")
    ("MHtemp", po::value(&parameters.temperature)->default_value(parameters.temperature),
     "Temperature for Metropolis-Hastings sampling of R.");

  cli_options.add(generic_options)
      .add(required_options)
      .add(basic_options)
      .add(gaussian_process_options)
      .add(advanced_options)
      .add(grad_options)
      .add(lbfgs_options)
      .add(rprop_options)
      .add(hmc_options)
      .add(inference_options)
      .add(hyperparameter_options);

  po::positional_options_description positional_options;
  positional_options.add("file", -1);

  ExecutionInformation exec_info;
  int ret_val
      = process_cli_options(argc, const_cast<const char **>(argv), exec_info,
                            usage_info, cli_options, true, positional_options);
  if (ret_val != PROCESSING_SUCCESSFUL)
    return ret_val;

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

  ifstream ifs(options.design_path);
  options.design.from_stream(ifs);

  for (auto &path : options.tsv_paths)
    options.design.add_dataset_specification(path);

  options.design.add_covariate_section();
  options.design.add_covariate_coordsys();
  options.design.add_covariate_unit();

  LOG(verbose) << "Design: " << options.design;

  vector<string> paths;
  for (auto &spec : options.design.dataset_specifications) {
    LOG(debug) << "Adding path " << spec.path;
    paths.push_back(spec.path);
  }

  if (paths.empty()) {
    LOG(fatal) << "Error: No input files specified.\n"
                  "You must either give paths to one or more count matrix "
                  "files or use the --design switch.";
    return EXIT_FAILURE;
  }

  auto data_sets
      = load_data(paths, options.intersect, options.top, options.bottom,
                  not options.keep_empty, options.transpose);

  LOG(verbose) << "Rate regression formula = " << parameters.rate_formula;
  LOG(verbose) << "Variance regression formula = " << parameters.variance_formula;

  LOG(verbose) << "gp = " << parameters.gp.use;
  LOG(verbose) << "gp.length_scale = " << parameters.gp.length_scale;
  LOG(verbose) << "gp.spatial_variance = " << parameters.gp.spatial_variance;
  LOG(verbose) << "gp.indep_variance = " << parameters.gp.independent_variance;

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
