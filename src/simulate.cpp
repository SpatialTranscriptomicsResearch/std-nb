#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>
#include <vector>
#include <boost/filesystem.hpp>
#include "cli.hpp"
#include "counts.hpp"
#include "io.hpp"
#include "aux.hpp"
#include "montecarlo.hpp"
#include "PoissonModel.hpp"
#include "VariantModel.hpp"

using namespace std;

const string default_output_string = "THIS PATH SHOULD NOT EXIST";

struct Options {
  struct Paths {
    string r = "r.txt";
    string p = "p.txt";
    string theta = "theta.txt";
    string phi = "phi.txt";
    string spot_scaling = "spot_scaling.txt";
    string experiment_scaling = "experiment_scaling.txt";
  };

  string dir;
  Paths paths;

  Verbosity verbosity = Verbosity::Info;
  size_t num_samples = 20;
  string output = default_output_string;
  bool timing = true;
};

void write_results(const FactorAnalysis::PoissonModel &pfa, const Counts &counts, const string &prefix) {
  vector<string> factor_names;
  for (size_t t = 1; t <= pfa.T; ++t)
    factor_names.push_back("Factor " + to_string(t));
  write_matrix(pfa.phi, prefix + "phi.txt", counts.row_names, factor_names);
  write_matrix(pfa.theta, prefix + "theta.txt", counts.col_names, factor_names);
  write_vector(pfa.r, prefix + "r.txt", factor_names);
  write_vector(pfa.p, prefix + "p.txt", factor_names);
}

void write_results(const FactorAnalysis::VariantModel &pfa,
                   const Counts &counts, const string &prefix,
                   bool do_sample = false) {
  vector<string> factor_names;
  for (size_t t = 1; t <= pfa.T; ++t)
    factor_names.push_back("Factor " + to_string(t));
  write_matrix(pfa.phi, prefix + "phi.txt", counts.row_names, factor_names);
  write_matrix(pfa.r, prefix + "r.txt", counts.row_names, factor_names);
  write_matrix(pfa.p, prefix + "p.txt", counts.row_names, factor_names);
  write_matrix(pfa.theta, prefix + "theta.txt", counts.col_names, factor_names);
  write_vector(pfa.spot_scaling, prefix + "spot_scaling.txt", counts.col_names);
  write_vector(pfa.experiment_scaling, prefix + "experiment_scaling.txt", counts.experiment_names);
  if (do_sample)
    for (size_t g = 0; g < 1; g++)
      for (size_t s = 0; s < pfa.S; ++s) {
        cout << "SAMPLE\t" << counts.row_names[g] << "\t"
             << counts.col_names[s];
        for (auto x : pfa.sample_reads(g, s, 10000))
          cout << "\t" << x;
        cout << endl;
      }
}

int main(int argc, char **argv) {
  EntropySource::seed();

  Options options;

  FactorAnalysis::Priors priors;
  FactorAnalysis::Parameters parameters;

  string config_path;
  string usage_info = "This software implements the βγΓ-Poisson Factor Analysis model of\n"
    "Beta-Negative Binomial Process and Poisson Factor Analysis\n"
    "by Mingyuan Zhou, Lauren A. Hannah, David B. Dunson, and Lawrence Carin\n"
    "Proceedings of the 15th International Conference on Artificial Intelligence and\n"
    "Statistics (AISTATS) 2012, La Palma, Canary Islands. Volume XX of JMLR: W&CP XX.\n"
    "\n"
    "Please provide the path to a directory with the parameter files as argument to\n"
    "the --dir switch or as free arguments. Alternatively, you can selectively over-"
    "ride paths to these files individually.";

  size_t num_cols = 80;
  namespace po = boost::program_options;
  po::options_description cli_options;
  po::options_description generic_options =
      gen_generic_options(config_path, num_cols);

  po::options_description required_options("Required options", num_cols);
  po::options_description basic_options("Basic options", num_cols);
  po::options_description prior_options("Prior options", num_cols);

  required_options.add_options()
    ("dir,d", po::value(&options.dir)->required(),
     "Directory in which the parameter files are found. "
     "By default, files names 'phi.txt', 'r.txt', p.txt', 'theta.txt', spot_scaling.txt', and 'experiment_scaling.txt' are loaded, if they are not overriden.");

  basic_options.add_options()
    ("num,n", po::value(&options.num_samples),
     "Prefix for generated output files.")
    ("output,o", po::value(&options.output),
     "Prefix for generated output files.")
    (",r", po::value(&options.paths.r)->default_value(options.paths.r),
     "Path to parameter for r.")
    (",p", po::value(&options.paths.p)->default_value(options.paths.p),
     "Path to parameter for p.")
    ("phi", po::value(&options.paths.phi)->default_value(options.paths.phi),
     "Path to parameter for theta.")
    ("theta,t", po::value(&options.paths.theta)->default_value(options.paths.theta),
     "Path to parameter for theta.")
    ("spot_scale,s", po::value(&options.paths.spot_scaling)->default_value(options.paths.spot_scaling),
     "Path to parameter for spot scaling.")
    ("exp_scale,e", po::value(&options.paths.experiment_scaling)->default_value(options.paths.experiment_scaling),
     "Path to parameter for experiment scaling.")
    ("timing", po::bool_switch(&options.timing),
     "Print out timing information.");

  prior_options.add_options()
    ("alpha", po::value(&priors.alpha)->default_value(priors.alpha),
     "Dirichlet prior alpha of the factor loading matrix.")
    ("phi_r_1", po::value(&priors.phi_r_1)->default_value(priors.phi_r_1),
     "Gamma prior 1 of r[g][t].")
    ("phi_r_2", po::value(&priors.phi_r_2)->default_value(priors.phi_r_2),
     "Gamma prior 2 of r[g][t].")
    ("phi_p_1", po::value(&priors.phi_p_1)->default_value(priors.phi_p_1),
     "Gamma prior 1 of p[g][t].")
    ("phi_p_2", po::value(&priors.phi_p_2)->default_value(priors.phi_p_2),
     "Gamma prior 2 of p[g][t].");

  cli_options.add(generic_options)
      .add(required_options)
      .add(basic_options)
      .add(prior_options);

  po::positional_options_description positional_options;
  positional_options.add("dir", -1);

  ExecutionInformation exec_info;
  int ret_val = process_cli_options(argc, const_cast<const char **>(argv),
                                    options.verbosity, exec_info, usage_info,
                                    cli_options, true, positional_options);
  if (ret_val != PROCESSING_SUCCESSFUL)
    return ret_val;

  if(options.verbosity >= Verbosity::Verbose)
    cout << exec_info.name_and_version() << endl;

  if (options.output == default_output_string) {
    options.output =
        generate_random_label(exec_info.program_name, 0, options.verbosity) +
        "/";
  }
  if(not options.output.empty() and *options.output.rbegin() == '/')
    if (not boost::filesystem::create_directory(options.output)) {
      cout << "Error creating output directory " << options.output << endl;
      return EXIT_FAILURE;
    }

  FactorAnalysis::VariantModel pfa(
      options.paths.phi, options.paths.theta, options.paths.spot_scaling,
      options.paths.experiment_scaling, options.paths.r,
      options.paths.p, priors, parameters, options.verbosity);

  // TODO

  return EXIT_SUCCESS;
}
