#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <boost/filesystem.hpp>
#include "cli.hpp"
#include "PoissonFactorAnalysis.hpp"
#include "io.hpp"

using namespace std;
using PFA = PoissonFactorAnalysis;

const string default_output_string = "THIS PATH SHOULD NOT EXIST";

struct Options {
  vector<string> paths;
  Verbosity verbosity = Verbosity::Verbose;
  size_t num_factors = 20;
  size_t num_steps = 2000;
  size_t report_interval = 20;
  string output = default_output_string;
};

void write_resuls(const PFA &pfa, const string &prefix) {
  write_matrix(pfa.phi, prefix + "phi.txt");
  write_matrix(pfa.theta, prefix + "theta.txt");
  write_vector(pfa.r, prefix + "r.txt");
  write_vector(pfa.p, prefix + "p.txt");
}

void perform_metropolis_hastings(const PFA::IMatrix &counts, PFA &pfa,
                                 const Options &options) {
  MCMC::Evaluator<PFA> evaluator(counts);
  MCMC::Generator<PFA> generator(counts);
  MCMC::MonteCarlo<PFA> mc(generator, evaluator, options.verbosity);

  double temperature = 10.0;
  double anneal = 1.0;

  auto res = mc.run(temperature, anneal, pfa, options.num_steps);
  write_resuls(res.rbegin()->first, options.output);
}

void perform_gibbs_sampling(const PFA::IMatrix &counts, PFA &pfa,
                            const Options &options) {
  if (options.verbosity >= Verbosity::Info)
    cout << "Initial model" << endl << pfa << endl;
  if (options.verbosity >= Verbosity::Debug)
    cout << "Log-likelihood = " << pfa.log_likelihood(counts) << endl;
  for (size_t iteration = 1; iteration <= options.num_steps; ++iteration) {
    if (options.verbosity >= Verbosity::Info)
      cout << "Performing iteration " << iteration << endl;
    pfa.gibbs_sample(counts);
    if (options.verbosity >= Verbosity::Info)
      cout << "Current model" << endl << pfa << endl;
    if (iteration % options.report_interval == 0 and
        options.verbosity >= Verbosity::Info and
        options.verbosity < Verbosity::Debug)
      cout << "Log-likelihood = " << pfa.log_likelihood(counts) << endl;
  }
  if (options.verbosity >= Verbosity::Info)
    cout << "Final log-likelihood = " << pfa.log_likelihood(counts) << endl;
  write_resuls(pfa, options.output);
}

int main(int argc, char **argv) {
  // TODO allow multiple samples - in that case take the conjunction of genes
  // TODO when reading in count tables, store the row and column labels, and use
  // them when printing out the results

  EntropySource::seed();
  MCMC::EntropySource::seed();

  Options options;

  string config_path;
  string usage_info = "How to use this software";

  size_t num_cols = 80;
  namespace po = boost::program_options;
  po::options_description cli_options;
  po::options_description generic_options =
      gen_generic_options(config_path, num_cols);

  po::options_description required_options("Required options", num_cols);
  po::options_description basic_options("Basic options", num_cols);

  required_options.add_options()
    ("file,f", po::value(&options.paths)->required(),
     "Path to a count matrix file, can be given multiple times. "
     "Format: tab-separated, including a header line, "
     "and row names in the first column of each row.");

  basic_options.add_options()
    ("types,t", po::value(&options.num_factors)->default_value(options.num_factors),
     "Maximal number of cell types to look for.")
    ("iter,i", po::value(&options.num_steps)->default_value(options.num_steps),
     "Number of iterations to perform.")
    ("report,r", po::value(&options.report_interval)->default_value(options.report_interval),
     "Interval for computing and printing the likelihood.")
    ("output,o", po::value(&options.output),
     "Prefix for generated output files.");

  cli_options.add(generic_options).add(required_options).add(basic_options);

  po::positional_options_description positional_options;
  positional_options.add("file", -1);

  ExecutionInformation exec_info = process_cli_options(
      argc, const_cast<const char **>(argv), options.verbosity, usage_info,
      cli_options, true, positional_options);

  if(options.output == default_output_string) {
    options.output = generate_random_label(exec_info.program_name, 0, options.verbosity) + "/";
    if(not boost::filesystem::create_directory(options.output)) {
      cout << "Error creating output directory " << options.output << endl;
      exit(EXIT_FAILURE);
    }
  }

  auto counts = read_matrix(options.paths[0]);

  PFA::Priors priors;
  PoissonFactorAnalysis pfa(counts, options.num_factors, priors, options.verbosity);

  if (0)
    perform_metropolis_hastings(counts, pfa, options);
  else
    perform_gibbs_sampling(counts, pfa, options);

  return EXIT_SUCCESS;
}
