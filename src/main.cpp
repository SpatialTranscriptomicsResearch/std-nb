#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>
#include <vector>
#include <boost/filesystem.hpp>
#include "cli.hpp"
#include "io.hpp"
#include "aux.hpp"
#include "montecarlo.hpp"
#include "PoissonModel.hpp"
#include "VariantModel.hpp"

using namespace std;

const string default_output_string = "THIS PATH SHOULD NOT EXIST";

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
  vector<string> paths;
  Verbosity verbosity = Verbosity::Info;
  size_t num_factors = 20;
  size_t num_steps = 2000;
  size_t report_interval = 20;
  string output = default_output_string;
  bool intersect = false;
  bool original_model = false;
  Labeling labeling = Labeling::Auto;
  bool compute_likelihood = false;
  bool timing = true;
};

istream &operator>>(istream &is, Options::Labeling &label) {
  string token;
  is >> token;
  if (to_lower(token) == "auto")
    label = Options::Labeling::Auto;
  else if (to_lower(token) == "none")
    label = Options::Labeling::None;
  else if (to_lower(token) == "path")
    label = Options::Labeling::Path;
  else if (to_lower(token) == "alpha")
    label = Options::Labeling::Alpha;
  else
    throw std::runtime_error("Error: could not parse labeling '" + token +
                             "'.");
  return is;
}

void write_results(const FactorAnalysis::PoissonModel &pfa, const Counts &counts, const string &prefix) {
  vector<string> factor_names;
  for (size_t t = 1; t <= pfa.T; ++t)
    factor_names.push_back("Factor " + to_string(t));
  write_matrix(pfa.phi, prefix + "phi.txt", counts.row_names, factor_names);
  write_matrix(pfa.theta, prefix + "theta.txt", counts.col_names, factor_names);
  write_vector(pfa.r, prefix + "r.txt", factor_names);
  write_vector(pfa.p, prefix + "p.txt", factor_names);
}


void write_results(const FactorAnalysis::VariantModel &pfa, const Counts &counts, const string &prefix) {
  vector<string> factor_names;
  for (size_t t = 1; t <= pfa.T; ++t)
    factor_names.push_back("Factor " + to_string(t));
  write_matrix(pfa.phi, prefix + "phi.txt", counts.row_names, factor_names);
  write_matrix(pfa.r, prefix + "r.txt", counts.row_names, factor_names);
  write_matrix(pfa.p, prefix + "p.txt", counts.row_names, factor_names);
  write_matrix(pfa.theta, prefix + "theta.txt", counts.col_names, factor_names);
  write_vector(pfa.scaling, prefix + "scaling.txt", counts.col_names);
}

template <typename T>
void perform_gibbs_sampling(const Counts &data, T &pfa,
                            const Options &options) {
  if (options.verbosity >= Verbosity::Info)
    cout << "Initial model" << endl << pfa << endl;
  if (options.compute_likelihood and options.verbosity >= Verbosity::Info)
    cout << "Log-likelihood = " << pfa.log_likelihood(data.counts) << endl;
  for (size_t iteration = 1; iteration <= options.num_steps; ++iteration) {
    if (options.verbosity >= Verbosity::Info)
      cout << "Performing iteration " << iteration << endl;
    pfa.gibbs_sample(data.counts, options.timing);
    if (options.verbosity >= Verbosity::Info)
      cout << "Current model" << endl << pfa << endl;
    if (iteration % options.report_interval == 0) {
      if (options.compute_likelihood and options.verbosity >= Verbosity::Info)
        cout << "Log-likelihood = " << pfa.log_likelihood(data.counts) << endl;
      write_results(pfa, data,
                    options.output + "iter" + to_string(iteration) + "_");
    }
  }
  if (options.compute_likelihood and options.verbosity >= Verbosity::Info)
    cout << "Final log-likelihood = " << pfa.log_likelihood(data.counts)
         << endl;
  write_results(pfa, data, options.output);
}

int main(int argc, char **argv) {
  EntropySource::seed();
  MCMC::EntropySource::seed();

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
    "Please provide one or more count matrices as argument to the --file switch or as\n"
    "free arguments.";

  size_t num_cols = 80;
  namespace po = boost::program_options;
  po::options_description cli_options;
  po::options_description generic_options =
      gen_generic_options(config_path, num_cols);

  po::options_description required_options("Required options", num_cols);
  po::options_description basic_options("Basic options", num_cols);
  po::options_description prior_options("Prior options", num_cols);
  po::options_description inference_options("MCMC inference options", num_cols);

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
     "Interval for reporting the parameters.")
    ("likelihood", po::bool_switch(&options.compute_likelihood),
     "Compute and print the likelihood every time parameters are reported.")
    ("output,o", po::value(&options.output),
     "Prefix for generated output files.")
    ("original", po::bool_switch(&options.original_model),
     "Use the original model.")
    ("intersect", po::bool_switch(&options.intersect),
     "When using multiple count matrices, use the intersection of rows, rather than their union.")
    ("timing", po::bool_switch(&options.timing),
     "Print out timing information.")
    ("label", po::value(&options.labeling),
     "How to label the spots. Can be one of 'alpha', 'path', 'none'. If only one count table is given, the default is to use 'none'. If more than one is given, the default is 'alpha'.");

  prior_options.add_options()
    ("alpha", po::value(&priors.alpha)->default_value(priors.alpha),
     "Dirichlet prior alpha of the factor loading matrix.")
    ("beta_c", po::value(&priors.c)->default_value(priors.c),
     "Beta prior c.")
    ("beta_eps", po::value(&priors.epsilon)->default_value(priors.epsilon),
     "Beta prior epsilon.")
    ("gamma_c", po::value(&priors.c0)->default_value(priors.c0),
     "Gamma prior c0.")
    ("gamma_r", po::value(&priors.r0)->default_value(priors.r0),
     "Gamma prior r0.")
    ("gamma", po::value(&priors.gamma)->default_value(priors.gamma),
     "Prior gamma.");

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
      .add(prior_options)
      .add(inference_options);

  po::positional_options_description positional_options;
  positional_options.add("file", -1);

  ExecutionInformation exec_info = process_cli_options(
      argc, const_cast<const char **>(argv), options.verbosity, usage_info,
      cli_options, true, positional_options);

  if(options.verbosity >= Verbosity::Verbose)
    cout << exec_info.name_and_version() << endl;

  if (options.output == default_output_string) {
    options.output =
        generate_random_label(exec_info.program_name, 0, options.verbosity) +
        "/";
    if (not boost::filesystem::create_directory(options.output)) {
      cout << "Error creating output directory " << options.output << endl;
      exit(EXIT_FAILURE);
    }
  }

  vector<string> labels;
  switch (options.labeling) {
    case Options::Labeling::None:
      labels = vector<string>(options.paths.size(), "");
      break;
    case Options::Labeling::Path:
      labels = options.paths;
      break;
    case Options::Labeling::Alpha:
      labels = alphabetic_labels;
      break;
    case Options::Labeling::Auto:
      if (options.paths.size() > 1)
        labels = alphabetic_labels;
      else
        labels = vector<string>(options.paths.size(), "");
      break;
  }

  if (labels.size() < options.paths.size()) {
    cout << "Warning: too few labels available! Using paths as labels." << endl;
    labels = options.paths;
  }

  Counts data(options.paths[0], labels[0]);
  for (size_t i = 1; i < options.paths.size(); ++i)
    if (options.intersect)
      data = data * Counts(options.paths[i], labels[i]);
    else
      data = data + Counts(options.paths[i], labels[i]);

  if (options.original_model) {
    FactorAnalysis::PoissonModel pfa(data.counts, options.num_factors, priors,
                                     parameters, options.verbosity);

    perform_gibbs_sampling(data, pfa, options);
  } else {
    FactorAnalysis::VariantModel pfa(data.counts, options.num_factors, priors,
                                     parameters, options.verbosity);

    perform_gibbs_sampling(data, pfa, options);
  }

  return EXIT_SUCCESS;
}
