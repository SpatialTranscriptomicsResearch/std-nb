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
#include "montecarlo.hpp"
#include "VariantModel.hpp"

using namespace std;
namespace PF = PoissonFactorization;

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
  vector<string> tsv_paths;
  string simulate_path = "";
  Verbosity verbosity = Verbosity::Info;
  size_t num_factors = 20;
  size_t num_steps = 2000;
  size_t report_interval = 20;
  string output = default_output_string;
  bool intersect = false;
  Labeling labeling = Labeling::Auto;
  bool compute_likelihood = false;
  bool timing = true;
  size_t top = 0;
  bool phi_dirichlet = false;
  PF::Target sample_these = PF::DefaultTarget();
  PF::Feature::Kind feature_type = PF::Feature::Kind::Gamma;
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
void simulate(const T &pfa, const Counts &counts) {
  // sample the highest, the median, and the lowest genes' counts for all spots
  vector<size_t> counts_per_gene(pfa.G, 0);
  for (size_t g = 0; g < pfa.G; ++g)
    for (size_t s = 0; s < pfa.S; ++s)
      counts_per_gene[g] += counts.counts(g, s);
  vector<pair<size_t, size_t>> v;
  for (size_t g = 0; g < pfa.G; ++g)
    v.push_back(pair<size_t, size_t>(counts_per_gene[g], g));
  sort(begin(v), end(v));
  vector<size_t> to_sample;
  to_sample.push_back(v.rbegin()->second);
  to_sample.push_back((v.begin() + pfa.G / 2)->second);
  to_sample.push_back(v.begin()->second);
  for (auto g : to_sample)
    for (size_t s = 0; s < pfa.S; ++s) {
      cout << "SAMPLE ACROSS SPOTS\t" << counts.row_names[g] << "\t"
           << counts.col_names[s];
      for (auto x : pfa.sample_reads(g, s, 10000))
        cout << "\t" << x;
      cout << endl;
    }

  // for spot with the highest number of reads, sample all genes' counts
  PF::Int max_count = 0;
  size_t max_idx = 0;
  for (size_t s = 0; s < pfa.S; ++s) {
    PF::Int count = 0;
    for (size_t g = 0; g < pfa.G; ++g)
      count += counts.counts(g, s);
    if (count > max_count) {
      max_count = count;
      max_idx = s;
    }
  }
  for (size_t g = 0; g < pfa.G; ++g) {
    cout << "SAMPLE ACROSS GENES\t" << counts.row_names[g] << "\t"
         << counts.col_names[max_idx];
    for (auto x : pfa.sample_reads(g, max_idx, 10000))
      cout << "\t" << x;
    cout << endl;
  }
}

template <typename T>
void perform_gibbs_sampling(const Counts &data, T &pfa,
                            const Options &options) {
  if (options.verbosity >= Verbosity::Info)
    cout << "Initial model" << endl << pfa << endl;
  if (options.compute_likelihood and options.verbosity >= Verbosity::Info)
    cout << "Log-likelihood = " << pfa.log_likelihood_poisson_counts(data.counts) << endl;
  for (size_t iteration = 1; iteration <= options.num_steps; ++iteration) {
    if (iteration > pfa.parameters.enforce_iter)
      pfa.parameters.enforce_mean = PF::ForceMean::None;
    if (options.verbosity >= Verbosity::Info)
      cout << "Performing iteration " << iteration << endl;
    pfa.gibbs_sample(data, options.sample_these, options.timing);
    if (options.verbosity >= Verbosity::Info)
      cout << "Current model" << endl << pfa << endl;
    if (iteration % options.report_interval == 0)
      pfa.store(data, options.output + "iter" + to_string(iteration) + "_");
    if (options.compute_likelihood and options.verbosity >= Verbosity::Info)
      cout << "Log-likelihood = " << pfa.log_likelihood_poisson_counts(data.counts) << endl;
  }
  if (options.compute_likelihood and options.verbosity >= Verbosity::Info)
    cout << "Final log-likelihood = " << pfa.log_likelihood_poisson_counts(data.counts)
         << endl;
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
    ("file,f", po::value(&options.tsv_paths)->required(),
     "Path to a count matrix file, can be given multiple times. "
     "Format: tab-separated, including a header line, "
     "and row names in the first column of each row.");

  basic_options.add_options()
    ("dirichlet,d", po::bool_switch(&options.phi_dirichlet),
     "Use a Dirichlet distribution for the features.")
    ("simulate,s", po::value(&options.simulate_path),
     "Prefix to a set of parameter files. ")
    ("types,t", po::value(&options.num_factors)->default_value(options.num_factors),
     "Maximal number of cell types to look for.")
    ("iter,i", po::value(&options.num_steps)->default_value(options.num_steps),
     "Number of iterations to perform.")
    ("report,r", po::value(&options.report_interval)->default_value(options.report_interval),
     "Interval for reporting the parameters.")
    ("nolikel", po::bool_switch(&options.compute_likelihood),
     "Do not compute and print the likelihood every iteration.")
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
  int ret_val = process_cli_options(argc, const_cast<const char **>(argv),
                                    options.verbosity, exec_info, usage_info,
                                    cli_options, true, positional_options);
  if (ret_val != PROCESSING_SUCCESSFUL)
    return ret_val;

  // invert the negative CLI switch value
  options.compute_likelihood = !options.compute_likelihood;

  if (options.verbosity >= Verbosity::Info) {
    cout << exec_info.name_and_version() << endl;
    cout << exec_info.datetime << endl;
    cout << "Working directory = " << exec_info.directory << endl;
    cout << "Command =" << exec_info.cmdline << endl;
    cout << endl;
  }

  if (options.output == default_output_string) {
    options.output
        = generate_random_label(exec_info.program_name, 0, options.verbosity)
          + "/";
  }
  if (not options.output.empty() and *options.output.rbegin() == '/')
    if (not boost::filesystem::create_directory(options.output)) {
      cout << "Error creating output directory " << options.output << endl;
      return EXIT_FAILURE;
    }

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
    cout << "Warning: too few labels available! Using paths as labels." << endl;
    labels = options.tsv_paths;
  }

  Counts data(options.tsv_paths[0], labels[0]);
  for (size_t i = 1; i < options.tsv_paths.size(); ++i)
    if (options.intersect)
      data = data * Counts(options.tsv_paths[i], labels[i]);
    else
      data = data + Counts(options.tsv_paths[i], labels[i]);

  if (options.top > 0)
    data.select_top(options.top);

  if (options.simulate_path != "") {
    PF::Paths paths(options.simulate_path, "");
    PF::Model<PF::Feature::Kind::Gamma, PF::Mix::Kind::Gamma> pfa(
        data, paths, parameters, options.verbosity);
    simulate(pfa, data);
  } else {
    if (options.phi_dirichlet)
      options.feature_type = PF::Feature::Kind::Dirichlet;
    if (options.feature_type == PF::Feature::Kind::Dirichlet) {
      if (options.verbosity >= Verbosity::Info)
        cout << "Using Dirichlet-distribution for the features." << endl;
      PF::Model<PF::Feature::Kind::Dirichlet, PF::Mix::Kind::Gamma> pfa(
          data, options.num_factors, parameters, options.verbosity);
      perform_gibbs_sampling(data, pfa, options);
    } else if (options.feature_type == PF::Feature::Kind::Gamma) {
      if (options.verbosity >= Verbosity::Info)
        cout << "Using gamma-distribution for the features." << endl;
      PF::Model<PF::Feature::Kind::Gamma, PF::Mix::Kind::Gamma> pfa(
          data, options.num_factors, parameters, options.verbosity);
      perform_gibbs_sampling(data, pfa, options);
    }
  }

  return EXIT_SUCCESS;
}
