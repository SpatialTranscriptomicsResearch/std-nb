#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>
#include <vector>
#include <boost/filesystem.hpp>
#include "aux.hpp"
#include "cli.hpp"
#include "counts.hpp"
#include "entropy.hpp"
#include "io.hpp"
#include "log.hpp"
// #include "Model.hpp"
#include "hdp.hpp"
#include "hierarchical_kmeans.hpp"

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <fenv.h>

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
  size_t num_factors = 1000;
  size_t num_steps = 2000;
  size_t burn_in = 100;
  size_t report_interval = 20;
  string output = default_output_string;
  bool intersect = false;
  bool kmeans = false;
  Labeling labeling = Labeling::Auto;
  bool compute_likelihood = false;
  bool perform_splitmerge = false;
  bool posterior_switches = false;
  size_t top = 0;
  // PF::Target sample_these = PF::DefaultTarget();
  // PF::Partial::Kind feature_type = PF::Partial::Kind::Gamma;
  // PF::Partial::Kind mixing_type = PF::Partial::Kind::HierGamma;
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

ostream &print(ostream &os, const PF::Matrix &m,
           const vector<string> &row_names = vector<string>(),
           const vector<string> &col_names = vector<string>()) {
  for (auto name : col_names)
    os << "\t" << name;
  os << endl;
  for (size_t i = 0; i < m.n_rows; ++i) {
    os << row_names[i];
    for (size_t j = 0; j < m.n_cols; ++j)
      os << "\t" << m(i, j);
    os << endl;
  }
  return os;
}

std::pair<size_t, size_t> draw_read(const Counts &data,
                                    vector<size_t> colSums) {
  size_t s = std::discrete_distribution<size_t>(
      begin(colSums), end(colSums))(EntropySource::rng);
  size_t g = std::discrete_distribution<size_t>(
      data.counts.begin_col(s), data.counts.end_col(s))(EntropySource::rng);
  return make_pair(g, s);
}

void store(const PF::HDP &model, const Counts &data, const string &prefix = "",
           const string &suffix = "") {
  vector<string> type_names;
  for (size_t t = 0; t < model.T; ++t)
    type_names.push_back("Factor " + to_string(t + 1));
  string stem = prefix + "hdp" + suffix;
  if (model.T > 0) {
    ofstream os(stem + "-features.txt");
    print(os, model.counts_gene_type.cols(0, model.T - 1), data.row_names,
          type_names);
    os = ofstream(stem + "-mix.txt");
    print(os, model.counts_spot_type.cols(0, model.T - 1), data.col_names,
          type_names);
  } else
    LOG(info) << "Warning: 0 factors!";
}

int main(int argc, char **argv) {
  EntropySource::seed();

  Options options;

  PF::HDP::Parameters parameters;

  string config_path;
  string usage_info = "This software implements the nested hierarchical Dirichlet process model of\n"
    "Nested Hierarchical Dirichlet Processes\n"
    "by John Paisley, Chong Wang, David M. Blei and Michael I. Jordan\n"
    "Proceedings of the 15th International Conference on Artificial Intelligence and\n"
    "Journal of Pattern Analysis and Machine Intelligence. Volume XX.\n"
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
  po::options_description parameter_options("Hyper-parameter options", num_cols);

  required_options.add_options()
    ("file", po::value(&options.tsv_paths)->required(),
     "Path to a count matrix file, can be given multiple times. "
     "Format: tab-separated, including a header line, "
     "and row names in the first column of each row.");

  basic_options.add_options()
    // ("feature,f", po::value(&options.feature_type)->default_value(options.feature_type),
     // "Which type of distribution to use for the features. "
     // "Can be one of 'gamma' or 'dirichlet'.")
    // ("mix,m", po::value(&options.mixing_type)->default_value(options.mixing_type),
     // "Which type of distribution to use for the mixing weights. "
     // "Can be one of 'gamma' or 'dirichlet'.")
    ("types,t", po::value(&options.num_factors)->default_value(options.num_factors),
     "Maximal number of cell types to look for.")
    ("iter,i", po::value(&options.num_steps)->default_value(options.num_steps),
     "Number of iterations to perform.")
    ("burn,b", po::value(&options.burn_in)->default_value(options.burn_in),
     "Number of iterations to discard before accumulating statistics.")
    ("report,r", po::value(&options.report_interval)->default_value(options.report_interval),
     "Interval for reporting the parameters.")
    ("nolikel", po::bool_switch(&options.compute_likelihood),
     "Do not compute and print the likelihood every iteration.")
    ("split,s", po::bool_switch(&options.perform_splitmerge),
     "Perform split/merge steps.")
    ("kmeans,k", po::bool_switch(&options.kmeans),
     "Perform hierarchical K-means for initialization.")
    ("switches", po::bool_switch(&options.posterior_switches),
     "Don't sample activation switches independently but from the posterior.")
    ("output,o", po::value(&options.output),
     "Prefix for generated output files.")
    ("top", po::value(&options.top)->default_value(options.top),
     "Use only those genes with the highest read count across all spots. Zero indicates all genes.")
    ("intersect", po::bool_switch(&options.intersect),
     "When using multiple count matrices, use the intersection of rows, rather than their union.")
    ("label", po::value(&options.labeling),
     "How to label the spots. Can be one of 'alpha', 'path', 'none'. If only one count table is given, the default is to use 'none'. If more than one is given, the default is 'alpha'.");

  parameter_options.add_options()
    ("feat_a", po::value(&parameters.feature_alpha)->default_value(parameters.feature_alpha),
     "Feature alpha.")
    ("mix_a", po::value(&parameters.mix_alpha)->default_value(parameters.mix_alpha),
     "Mixture alpha.")
    ("mix_a0", po::value(&parameters.mix_alpha_zero)->default_value(parameters.mix_alpha_zero),
     "Mixture alpha zero.");

  cli_options.add(generic_options)
      .add(required_options)
      .add(basic_options)
      .add(parameter_options)
      ;

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

  Counts data(options.tsv_paths[0], labels[0]);
  for (size_t i = 1; i < options.tsv_paths.size(); ++i)
    if (options.intersect)
      data = data * Counts(options.tsv_paths[i], labels[i]);
    else
      data = data + Counts(options.tsv_paths[i], labels[i]);

  if (options.top > 0)
    data.select_top(options.top);

  PF::HDP model(data.counts.n_rows, data.counts.n_cols, options.num_factors,
                parameters);

  feenableexcept(FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW);
  // feenableexcept(FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW | FE_UNDERFLOW);

  if (true) {
    PF::HDP cumul_model = model;
    for (size_t i = 0; i < options.num_steps; ++i) {
      LOG(info) << "Iteration " << i;
      if (i >= options.burn_in)
        cumul_model += model;
      model = model.sample(data.counts);
      if (i > 0 and i % options.report_interval == 0)
        store(model, data, options.output, "-iter" + to_string(i));
      if (i > 0 and i % (options.report_interval * 10) == 0)
        store(cumul_model, data, options.output, "-cumul-iter" + to_string(i));
    }
    store(cumul_model, data, options.output, "-cumul-final");
  } else {
    vector<size_t> colSums(model.S, 0);
    for (size_t s = 0; s < model.S; ++s)
      for (size_t g = 0; g < model.G; ++g)
        colSums[s] += data.counts(g, s);

    for (size_t i = 0; i < options.num_steps; ++i) {
      auto read = draw_read(data, colSums);
      LOG(info) << "Iteration " << i << " gene " << data.row_names[read.first]
                << " spot " << data.col_names[read.second];
      LOG(info) << "G = " << model.G << " S = " << model.S
                << " T = " << model.T;
      model.register_read(read.first, read.second);
      if (i > 0 and i % 100000 == 0)
        store(model, data, options.output, "-iter" + to_string(i));
    }
  }

  store(model, data, options.output, "-final");

  return EXIT_SUCCESS;
}
