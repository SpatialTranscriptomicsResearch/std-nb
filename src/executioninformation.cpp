
#include <cstdio>
#include <ctime>
#include <boost/filesystem.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include "executioninformation.hpp"

using namespace std;

string reconstitute_cmdline(int argc, const char **argv) {
  string cmd;
  for (int i = 0; i < argc; i++) cmd += (i != 0 ? " " : "") + string(argv[i]);
  return cmd;
}

ExecutionInformation::ExecutionInformation()
    : ExecutionInformation("program_name", "unknown", "unknown", "") {}

ExecutionInformation::ExecutionInformation(const string &name,
                                           const string &version,
                                           const string &branch, int argc,
                                           const char **argv)
    : ExecutionInformation(name, version, branch,
                           reconstitute_cmdline(argc, argv)) {}

ExecutionInformation::ExecutionInformation(const string &name,
                                           const string &version,
                                           const string &branch,
                                           const string &cmd)
    : program_name(boost::filesystem::path(name).filename().string()),
      program_version(version),
      git_branch(branch),
      cmdline(cmd),
      datetime(),
      directory(boost::filesystem::initial_path().string()) {
  time_t rawtime;
  time(&rawtime);
  datetime = ctime(&rawtime);
  datetime = datetime.substr(0, datetime.size() - 1);
}

string generate_random_label(const string &prefix, size_t n_rnd_char,
                             Verbosity verbosity) {
  random_device rng;
  uniform_int_distribution<char> r_char('a', 'z');
  using namespace boost::posix_time;

  string label = prefix;

  // TODO perhaps add the process ID -> getpid()
  if (verbosity >= Verbosity::Debug)
    cout << "Generating random label with prefix " << prefix << " and "
         << n_rnd_char << " random characters." << endl;

  try {
    ptime t = microsec_clock::universal_time();
    string datetime = to_iso_extended_string(t) + "Z";
    label += "_" + datetime;
  } catch (...) {
    cout << "WARNING: An error occurred while generating the date part of a "
            "label." << endl << "Although this shouldn't occur (something "
                                "weird is going on with your system!)," << endl
         << "you can likely circumvent this issue by using the --output "
            "command line switch)" << endl << "to provide you own output label."
         << endl;
    if (n_rnd_char < 5) n_rnd_char = 5;
  }

  if (n_rnd_char > 0) {
    label += "_";
    for (size_t i = 0; i < n_rnd_char; i++) label += r_char(rng);
  }

  if (verbosity >= Verbosity::Debug)
    cout << "Generated random label " << label << endl;

  return label;
}
