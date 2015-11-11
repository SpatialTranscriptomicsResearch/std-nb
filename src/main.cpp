#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include "PoissonFactorAnalysis.hpp"
#include "io.hpp"

using namespace std;
using PFA = PoissonFactorAnalysis;

void write_resuls(const PFA &pfa) {
  write_matrix(pfa.phi, "phi.txt");
  write_matrix(pfa.theta, "theta.txt");
  write_vector(pfa.r, "r.txt");
  write_vector(pfa.p, "p.txt");
}

void perform_metropolis_hastings(const PFA::IMatrix &counts, PFA &pfa,
                                 size_t num_steps, Verbosity verbosity) {
  MCMC::Evaluator<PFA> evaluator(counts);
  MCMC::Generator<PFA> generator(counts);
  MCMC::MonteCarlo<PFA> mc(generator, evaluator, verbosity);

  double temperature = 10.0;
  double anneal = 1.0;

  auto res = mc.run(temperature, anneal, pfa, num_steps);
  write_resuls(res.rbegin()->first);
}

void perform_gibbs_sampling(const PFA::IMatrix &counts, PFA &pfa,
                            size_t num_steps, Verbosity verbosity) {
  for (size_t iteration = 1; iteration <= num_steps; ++iteration) {
    if (verbosity >= Verbosity::Info)
      cout << "Performing iteration " << iteration << endl;
    pfa.sample_contributions(counts);
    pfa.sample_phi();
    pfa.sample_p();
    pfa.sample_r();
    pfa.sample_theta();
    if (iteration % 20 == 0)
      cout << "Log-likelihood = " << pfa.log_likelihood(counts) << endl;
  }
  cout << "Final log-likelihood = " << pfa.log_likelihood(counts) << endl;
  write_resuls(pfa);
}

int main(int argc, char **argv) {
  if (argc != 2) {
    cerr << "Please specify the path to the count matrix." << endl;
    return EXIT_FAILURE;
  }
  string path = argv[1];

  EntropySource::seed();
  MCMC::EntropySource::seed();

  auto counts = read_matrix(path);

  Verbosity verbosity = Verbosity::Verbose;

  size_t K = 10;
  PFA::Priors priors;
  PoissonFactorAnalysis pfa(counts, K, priors, verbosity);

  size_t num_steps = 1000;

  if (0)
    perform_metropolis_hastings(counts, pfa, num_steps, verbosity);
  else
    perform_gibbs_sampling(counts, pfa, num_steps, verbosity);

  return EXIT_SUCCESS;
}
