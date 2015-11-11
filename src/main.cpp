#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include "PoissonFactorAnalysis.hpp"
#include "io.hpp"

using namespace std;
using PFA = PoissonFactorAnalysis;

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

  MCMC::Evaluator<PFA> evaluator(counts);
  MCMC::Generator<PFA> generator(counts, verbosity);
  MCMC::MonteCarlo<PFA> mc(generator, evaluator, verbosity);

  size_t K = 10;
  PFA::Priors priors;
  PoissonFactorAnalysis pfa(counts, K, priors);

  double temperature = 10.0;
  double anneal = 1.0;
  size_t num_steps = 1000;

  auto res = mc.run(temperature, anneal, pfa, num_steps);

  write_matrix(res.rbegin()->first.phi, "phi.txt");
  write_matrix(res.rbegin()->first.theta, "theta.txt");
  write_vector(res.rbegin()->first.r, "r.txt");
  write_vector(res.rbegin()->first.p, "p.txt");

  return EXIT_SUCCESS;
}
