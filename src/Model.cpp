#include <boost/tokenizer.hpp>
#include "Model.hpp"

using namespace std;
namespace PoissonFactorization {

// extern size_t sub_model_cnt = 0; // TODO

bool gibbs_test(Float nextG, Float G, Float temperature) {
  double dG = nextG - G;
  double r = RandomDistribution::Uniform(EntropySource::rng);
  double p = std::min<double>(1.0, MetropolisHastings::boltzdist(-dG, temperature));
  LOG(debug) << "T = " << temperature << " nextG = " << nextG << " G = " << G
      << " dG = " << dG << " p = " << p << " r = " << r;
  if (std::isnan(nextG) == 0 and (dG > 0 or r <= p)) {
    LOG(verbose) << "Accepted!";
    return true;
  } else {
    LOG(verbose) << "Rejected!";
    return false;
  }
}

size_t num_lines(const string &path) {
  int number_of_lines = 0;
  string line;
  ifstream ifs(path);

  while (getline(ifs, line))
    ++number_of_lines;
  return number_of_lines;
}
}
