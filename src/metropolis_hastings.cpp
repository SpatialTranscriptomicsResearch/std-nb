#include "metropolis_hastings.hpp"

MetropolisHastings::MetropolisHastings(double temp, double prop_sd_,
                                       Verbosity verb)
    : temperature(temp), prop_sd(prop_sd_), verbosity(verb){};
