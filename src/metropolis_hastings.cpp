#include "metropolis_hastings.hpp"

MetropolisHastings::MetropolisHastings(double temp, double prop_sd_)
    : temperature(temp), prop_sd(prop_sd_){};
