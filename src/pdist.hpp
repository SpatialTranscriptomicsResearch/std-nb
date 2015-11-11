#ifndef PDIST_HPP
#define PDIST_HPP

#include <unistd.h>

double log_poisson(size_t k, double lambda);
double log_gamma(double x, double k, double theta);
double log_beta(double a, double b, double c);

#endif
