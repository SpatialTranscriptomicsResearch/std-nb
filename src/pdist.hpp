#ifndef PDIST_HPP
#define PDIST_HPP

#include <unistd.h>
#include <vector>

double log_poisson(size_t k, double lambda);
double log_gamma(double x, double k, double theta);
double log_beta(double a, double b, double c);
double log_dirichlet(const std::vector<double> &alpha, const std::vector<double> &p);

#endif
