#ifndef PDIST_HPP
#define PDIST_HPP

#include <unistd.h>
#include <vector>

// Discrete probability distributions

/** Poisson probability mass function for k given a rate parameter lambda */
double log_poisson(size_t k, double lambda);
/** Negative binomial probability mass function for x given a number of failures r and a success probability p */
double log_negative_binomial(double x, double r, double p);
double log_negative_binomial(double x, double r, double p1, double p2);

// Continuous probability distributions

/** Gamma probability density function for x given a shape parameter k and a scale parameter theta */
double log_gamma(double x, double k, double theta);
/** Beta probability density function for x given shape parameters alpha and beta */
double log_beta(double x, double alpha, double beta);
/** Dirichlet probability density function for a probability distribution p given a vector of concentration aparameters alpha */
double log_dirichlet(const std::vector<double> &p, const std::vector<double> &alpha);

#endif
