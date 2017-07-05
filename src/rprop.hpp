#ifndef RPROP_HPP
#define RPROP_HPP

#include <cmath>
#include "log.hpp"
#include "stats.hpp"

template <typename T>
int sgn(T val) {
  if (val < 0)
    return -1;
  if (val > 0)
    return 1;
  return 0;
}

struct rprop_parameters {
  // double eta_plus = 1.2;
  // double eta_minus = 0.5;
  double eta_plus = 1.1;
  double eta_minus = 1.0 - 1.0 / 3;

  // double max_change = log(50);
  double max_change = log(10);
  // double max_change = 50;
  // double max_change = 10;
  // double min_change = 1 / max_change;
  double min_change = 0;
  // double min_change = 1e-6;
};

template <typename T, typename U>
void rprop_update(const T &grad, U &prev_sgn, T &rate, T &data,
                  const rprop_parameters &params) {
  auto grad_iter = begin(grad);
  auto data_iter = begin(data);
  auto sgn_iter = begin(prev_sgn);
  size_t caseP = 0, case0 = 0, caseN = 0;
  for (auto &r : rate) {
    int sgn_grad = sgn(*grad_iter);
    switch (sgn_grad * int(*sgn_iter)) {
      case 1:
        r = std::min(params.max_change, r * params.eta_plus);
        caseP++;
      case 0:
        *data_iter += sgn_grad * r;
        *sgn_iter = sgn_grad;
        case0++;
        break;
      case -1:
        r = std::max(params.min_change, r * params.eta_minus);
        *sgn_iter = 0;
        caseN++;
        break;
      default:
        LOG(fatal) << "Unhandled case in switch statement!";
        exit(-1);
    }
    grad_iter++;
    data_iter++;
    sgn_iter++;
  }
  case0 -= caseP;
  LOG(verbose) << "+1/0/-1 " << caseP << "/" << case0 << "/" << caseN;
  LOG(verbose) << "learning rates:" << std::endl << Stats::summary(rate);
}

#endif
