#include "parameters.hpp"
#include "aux.hpp"

using namespace std;

namespace STD {
bool Parameters::targeted(Target target) const {
  return flagged(targets & target);
}
}
