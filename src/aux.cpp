#include <algorithm>
#include "aux.hpp"

using namespace std;

string to_lower(string x) {
  transform(begin(x), end(x), begin(x), ::tolower);
  return x;
}
