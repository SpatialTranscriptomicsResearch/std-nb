#include <sstream>
#include "aux.hpp"
#include "formula.hpp"

using namespace std;

Formula::Formula(const string &str) { from_string(str); }

// TODO : Consider emitting syntactic error on space inside of covariates
void Formula::from_string(const string &str)
{
  vector<string> terms = split_at('+', str);
  for (string& term : terms) {
    vector<string> covariates = split_at(':', term);
    for(auto &covariate: covariates) {
      covariate = trim(covariate);
    }
    formula.push_back(covariates);
  }
}

string Formula::to_string() const {
  string str;
  bool first_term = true;
  for (auto & term : formula) {
    if(not first_term)
      str += "+";
    first_term = false;
    bool first_covariate = true;
    for (auto &covariate : term) {
      if (not first_covariate) {
        str += ":";
      }
      first_covariate = false;
      str += covariate;
    }
  }
  return str;
}

ostream &operator<<(ostream &os, const Formula &formula) {
  os << formula.to_string();
  return os;
}

istream &operator>>(istream &is, Formula &formula) {
  string token;
  is >> token;
  formula.from_string(token);
  return is;
}
