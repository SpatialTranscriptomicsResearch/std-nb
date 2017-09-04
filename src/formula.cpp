#include "formula.hpp"

#include <functional>
#include <set>
#include <sstream>
#include <unordered_set>

#include "aux.hpp"

using std::function;
using std::pair;
using std::vector;
using std::set;
using std::string;
using std::unordered_set;

using vs = vector<string>;
using vvs = vector<vs>;
using combinator = function<vvs(const vvs&, const vvs&)>;

namespace std {
template <typename T>
struct hash<vector<T>> {
  size_t operator()(const vector<T>& xs) const {
    size_t h = 0;
    for (auto& x : xs) {
      h ^= hash<T>()(x);
    }
    return h;
  }
};
}

static vvs apply_operator(const char& sym, const combinator& c,
                          const string& s);
static vvs parse(const string& s);
static vvs simplify(const vvs& xss);
static vs split_outside_parens(const char& sep, const string& str);

static vvs concatenate(const vvs& a, const vvs& b);
static vvs exponentiate(const vvs& a, const vvs& b);
static vvs interact(const vvs& a, const vvs& b);
static vvs multiply(const vvs& a, const vvs& b);

static vector<pair<char, combinator>> operators{
  make_pair('+', concatenate),
  make_pair(':', interact),
  make_pair('*', multiply),
  make_pair('^', exponentiate),
};

static vvs concatenate(const vvs& a, const vvs& b) {
  vvs ret(a);
  ret.insert(ret.end(), b.begin(), b.end());
  return simplify(ret);
}

static vvs interact(const vvs& a, const vvs& b) {
  vvs ret;
  for (auto& x : a) {
    for (auto& y : b) {
      vs cur(x);
      cur.insert(cur.end(), y.begin(), y.end());
      ret.push_back(cur);
    }
  }
  return simplify(ret);
}

static vvs multiply(const vvs& a, const vvs& b) {
  return concatenate(concatenate(a, b), interact(a, b));
}

static vvs exponentiate(const vvs& a, const vvs& b) {
  if (b.size() != 1 or b[0].size() != 1) {
    throw std::invalid_argument("Exponent cannot be an expression.");
  }
  int exp;
  try {
    exp = stoi(b[0][0]);
  } catch (const std::invalid_argument&) {
    throw std::invalid_argument("Exponent must be numeric.");
  }
  if (not(exp > 0)) {
    throw std::invalid_argument("Exponent must be positive.");
  }
  if (exp == 1) {
    return a;
  }
  return multiply(a, exponentiate(a, vvs{{std::to_string(exp - 1)}}));
}

static vs split_outside_parens(const char& sep, const string& str) {
  string trimmed = trim(str);
  int pstack = 0, pcount = 0;
  vector<string> ret;
  string cur;
  for (auto& c : trimmed) {
    switch (c) {
      case '(':
        ++pstack;
        break;
      case ')':
        if (--pstack < 0) {
          throw std::invalid_argument("Too many closing brackets.");
        }
        pcount += pstack == 0;
        break;
      default:
        if (c == sep and pstack == 0) {
          ret.push_back(trim(cur));
          cur.clear();
          continue;
        }
    }
    cur.push_back(c);
  }
  if (pstack > 0) {
    throw std::invalid_argument("Too many opening brackets.");
  }
  if (pcount == 1 and trimmed.front() == '(' and trimmed.back() == ')') {
    return split_outside_parens(sep, trimmed.substr(1, trimmed.size() - 2));
  }
  ret.push_back(trim(cur));
  return ret;
}

static vvs apply_operator(const char& sym, const combinator& c,
                          const string& s) {
  vs split = split_outside_parens(sym, s);
  if (any_of(split.begin(), split.end(),
             [](const string& s_) { return s_ == ""; })) {
    throw std::invalid_argument("Found empty operand.");
  }
  if (split.size() == 1) {
    return vvs();
  }
  vvs ret{parse(split[0])};
  for (size_t i = 1; i < split.size(); ++i) {
    vvs sub_res = parse(split[i]);
    ret = c(ret, sub_res);
  }
  return ret;
}

static vvs simplify(const vvs& xss) {
  unordered_set<vs> seen_covariates;
  for (auto& xs : xss) {
    set<string> seen_terms;
    for (auto& x : xs) {
      seen_terms.insert(x);
    }
    seen_covariates.insert(vs(seen_terms.begin(), seen_terms.end()));
  }
  return vvs(seen_covariates.begin(), seen_covariates.end());
}

static vvs parse(const string& s) {
  if (s.empty())
    return vvs();
  for (auto& op : operators) {
    auto res = apply_operator(op.first, op.second, s);
    if (not res.empty()) {
      return res;
    }
  }
  return vvs{{s}};
}

Formula::Formula(const string& str) { from_string(str); }

void Formula::from_string(const string& str) {
  terms = parse(str);
  sort(terms.begin(), terms.end());
  add_section_to_spots();
}

string Formula::to_string() const {
  vs covariates;
  for (const Term& x : terms) {
    covariates.push_back(
        intercalate<std::vector<std::string>::const_iterator, std::string>(
            x.begin(), x.end(), string(":")));
  }
  return intercalate<std::vector<std::string>::const_iterator, std::string>(
      covariates.begin(), covariates.end(), string("+"));
}

Formula DefaultRateFormula() { return Formula("1+gene*(type+section)"); }
// Formula DefaultVarianceFormula() { return Formula("gene+section+1"); }
Formula DefaultVarianceFormula() { return Formula("gene*type+section+1"); }

void Formula::add_section_to_spots() {
  for (auto& term : terms)
    if (find(begin(term), end(term), "spot") != end(term)
        and find(begin(term), end(term), "section") == end(term))
      term.push_back("section");
}

std::ostream& operator<<(std::ostream& os, const Formula& formula) {
  os << formula.to_string();
  return os;
}

std::istream& operator>>(std::istream& is, Formula& formula) {
  string token;
  getline(is, token);
  formula.from_string(token);
  return is;
}
