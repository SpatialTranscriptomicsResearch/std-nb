#ifndef FORMULA_HPP
#define FORMULA_HPP

#include <vector>
#include "covariate.hpp"
// #include "term.hpp"

// struct Formula {
// };

using Term = std::vector<Covariate>;

using Formula = std::vector<Term>;

Formula parse_formula(const std::string &formula_str);
bool check_formula(const Formula &formula);

#endif
