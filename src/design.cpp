#include "design.hpp"
#include <map>
#include "aux.hpp"

using namespace std;

void read_design(std::istream &is, Design &design, Covariates &covariates) {
  string line;
  getline(is, line);
  int path_column = -1;
  int name_column = -1;

  auto tokens = split_at('\t', line);

  // check uniqueness
  {
    auto uniq_tokens = tokens;
    sort(begin(uniq_tokens), end(uniq_tokens));
    auto iter = adjacent_find(begin(uniq_tokens), end(uniq_tokens));
    if (iter != end(uniq_tokens))
      throw Exception::Design::RepeatedColumnName(*iter);
  }

  // determine index of path column
  for (size_t i = 0; i < tokens.size(); ++i)
    if (to_lower(tokens[i]) == "path") {
      if (path_column >= 0)
        throw Exception::Design::MultiplePathColumns();
      else
        path_column = i;
    }

  if (path_column < 0)
    throw Exception::Design::NoPathColumn();

  // determine index of name column
  for (size_t i = 0; i < tokens.size(); ++i)
    if (to_lower(tokens[i]) == "name") {
      if (name_column >= 0)
        throw Exception::Design::MultipleNameColumns();
      else
        name_column = i;
    }

  map<size_t, size_t> col2cov;
  for (size_t i = 0; i < tokens.size(); ++i)
    if (i != static_cast<size_t>(path_column)
        and (name_column < 0 or i != static_cast<size_t>(name_column))) {
      Covariate covariate;
      covariate.label = tokens[i];
      col2cov[i] = covariates.size();
      covariates.push_back(covariate);
    }

  size_t dataset_idx = 0;
  while (getline(is, line)) {
    tokens = split_at('\t', line);
    Specification spec;
    spec.path = tokens[path_column];
    if (name_column > 0)
      spec.name = tokens[name_column];
    else
      spec.name = "Dataset " + to_string(dataset_idx++);

    for (size_t i = 0; i < tokens.size(); ++i)
      if (i != static_cast<size_t>(path_column)
          and (name_column < 0 or i != static_cast<size_t>(name_column))) {
        size_t cov_idx = col2cov[i];
        auto iter = find(begin(covariates[cov_idx].values),
                         end(covariates[cov_idx].values), tokens[i]);
        size_t val_idx;
        if (iter == end(covariates[cov_idx].values))
          val_idx = distance(begin(covariates[cov_idx].values), iter);
        else {
          val_idx = covariates[cov_idx].values.size();
          covariates[cov_idx].values.push_back(tokens[i]);
        }
        spec.covariate_values.push_back(val_idx);
      }
    design.push_back(spec);
  }
}
