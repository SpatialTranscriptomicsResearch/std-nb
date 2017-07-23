#include "design.hpp"
#include <map>
#include "aux.hpp"

using namespace std;

void Design::from_string(const string &str) {
  istringstream is(str);
  from_stream(is);
}

bool Design::is_reserved_name(const string &s) const {
  if (s == "1")
    return true;
  return false;
}

void Design::from_stream(istream &is) {
  string line;
  if (getline(is, line)) {
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
      if (to_lower(tokens[i]) == DesignNS::path_label) {
        if (path_column >= 0)
          throw Exception::Design::MultiplePathColumns();
        else
          path_column = i;
      }

    if (path_column < 0)
      throw Exception::Design::NoPathColumn();

    // determine index of name column
    for (size_t i = 0; i < tokens.size(); ++i)
      if (to_lower(tokens[i]) == DesignNS::name_label) {
        if (name_column >= 0)
          throw Exception::Design::MultipleNameColumns();
        else
          name_column = i;
      }

    map<size_t, size_t> col2cov;
    for (size_t i = 0; i < tokens.size(); ++i)
      if (i != static_cast<size_t>(path_column)
          and (name_column < 0 or i != static_cast<size_t>(name_column))) {
        if (is_reserved_name(to_lower(tokens[i])))
          throw Exception::Design::ReservedCovariateName(tokens[i]);
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
        spec.name = "Dataset " + std::to_string(++dataset_idx);

      for (size_t i = 0; i < tokens.size(); ++i)
        if (i != static_cast<size_t>(path_column)
            and (name_column < 0 or i != static_cast<size_t>(name_column))) {
          LOG(debug) << "Treating token " << i << ": '" << tokens[i] << "'";
          size_t cov_idx = col2cov[i];
          auto iter = find(begin(covariates[cov_idx].values),
                           end(covariates[cov_idx].values), tokens[i]);
          size_t val_idx;
          if (iter != end(covariates[cov_idx].values))
            val_idx = distance(begin(covariates[cov_idx].values), iter);
          else {
            val_idx = covariates[cov_idx].values.size();
            covariates[cov_idx].values.push_back(tokens[i]);
          }
          spec.covariate_values.push_back(val_idx);
        }
      dataset_specifications.push_back(spec);
    }
  }

  if (find_if(begin(covariates), end(covariates),
              [&](const Covariate &cov) {
                return cov.label == DesignNS::section_label;
              })
      == end(covariates))
    add_covariate_section();

  if (find_if(begin(covariates), end(covariates),
              [&](const Covariate &cov) {
                return cov.label == DesignNS::unit_label;
              })
      == end(covariates))
    add_covariate_unit();
}

string Design::to_string() const {
  // TODO do not print unit column
  string str = DesignNS::path_label + "\t" + DesignNS::name_label;
  for (auto &covariate : covariates)
    str += "\t" + covariate.label;
  str += "\n";
  for (auto &spec : dataset_specifications) {
    str += spec.path + "\t" + spec.name;
    for (size_t i = 0; i < spec.covariate_values.size(); ++i)
      str += "\t" + covariates[i].values[spec.covariate_values[i]];
    str += "\n";
  }
  return str;
}

void Design::add_covariate_section() {
  Covariate cov = {DesignNS::section_label, {}};
  for (size_t i = 0; i < dataset_specifications.size(); ++i) {
    cov.values.push_back(std::to_string(i + 1));
    dataset_specifications[i].covariate_values.push_back(i);
  }
  covariates.push_back(cov);
}

void Design::add_covariate_unit() {
  Covariate cov = {DesignNS::unit_label, {}};
  cov.values.push_back(std::to_string(1));
  for (size_t i = 0; i < dataset_specifications.size(); ++i) {
    dataset_specifications[i].covariate_values.push_back(0);
  }
  covariates.push_back(cov);
}

istream &operator>>(istream &is, Design &design) {
  design.from_stream(is);
  return is;
}

ostream &operator<<(ostream &os, const Design &design) {
  os << design.to_string();
  return os;
}
