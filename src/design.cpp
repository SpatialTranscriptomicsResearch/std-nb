#include "design.hpp"
#include <map>
#include "aux.hpp"

using namespace std;

vector<size_t> Design::determine_covariate_idxs(const set<string> &term) const {
  using namespace DesignNS;
  vector<size_t> cov_idxs;
  for (auto &covariate_label : term) {
    LOG(debug) << "Treating covariate label: " << covariate_label;
    string label = to_lower(covariate_label);
    if (label != gene_label and label != spot_label and label != type_label) {
      auto cov_iter = find_if(begin(covariates), end(covariates),
                              [&](const Covariate &covariate) {
                                return covariate.label == covariate_label;
                              });
      if (cov_iter == end(covariates)) {
        throw(runtime_error("Error: a covariate mentioned in the formula '"
                            + covariate_label
                            + "' is not found in the design."));
      } else {
        cov_idxs.push_back(distance(begin(covariates), cov_iter));
      }
    }
  }
  return cov_idxs;
}

vector<size_t> Design::get_covariate_value_idxs(
    size_t e, const vector<size_t> &covariate_idxs) const {
  vector<size_t> cov_values;
  for (auto &cov_idx : covariate_idxs)
    cov_values.push_back(dataset_specifications[e].covariate_values[cov_idx]);
  return cov_values;
}

void Design::from_string(const string &str) {
  istringstream is(str);
  from_stream(is);
}

bool Design::is_reserved_name(const string &s) const {
  using namespace DesignNS;
  if (s == unit_label or s == gene_label or s == spot_label or s == type_label)
    return true;
  return false;
}

void Design::add_dataset_specification(const string &path) {
  size_t size = dataset_specifications.size();
  string name = "Dataset " + std::to_string(size + 1);

  size_t num_covariates = covariates.size();
  vector<size_t> v(num_covariates, 0);

  Specification spec = {path, name, v};
  dataset_specifications.push_back(spec);
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
  if (find_if(begin(covariates), end(covariates),
              [&](const Covariate &cov) {
                return cov.label == DesignNS::section_label;
              })
      != end(covariates))
    return;

  Covariate cov = {DesignNS::section_label, {}};
  for (size_t i = 0; i < dataset_specifications.size(); ++i) {
    cov.values.push_back(std::to_string(i + 1));
    dataset_specifications[i].covariate_values.push_back(i);
  }
  covariates.push_back(cov);
}

void Design::add_covariate_coordsys(bool share_coord_sys) {
  if (find_if(begin(covariates), end(covariates),
              [&](const Covariate &cov) {
                return cov.label == DesignNS::coordsys_label;
              })
      != end(covariates))
    return;

  Covariate cov = {DesignNS::coordsys_label, {}};
  size_t coord_sys_idx = 0;
  for (size_t i = 0; i < dataset_specifications.size(); ++i) {
    cov.values.push_back(std::to_string(coord_sys_idx + 1));
    dataset_specifications[i].covariate_values.push_back(coord_sys_idx);
    if (not share_coord_sys)
      coord_sys_idx++;
  }
  covariates.push_back(cov);
}

void Design::add_covariate_unit() {
  if (find_if(begin(covariates), end(covariates),
              [&](const Covariate &cov) {
                return cov.label == DesignNS::unit_label;
              })
      != end(covariates))
    return;

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
