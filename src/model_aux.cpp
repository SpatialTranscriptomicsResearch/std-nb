CovariateInformation drop_covariate(const CovariateInformation &info,
                                    const Design::Design &design,
                                    const string &cov_label) {
  auto mod_info = info;
  for (size_t idx = 0; idx < info.idxs.size(); ++idx)
    if (design.covariates[info.idxs[idx]].label == cov_label) {
      mod_info.idxs.erase(begin(mod_info.idxs) + idx);
      mod_info.vals.erase(begin(mod_info.vals) + idx);
    }
  return mod_info;
}

CovariateInformation drop_covariates(CovariateInformation info,
                                     const Design::Design &design,
                                     const vector<string> &cov_labels) {
  for (auto &cov_label : cov_labels)
    info = drop_covariate(info, design, cov_label);
  return info;
}

CovariateInformation get_covariate_info(const Design::Design &design,
                                        const set<string> &covariates,
                                        size_t experiment) {
  auto covariates_ = covariates;

  using namespace Design;

  // spot dependency implies section dependency
  if (covariates_.find(spot_label) != covariates_.end()) {
    covariates_.insert(section_label);
  }

  vector<size_t> cov_idxs = design.determine_covariate_idxs(covariates_);
  vector<size_t> cov_vals;
  for (auto &covariate_idx : cov_idxs)
    cov_vals.push_back(design.dataset_specifications[experiment]
                           .covariate_values[covariate_idx]);

  return CovariateInformation{cov_idxs, cov_vals};
}

// Removes trailing zeros in a numeric string's decimals.
string remove_trailing_zeros(const string &str) {
  if (str.find('.') == string::npos) {
    return str;
  }
  auto pos = str.end();
  while (*(--pos) != '.') {
    if (*pos != '0') {
      pos += 1;
      break;
    }
  }
  return string(str.begin(), pos);
}

void verify_model(const Model &m) {
  {  // check for overspecification
    static const auto input_dim
        = Coefficient::Kind::gene | Coefficient::Kind::spot;
    for (auto &x : m.coeffs) {
      if ((x->kind & input_dim) == input_dim) {
        throw runtime_error(
            "Error: coefficient '" + x->name
            + "' has dimensionality greater than or equal to the input data.");
      }
    }
  }

  /* TODO FIXUP coeffs
  {  // check for cycles in model spec
    auto check_cycles = [&m](size_t root) {
      vector<bool> visited(m.coeffs.size());
      function<void(size_t)> go = [&m, &visited, &go](size_t x) {
        if (visited[x]) {
          throw runtime_error(
              "Error: cyclic model specifications are currently not "
              "supported.");
        }
        visited[x] = true;
        for (auto &p : m.coeffs[x]->prior_idxs) {
          go(p);
        }
        visited[x] = false;
      };
      return go(root);
    };
    unordered_set<size_t> coeffs;
    for (auto &e : m.experiments) {
      coeffs.insert(e.rate_coeff_idxs.begin(), e.rate_coeff_idxs.end());
      coeffs.insert(e.odds_coeff_idxs.begin(), e.odds_coeff_idxs.end());
    }
    for (auto &x : coeffs) {
      check_cycles(x);
    }
  } */
}

}  // namespace

vector<CoefficientPtr>::iterator Model::find_coefficient(
    const Coefficient::Id &cid) {
  return find_if(begin(coeffs), end(coeffs), [&](const CoefficientPtr &coeff) {
    return coeff->name == cid.name and coeff->kind == cid.kind
           and coeff->type == cid.type and coeff->info == cid.info;
  });
}

CoefficientPtr Model::register_coefficient(
    const unordered_map<string, ModelSpec::Variable> &variable_map, string id,
    size_t experiment) {
  // Register coefficient if it doesn't already exist and return a pointer to it
  auto do_registration
      = [this, &experiment](
            const Coefficient::Id &cid, size_t _G, size_t _T, size_t _S,
            const vector<CoefficientPtr> &priors) -> CoefficientPtr {
    auto it = find_coefficient(cid);
    if (it != end(coeffs)) {
      (*it)->experiments.push_back(&experiments[experiment]);
      return *it;
    } else {
      LOG(debug) << "Adding new coefficient for " << cid.name << ".";
      coeffs.emplace_back(Coefficient::make_shared(_G, _T, _S, cid, priors));
      auto coeff = coeffs.back();
      coeff->experiments.push_back(&experiments[experiment]);
      LOG(debug) << "Added new coefficient: " << *coeff << ".";
      return coeff;
    }
  };

  auto register_fixed = [&](double value) {
    Coefficient::Id cid{
        .name = id,
        .kind = Coefficient::Kind::scalar,
        .type = Coefficient::Type::fixed,
        .info = CovariateInformation{},
    };
    auto coeff = do_registration(cid, 0, 0, 0, {});
    coeff->get_raw(0, 0, 0) = value;
    return coeff;
  };

  auto register_random = [&]() {
    auto it = variable_map.find(id);
    if (it == variable_map.end()) {
      throw runtime_error("Unknown variable id '" + id + "'.");
    }
    auto variable = it->second;

    auto info = get_covariate_info(design, variable->covariates, experiment);
    auto kind = Coefficient::determine_kind(variable->covariates);

    if (variable->distribution == nullptr) {
      auto dist = parameters.default_distribution;
      LOG(debug) << id << " does not have a distribution specification. Using "
                 << to_string(dist) << " as per defaults.";
      variable->distribution = make_shared<Distribution>(
          dist, vector<string>{
                    remove_trailing_zeros(to_string(
                        parameters.hyperparameters.get_param(dist, 0))),
                    remove_trailing_zeros(to_string(
                        parameters.hyperparameters.get_param(dist, 1))),
                });
    }

    Coefficient::Id cid{
        .name = id,
        .kind = kind,
        .type = variable->distribution->type,
        .info = info,
    };

    vector<CoefficientPtr> priors;
    {  // register prior coefficients
      size_t i = 1;
      for (auto &argument : variable->distribution->arguments) {
        auto prior = register_coefficient(variable_map, argument, experiment);
        priors.emplace_back(prior);
        LOG(debug) << "Prior " << i++ << " of " << id << " is " << argument
                   << " (" << *prior << ").";
      }
    }

    CoefficientPtr coeff
        = do_registration(cid, G, T, experiments[experiment].S, priors);
    if (variable->distribution->type == Coefficient::Type::gp_points) {
      auto gp_kind = kind & ~Coefficient::Kind::spot;

      // Create or update coordinate system
      auto gp_coord_info = drop_covariates(
          info, design, {Design::spot_label, Design::section_label});
      auto gp_coord_id = Coefficient::Id{
          .name
          = id + "-gp-coord-"
            + design.get_covariate_value(experiment, Design::coordsys_label),
          .kind = gp_kind,
          .type = Coefficient::Type::gp_coord,
          .info = gp_coord_info,
      };
      auto gp_coord_coeff = dynamic_pointer_cast<Coefficient::Spatial::Coord>(
          do_registration(gp_coord_id, G, T, 0, priors));
      LOG(debug) << "Updating GP coordinate system (" << *gp_coord_coeff
                 << ").";
      gp_coord_coeff->points.emplace_back(
          dynamic_pointer_cast<Coefficient::Spatial::Points>(coeff));
    }

    return coeff;
  };

  double value;
  try {
    value = stod(id);
  } catch (const invalid_argument &) {
    return register_random();
  }
  return register_fixed(value);
}

void Model::add_covariates() {
  auto rate_variables = collect_variables(model_spec.rate_expr);
  auto odds_variables = collect_variables(model_spec.odds_expr);
  for (size_t e = 0; e < E; ++e) {
    LOG(debug) << "Registering coefficients for experiment " << e;

    for (auto &variable : rate_variables) {
      auto coeff
          = register_coefficient(model_spec.variables, variable->full_id(), e);
      experiments[e].rate_coeffs.emplace_back(coeff);
    }

    for (auto &variable : odds_variables) {
      auto coeff
          = register_coefficient(model_spec.variables, variable->full_id(), e);
      experiments[e].odds_coeffs.emplace_back(coeff);
    }
  }
}

void Model::construct_GPs() {
  LOG(debug) << "Constructing GPs";
  for (size_t idx = 0; idx < coeffs.size(); ++idx)
    if (coeffs[idx]->type == Coefficient::Type::gp_coord) {
      LOG(debug) << "Constructing GP " << idx << ": " << *coeffs[idx];
      auto coord_coeff
          = dynamic_pointer_cast<Coefficient::Spatial::Coord>(coeffs[idx]);
      coord_coeff->construct_gp();
    }
}

void Model::coeff_debug_dump(const string &tag) const {
  size_t index = 0;
  for (auto coeff : coeffs)
    LOG(debug) << tag << " " << index++ << " " << *coeff << ": "
               << coeff->info.to_string(design.covariates);
  auto fnc = [&](const string &s, CoefficientPtr coeff, size_t e) {
    LOG(debug) << tag << " " << s << " experiment " << e << " " << *coeff
               << ": " << coeff->info.to_string(design.covariates);
  };
  for (size_t e = 0; e < E; ++e) {
    for (auto coeff : experiments[e].rate_coeffs)
      fnc("rate", coeff, e);
    for (auto coeff : experiments[e].odds_coeffs)
      fnc("odds", coeff, e);
  }
}

set<CoefficientPtr> find_redundant(const map<CoefficientPtr, set<size_t>> &v) {
  using inv_map_t = multimap<set<size_t>, CoefficientPtr>;
  inv_map_t m;
  set<CoefficientPtr> redundant;
  for (auto entry : v) {
    if (not entry.second.empty()) {
      pair<set<size_t>, CoefficientPtr> inv_entry = {entry.second, entry.first};
      m.insert(inv_entry);
      if (m.count(entry.second) > 1)
        redundant.insert(entry.first);
    }
  }
  return redundant;
}

void Model::remove_redundant_terms() {
  using Kind = Coefficient::Kind;
  for (auto kind : {Kind::scalar, Kind::gene, Kind::type, Kind::spot,
                    Kind::gene_type, Kind::spot_type})
    remove_redundant_terms(kind);
}

// TODO covariates: add redundant term labels
void Model::remove_redundant_terms(Coefficient::Kind kind) {
  using map_t = map<CoefficientPtr, set<size_t>>;
  map_t cov2groups_rate;
  for (size_t e = 0; e < E; ++e)
    for (auto coeff : experiments[e].rate_coeffs)
      if (coeff->kind == kind)
        cov2groups_rate[coeff].insert(e);
  remove_redundant_terms_sub(cov2groups_rate);

  map_t cov2groups_odds;
  for (size_t e = 0; e < E; ++e)
    for (auto coeff : experiments[e].odds_coeffs)
      if (coeff->kind == kind)
        cov2groups_rate[coeff].insert(e);
  remove_redundant_terms_sub(cov2groups_odds);
}

void Model::remove_redundant_terms_sub(
    const map<CoefficientPtr, set<size_t>> &cov2groups) {
  // TODO print warning in case coefficients are used in both rate and odds eqs
  auto redundant = find_redundant(cov2groups);

  // drop redundant coefficients
  for (auto coeff : redundant) {
    LOG(verbose) << "Removing coefficient " << *coeff << ": "
                 << coeff->info.to_string(design.covariates);
    coeffs.erase(find(begin(coeffs), end(coeffs), coeff));
  }

  /* TODO FIXUP coeffs
  // fix prior_idxs for dropped redundant coefficients
  for (auto &coeff : coeffs) {
    auto &idxs = coeff->prior_idxs;
    for (auto r : redundant)
      idxs.erase(remove(begin(idxs), end(idxs), r), end(idxs));
    for (auto &idx : idxs)
      for (auto r : redundant)
        if (idx > r)
          idx--;
  }

  // fix experiment.rate_coeff_idxs and experiment.odds_coeff_idxs for dropped
  // redundant coefficients
  for (size_t e = 0; e < E; ++e) {
    for (auto idxs :
         {&experiments[e].rate_coeff_idxs, &experiments[e].odds_coeff_idxs}) {
      for (auto r : redundant)
        idxs->erase(remove(begin(*idxs), end(*idxs), r), end(*idxs));
      for (auto &idx : *idxs)
        for (auto r : redundant)
          if (idx > r)
            idx--;
    }
  } */
}

template <typename V>
vector<size_t> get_order(const V &v) {
  size_t N = v.size();
  vector<size_t> order(N);
  iota(begin(order), end(order), 0);
  sort(begin(order), end(order),
       [&v](size_t a, size_t b) { return v[a] > v[b]; });
  return order;
}

void Model::store(const string &prefix_, bool mean_and_var,
                  bool reorder) const {
  string prefix = parameters.output_directory + prefix_;
  {
    using namespace boost::filesystem;
    if (not((exists(prefix) and is_directory(prefix))
            or create_directory(prefix)))
      throw(std::runtime_error("Couldn't create directory " + prefix));
  }
  auto type_names = form_factor_names(T);
  auto &gene_names = experiments.begin()->counts.row_names;

  vector<size_t> order;
  if (reorder) {
    auto cs = colSums<Vector>(contributions_gene_type);
    order = get_order(cs);
  }

#pragma omp parallel sections if (DO_PARALLEL)
  {
#pragma omp section
    {
      ofstream ofs(prefix + "design.txt");
      ofs << design;
    }
#pragma omp section
    {
      ofstream ofs(prefix + "model.txt");
      log([&](const string &s) { ofs << s << endl; }, model_spec);
    }
#pragma omp section
    {
      // TODO use parse-able format
      ofstream ofs(prefix + "hyperparameters.txt");
      ofs << parameters.hyperparameters;
    }

// TODO cov perhaps write out a single file for the scalar covariates
#pragma omp section
    {
      for (auto &coeff : coeffs) {
        vector<string> spot_names;
        if (coeff->spot_dependent())
          for (Experiment *experiment : coeff->experiments)
            spot_names.insert(begin(spot_names),
                              begin(experiment->counts.col_names),
                              end(experiment->counts.col_names));
        coeff->store(prefix + "covariate-" + storage_type(coeff->kind) + "-"
                         + coeff->name + "-"
                         + coeff->info.to_string(design.covariates)
                         + FILENAME_ENDING,
                     parameters.compression_mode, gene_names, spot_names,
                     type_names, order);
      }
    }

#pragma omp section
    write_matrix(contributions_gene_type,
                 prefix + "contributions_gene_type" + FILENAME_ENDING,
                 parameters.compression_mode, gene_names, type_names, order);
#pragma omp section
    write_vector(contributions_gene,
                 prefix + "contributions_gene" + FILENAME_ENDING,
                 parameters.compression_mode, gene_names);
  }
  for (size_t e = 0; e < E; ++e) {
    string exp_prefix = prefix + "experiment"
                        + to_string_embedded(e, EXPERIMENT_NUM_DIGITS) + "-";
    experiments[e].store(exp_prefix, order);
  }

  if (mean_and_var)
    for (size_t e = 0; e < E; ++e) {
      string exp_prefix = prefix + "experiment"
                          + to_string_embedded(e, EXPERIMENT_NUM_DIGITS) + "-";
      auto mean_var = compute_mean_and_var(e);
      write_matrix(mean_var.first,
                   exp_prefix + "counts_expected" + FILENAME_ENDING,
                   parameters.compression_mode, gene_names,
                   experiments[e].counts.col_names);
      write_matrix(mean_var.second,
                   exp_prefix + "counts_variance" + FILENAME_ENDING,
                   parameters.compression_mode, gene_names,
                   experiments[e].counts.col_names);
    }
}

/* TODO covariates enable loading of subsets of covariates */
void Model::restore(const string &prefix) {
  {
    for (auto &coeff : coeffs) {
      coeff->restore(
          prefix + "covariate-" + storage_type(coeff->kind) + "-" + coeff->name
          + "-" + coeff->info.to_string(design.covariates) + FILENAME_ENDING);
    }
  }

  contributions_gene_type = parse_file<Matrix>(
      prefix + "contributions_gene_type" + FILENAME_ENDING, read_matrix, "\t");
  contributions_gene
      = parse_file<Vector>(prefix + "contributions_gene" + FILENAME_ENDING,
                           read_vector<Vector>, "\t");

  for (size_t e = 0; e < E; ++e) {
    string exp_prefix = prefix + "experiment"
                        + to_string_embedded(e, EXPERIMENT_NUM_DIGITS) + "-";
    experiments[e].restore(exp_prefix);
  }
}
