CovariateInformation drop_covariate(const CovariateInformation &info,
                                    const Design::Design &design,
                                    const std::string &cov_label) {
  auto mod_info = info;
  for (size_t idx = 0; idx < info.idxs.size(); ++idx)
    if (design.covariates[info.idxs[idx]].label == cov_label) {
      mod_info.idxs.erase(begin(mod_info.idxs) + idx);
      mod_info.vals.erase(begin(mod_info.vals) + idx);
    }
  return mod_info;
}

CovariateInformation drop_covariates(
    CovariateInformation info, const Design::Design &design,
    const std::vector<std::string> &cov_labels) {
  for (auto &cov_label : cov_labels)
    info = drop_covariate(info, design, cov_label);
  return info;
}

CovariateInformation get_covariate_info(
    const Design::Design& design, const set<string>& covariates, size_t experiment)
{
  auto covariates_ = covariates;

  using namespace Design;

  // spot dependency implies section dependency
  if (covariates_.find(spot_label) != covariates_.end()) {
    covariates_.insert(section_label);
  }

  vector<size_t> cov_idxs = design.determine_covariate_idxs(covariates_);
  vector<size_t> cov_vals;
  for (auto &covariate_idx : cov_idxs)
    cov_vals.push_back(
        design.dataset_specifications[experiment].covariate_values[covariate_idx]);

  return CovariateInformation { cov_idxs, cov_vals };
}

// Removes trailing zeros in a numeric string's decimals.
string remove_trailing_zeros(const string& str) {
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

void verify_model(const Model& m) {
  {  // check for overspecification
    static const auto input_dim
        = Coefficient::Kind::gene | Coefficient::Kind::spot;
    for (auto &x : m.coeffs) {
      if ((x->kind & input_dim) == input_dim) {
        throw runtime_error(
            "Error: coefficient '" + x->label
            + "' has dimensionality greater than or equal to the input data.");
      }
    }
  }

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
  }
}

}  // namespace

std::vector<CoefficientPtr>::iterator Model::find_coefficient(const Coefficient::Id& cid) {
  return find_if(begin(coeffs), end(coeffs), [&](const CoefficientPtr &coeff) {
    return coeff->label == cid.name and coeff->kind == cid.kind
           and coeff->distribution == cid.dist and coeff->info == cid.info;
  });
}

size_t Model::register_coefficient(
    const unordered_map<string, ModelSpec::Variable>& variable_map,
    string id,
    size_t experiment)
{
  // Register coefficient if it doesn't already exist and return its index in
  // the coeffs vector.
  auto do_registration = [this](
      const Coefficient::Id& cid,
      size_t _G,
      size_t _T,
      size_t _S,
      std::function<void(size_t)> on_add
      ) -> size_t {
    auto it = find_coefficient(cid);
    size_t idx;
    if (it != end(coeffs)) {
      idx = distance(begin(coeffs), it);
    } else {
      idx = coeffs.size();
      LOG(debug) << "Adding new coefficient for " << cid.name << " (" << idx
                 << ").";
      coeffs.emplace_back(std::make_shared<Coefficient>(_G, _T, _S, cid));
      on_add(idx);
    }
    return idx;
  };

  auto register_fixed = [&](double value) {
    Coefficient::Id cid{
      .name = id,
      .kind = Coefficient::Kind::scalar,
      .dist = Coefficient::Distribution::fixed,
      .info = CovariateInformation{},
    };
    return do_registration(
        cid, 0, 0, 0, [&](size_t idx) { coeffs[idx]->get_raw(0, 0, 0) = value; });
  };

  auto register_random = [&]() {
    auto it = variable_map.find(id);
    if (it == variable_map.end()) {
      throw runtime_error("Unknown variable id '" + id + "'.");
    }
    auto variable = it->second;

    auto info = get_covariate_info(design, variable->covariates, experiment);
    auto kind = determine_kind(variable->covariates);

    if (variable->distribution == nullptr) {
      auto dist = parameters.default_distribution;
      LOG(verbose) << id
                   << " does not have a distribution specification. Using "
                   << to_string(dist) << " as per defaults.";
      variable->distribution = make_shared<Distribution>(dist,
          vector<string> {
              remove_trailing_zeros(to_string(
                    parameters.hyperparameters.get_param(dist, 0))),
              remove_trailing_zeros(to_string(
                    parameters.hyperparameters.get_param(dist, 1))),
          });
    }

    Coefficient::Id cid{
      .name = id,
      .kind = kind,
      .dist = variable->distribution->type,
      .info = info,
    };

    size_t idx = do_registration(
        cid, G, T, experiments[experiment].S, [&](size_t _idx) {
          size_t i = 1;
          for (auto& prior : variable->distribution->arguments) {
            size_t prior_idx = register_coefficient(
                variable_map, prior, experiment);
            coeffs[_idx]->prior_idxs.push_back(prior_idx);
            LOG(debug) << "Prior " << i++ << " of " << id << " (" << _idx
                       << ") is " << prior << " (" << prior_idx << ").";
          }
        });

    if (variable->distribution->type == Coefficient::Distribution::gp) {
      auto gp_kind = kind & ~Coefficient::Kind::spot;

      // Create or update coordinate system
      auto gp_coord_info = drop_covariates(
          info, design, { Design::spot_label, Design::section_label });
      auto gp_coord_id = Coefficient::Id{
        .name = id + "-gp-coord",
        .kind = gp_kind,
        .dist = Coefficient::Distribution::gp_coord,
        .info = gp_coord_info,
      };
      auto gp_coord_idx
          = do_registration(gp_coord_id, G, T, 0, [](size_t _gp_coord_idx) {
              LOG(debug) << "Added new GP coordinate system (" << _gp_coord_idx
                         << ").";
            });
      LOG(debug) << "Updating GP coordinate system (" << gp_coord_idx << ").";
      coeffs[gp_coord_idx]->experiment_idxs.push_back(experiment);
      coeffs[gp_coord_idx]->prior_idxs.push_back(idx);

      // Create or update GP proxy
      auto gp_proxy_info = drop_covariates(info, design,
          { Design::spot_label, Design::section_label,
              Design::coordsys_label });
      auto gp_id = Coefficient::Id{
        .name = id + "-gp-proxy",
        .kind = gp_kind,
        .dist = Coefficient::Distribution::gp_proxy,
        .info = gp_proxy_info,
      };
      auto gp_proxy_idx
          = do_registration(gp_id, G, T, 0, [](size_t _gp_proxy_idx) {
              LOG(debug) << "Added new GP proxy (" << _gp_proxy_idx << ").";
            });
      LOG(debug) << "Updating GP proxy (" << gp_proxy_idx << ").";
      Coefficient& gp_coeff = *coeffs[gp_proxy_idx];
      gp_coeff.experiment_idxs.push_back(experiment);
      if (std::find(begin(gp_coeff.prior_idxs), end(gp_coeff.prior_idxs),
              gp_coord_idx)
          == end(gp_coeff.prior_idxs)) {
        gp_coeff.prior_idxs.push_back(gp_coord_idx);
      }
    }

    return idx;
  };

  double value;
  try {
    value = stod(id);
  } catch (const invalid_argument&) {
    return register_random();
  }
  return register_fixed(value);
}

void Model::add_covariates(const ModelSpec &model_spec) {
  auto rate_variables = collect_variables(model_spec.rate_expr);
  auto odds_variables = collect_variables(model_spec.odds_expr);
  for (size_t e = 0; e < E; ++e) {
    LOG(debug) << "Registering coefficients for experiment " << e;

    for(auto &variable: rate_variables) {
      auto idx = register_coefficient(model_spec.variables, variable->full_id(), e);
      coeffs[idx]->experiment_idxs.push_back(e);
      experiments[e].rate_coeff_idxs.push_back(idx);
    }

    for(auto &variable: odds_variables) {
      auto idx = register_coefficient(model_spec.variables, variable->full_id(), e);
      coeffs[idx]->experiment_idxs.push_back(e);
      experiments[e].odds_coeff_idxs.push_back(idx);
    }
  }
}

void Model::add_gp_proxies() {
  LOG(debug) << "Constructing GP proxies";
  for (size_t idx = 0; idx < coeffs.size(); ++idx)
    if (coeffs[idx]->distribution == Coefficient::Distribution::gp_proxy) {
      LOG(debug) << "Constructing GP proxy " << idx << ": " << *coeffs[idx];
      for (auto &coord_coeff_idx : coeffs[idx]->prior_idxs) {
        assert(coeffs[coord_coeff_idx]->distribution
               == Coefficient::Distribution::gp_coord);
        LOG(debug) << "using coordinate system coefficient " << coord_coeff_idx
                   << ": " << *coeffs[coord_coeff_idx];
        auto &coord_coeff = *coeffs[coord_coeff_idx];
        auto exp_idxs = coord_coeff.experiment_idxs;
        auto prior_idxs = coord_coeff.prior_idxs;
        size_t n = 0;
        for (size_t e : exp_idxs)
          n += experiments[e].S;
        size_t ncol = experiments[*exp_idxs.begin()].coords.cols();
        LOG(debug) << "n = " << n;
        Matrix m = Matrix::Zero(n, ncol);
        size_t i = 0;
        for (size_t e : exp_idxs) {
          for (size_t s = 0; s < experiments[e].S; ++s)
            for (size_t j = 0; j < ncol; ++j)
              m(i + s, j) = experiments[e].coords(s, j);
          i += experiments[e].S;
        }
        LOG(debug) << "m.dimesions = " << m.rows() << "x" << m.cols();
        coeffs[coord_coeff_idx]->gp = make_shared<GP::GaussianProcess>(
            GP::GaussianProcess(m, parameters.gp.length_scale));
      }
    }
}

void Model::coeff_debug_dump(const string &tag) const {
  size_t index = 0;
  for (auto coeff : coeffs)
    LOG(debug) << tag << " " << index++ << " " << *coeff << ": "
               << coeff->info.to_string(design.covariates);
  auto fnc = [&](const string &s, size_t idx, size_t e) {
    LOG(debug) << tag << " " << s << " experiment " << e << " " << idx << " "
               << *coeffs[idx] << ": "
               << coeffs[idx]->info.to_string(design.covariates);
  };
  for (size_t e = 0; e < E; ++e) {
    for (auto idx : experiments[e].rate_coeff_idxs)
      fnc("rate", idx, e);
    for (auto idx : experiments[e].odds_coeff_idxs)
      fnc("odds", idx, e);
  }
}

vector<size_t> find_redundant(const vector<vector<size_t>> &v) {
  using inv_map_t = multimap<vector<size_t>, size_t>;
  inv_map_t m;
  vector<size_t> redundant;
  for (size_t i = 0; i < v.size(); ++i) {
    auto key = v[i];
    if (not key.empty()) {
      pair<vector<size_t>, size_t> entry = {key, i};
      m.insert(entry);
      if (m.count(key) > 1)
        redundant.push_back(i);
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
  vector<vector<size_t>> cov2groups_rate(coeffs.size());
  for (size_t e = 0; e < E; ++e)
    for (auto idx : experiments[e].rate_coeff_idxs)
      if (coeffs[idx]->kind == kind)
        cov2groups_rate[idx].push_back(e);
  remove_redundant_terms_sub(cov2groups_rate);

  vector<vector<size_t>> cov2groups_odds(coeffs.size());
  for (size_t e = 0; e < E; ++e)
    for (auto idx : experiments[e].odds_coeff_idxs)
      if (coeffs[idx]->kind == kind)
        cov2groups_rate[idx].push_back(e);
  remove_redundant_terms_sub(cov2groups_odds);
}

void Model::remove_redundant_terms_sub(
    const vector<vector<size_t>> &cov2groups) {
  // TODO print warning in case coefficients are used in both rate and odds eqs
  auto redundant = find_redundant(cov2groups);
  sort(begin(redundant), end(redundant));
  reverse(begin(redundant), end(redundant));

  // drop redundant coefficients
  for (auto r : redundant) {
    LOG(verbose) << "Removing coefficient " << r << ": " << *coeffs[r] << ": "
               << coeffs[r]->info.to_string(design.covariates);
    coeffs.erase(begin(coeffs) + r);
  }

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
  }
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

  auto exp_gene_type = expected_gene_type();
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
      // TODO use parse-able format
      ofstream ofs(prefix + "hyperparameters.txt");
      ofs << parameters.hyperparameters;
    }
#pragma omp section
    write_matrix(exp_gene_type, prefix + "expected-features" + FILENAME_ENDING,
                 parameters.compression_mode, gene_names, type_names, order);

// TODO cov perhaps write out a single file for the scalar covariates
#pragma omp section
    {
      for (auto &coeff : coeffs) {
        vector<string> spot_names;
        if (coeff->spot_dependent())
          for (auto idx : coeff->experiment_idxs)
            spot_names.insert(begin(spot_names),
                              begin(experiments[idx].counts.col_names),
                              end(experiments[idx].counts.col_names));
        coeff->store(prefix + "covariate-" + storage_type(coeff->kind) + "-"
                        + coeff->label + "-"
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
}

/* TODO covariates enable loading of subsets of covariates */
void Model::restore(const string &prefix) {
  {
    for (auto &coeff : coeffs) {
      coeff->restore(
          prefix + "covariate-" + storage_type(coeff->kind) + "-" + coeff->label
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
