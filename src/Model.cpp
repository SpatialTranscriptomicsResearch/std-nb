#include "Model.hpp"
#include <LBFGS.h>
#include "gamma_func.hpp"
#include "rprop.hpp"

using namespace std;

namespace STD {

Model::Model(const vector<Counts> &c, size_t T_, const Parameters &parameters_,
             bool same_coord_sys)
    : G(max_row_number(c)),
      T(T_),
      E(0),
      S(0),
      experiments(),
      parameters(parameters_),
      phi_r(Matrix::Ones(G, T)),
      phi_p(Matrix::Ones(G, T)),
      mix_prior(sum_cols(c), T, parameters),
      contributions_gene_type(Matrix::Zero(G, T)),
      contributions_gene(Vector::Zero(G)) {
  LOG(debug) << "Model G = " << G << " T = " << T << " E = " << E;
  size_t coord_sys = 0;
  for (auto &counts : c)
    add_experiment(counts, same_coord_sys ? 0 : coord_sys++);
  update_contributions();

  if (parameters.targeted(Target::global))
    for (auto &x : phi_r)
      x = exp(0.1 * std::normal_distribution<double>()(EntropySource::rng));

  initialize_coordinate_systems(1);

  enforce_positive_parameters();
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

void Model::store(const string &prefix_, bool reorder) const {
  string prefix = parameters.output_directory + prefix_;
  if (not boost::filesystem::create_directory(prefix))
    throw(std::runtime_error("Couldn't create directory " + prefix));
  auto factor_names = form_factor_names(T);
  auto &gene_names = experiments.begin()->counts.row_names;

  auto exp_gene_type = expected_gene_type();
  vector<size_t> order;
  if (reorder) {
    auto cs = colSums<Vector>(exp_gene_type);
    order = get_order(cs);
  }

#pragma omp parallel sections if (DO_PARALLEL)
  {
#pragma omp section
    write_matrix(exp_gene_type, prefix + "expected-features" + FILENAME_ENDING,
                 parameters.compression_mode, gene_names, factor_names, order);
#pragma omp section
    write_matrix(phi_r, prefix + "feature-gamma_prior-r" + FILENAME_ENDING,
                 parameters.compression_mode, gene_names, factor_names, order);
#pragma omp section
    write_matrix(phi_p, prefix + "feature-gamma_prior-p" + FILENAME_ENDING,
                 parameters.compression_mode, gene_names, factor_names, order);
#pragma omp section
    {
      const size_t C = coordinate_systems.size();
      const size_t num_digits = 1 + floor(log(C) / log(10));
      for (size_t c = 0; c < C; ++c) {
        vector<string> rn;
        for(size_t n = 0; n < coordinate_systems[c].N; ++n)
          rn.push_back(to_string(n));
        write_matrix(coordinate_systems[c].field,
                     prefix + "field" + to_string_embedded(c, num_digits)
                         + FILENAME_ENDING,
                     parameters.compression_mode, rn, factor_names, order);
      }
    }
#pragma omp section
    mix_prior.store(prefix + "mixprior", factor_names, order);
#pragma omp section
    write_matrix(contributions_gene_type,
                 prefix + "contributions_gene_type" + FILENAME_ENDING,
                 parameters.compression_mode, gene_names, factor_names, order);
#pragma omp section
    write_vector(contributions_gene,
                 prefix + "contributions_gene" + FILENAME_ENDING,
                 parameters.compression_mode, gene_names);
#pragma omp section
    {
      auto print_field_matrix = [&](const string &path, const Vector &weights) {
        ofstream ofs(path);
        ofs << "coord_sys\tpoint_idx";
        for (size_t d = 0; d < coordinate_systems[0].mesh.dim; ++d)
          ofs << "\tx" << d;
        for (size_t t = 0; t < T; ++t)
          ofs << "\tFactor " << t + 1;
        ofs << endl;

        size_t coord_sys_idx = 0;
        for (size_t c = 0; c < coordinate_systems.size(); ++c) {
          for (size_t n = 0; n < coordinate_systems[c].N; ++n) {
            // print coordinate system index
            ofs << coord_sys_idx << "\t" << n;
            // print coordinate
            for (size_t d = 0; d < coordinate_systems[c].mesh.dim; ++d)
              ofs << "\t" << coordinate_systems[c].mesh.points[n][d];
            // print weighted field values of factors in order
            for (size_t t = 0; t < T; ++t)
              ofs << "\t"
                  << coordinate_systems[c].field(n, order[t])
                         * weights[order[t]];
            ofs << endl;
          }
          coord_sys_idx++;
        }
      };
      Vector w = Vector::Ones(T);
      // print un-weighted field
      print_field_matrix(prefix + "field" + FILENAME_ENDING, w);

      // NOTE we ignore local features and local baseline
      w = mix_prior.r.array() / mix_prior.p.array();
      Vector meanColSums = colSums<Vector>(phi_r.array() / phi_p.array());
      for (size_t t = 0; t < T; ++t)
        w(t) *= meanColSums(t);
      // print weighted field
      print_field_matrix(prefix + "expfield" + FILENAME_ENDING, w);
    }
  }
  for (size_t e = 0; e < E; ++e) {
    string exp_prefix = prefix + "experiment"
                        + to_string_embedded(e, EXPERIMENT_NUM_DIGITS) + "-";
    experiments[e].store(exp_prefix, order);
  }
}

void Model::restore(const string &prefix) {
  phi_r = parse_file<Matrix>(prefix + "feature-gamma_prior-r" + FILENAME_ENDING,
                             read_matrix, "\t");
  phi_p = parse_file<Matrix>(prefix + "feature-gamma_prior-p" + FILENAME_ENDING,
                             read_matrix, "\t");

  {
    const size_t C = coordinate_systems.size();
    const size_t num_digits = 1 + floor(log(C) / log(10));
    for (size_t c = 0; c < C; ++c)
      coordinate_systems[c].field = parse_file<Matrix>(
          prefix + "field" + to_string_embedded(c, num_digits)
              + FILENAME_ENDING,
          read_matrix, "\t");
  }

  mix_prior.restore(prefix + "mixprior");

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

Matrix Model::field_fitness_posterior_gradient() const {
  Matrix grad(S, T);
  size_t cumul = 0;
  for (auto &experiment : experiments) {
    grad.middleRows(cumul, experiment.S)
        = experiment.field_fitness_posterior_gradient();
    cumul += experiment.S;
  }
  return grad;
}

void Model::set_zero() {
  phi_r.setZero();
  phi_p.setZero();
  for (auto &coord_sys : coordinate_systems)
    coord_sys.field.setZero();
  mix_prior.r.setZero();
  mix_prior.p.setZero();
  for (auto &experiment : experiments)
    experiment.set_zero();
}

size_t Model::size() const {
  size_t s = 0;
  if (parameters.targeted(Target::global))
    s += phi_r.size();
  if (parameters.targeted(Target::variance))
    s += phi_p.size();
  if (parameters.targeted(Target::theta_prior))
    s += mix_prior.r.size() + mix_prior.p.size();
  if (parameters.targeted(Target::field))
    for (auto &coord_sys : coordinate_systems)
      s += coord_sys.field.size();
  for (auto &experiment : experiments)
    s += experiment.size();
  return s;
}

Vector Model::vectorize() const {
  Vector v(size());
  auto iter = begin(v);

  if (parameters.targeted(Target::global))
    for (auto &x : phi_r)
      *iter++ = x;

  if (parameters.targeted(Target::variance))
    for (auto &x : phi_p)
      *iter++ = x;

  if (parameters.targeted(Target::field))
    for (auto &coord_sys : coordinate_systems)
      for (auto &x : coord_sys.field)
        *iter++ = x;

  if (parameters.targeted(Target::theta_prior)) {
    for (auto &x : mix_prior.r)
      *iter++ = x;
    for (auto &x : mix_prior.p)
      *iter++ = x;
  }

  for (auto &experiment : experiments)
    for (auto &x : experiment.vectorize())
      *iter++ = x;

  assert(iter == end(v));

  return v;
}

Model Model::compute_gradient(double &score) const {
  LOG(verbose) << "Computing gradient";

  score = 0;
  Model gradient = *this;
  gradient.set_zero();
  gradient.contributions_gene_type.setZero();
  for (auto &experiment : gradient.experiments) {
    experiment.contributions_spot_type.setZero();
    experiment.contributions_gene_type.setZero();
  }
  if (parameters.targeted(Target::global)
      or parameters.targeted(Target::variance)
      or parameters.targeted(Target::local)
      or parameters.targeted(Target::baseline)
      or parameters.targeted(Target::theta)
      or parameters.targeted(Target::spot))
#pragma omp parallel if (DO_PARALLEL)
  {
    Model grad = gradient;
    auto rng = EntropySource::rngs[omp_get_thread_num()];
    double score_ = 0;

#pragma omp for
    for (size_t g = 0; g < G; ++g)
      for (auto &coord_sys : coordinate_systems)
        for (auto e : coord_sys.members)
          for (size_t s = 0; s < experiments[e].S; ++s)
            if (RandomDistribution::Uniform(rng)
                >= parameters.dropout_gene_spot) {
              auto cnts
                  = experiments[e].sample_contributions_gene_spot(g, s, rng);
              for (size_t t = 0; t < T; ++t)
                score += log_negative_binomial(
                    cnts[t],
                    phi_r(g, t) * experiments[e].phi_b(g)
                        * experiments[e].phi_l(g, t)
                        * experiments[e].theta(s, t) * experiments[e].spot(s),
                    neg_odds_to_prob(phi_p(g, t)));
              register_gradient(g, e, s, cnts, grad);
            }
#pragma omp critical
    {
      gradient = gradient + grad;
      score += score_;
    }
  }

  gradient.update_contributions();

  if (parameters.targeted(Target::field))
    for (size_t c = 0; c < coordinate_systems.size(); ++c)
      score
          += field_gradient(coordinate_systems[c], coordinate_systems[c].field,
                            gradient.coordinate_systems[c].field);

  if (not parameters.ignore_priors)
    finalize_gradient(gradient);
  score += param_likel();

  return gradient;
}

void Model::register_gradient(size_t g, size_t e, size_t s, const Vector &cnts,
                              Model &gradient) const {
  for (size_t t = 0; t < T; ++t)
    gradient.experiments[e].contributions_gene_type(g, t) += cnts[t];
  for (size_t t = 0; t < T; ++t)
    gradient.experiments[e].contributions_spot_type(s, t) += cnts[t];

  for (size_t t = 0; t < T; ++t) {
    const double no = phi_p(g, t);
    const double p = neg_odds_to_prob(no);
    const double log_one_minus_p = odds_to_log_prob(no);
    const double r = phi_r(g, t) * experiments[e].phi_l(g, t)
                     * experiments[e].phi_b(g) * experiments[e].theta(s, t)
                     * experiments[e].spot(s);
    const double k = cnts[t];
    const double term = r * (log_one_minus_p + digamma_diff(r, k));

    gradient.phi_r(g, t) += term;
    gradient.experiments[e].phi_l(g, t) += term;
    gradient.experiments[e].phi_b(g) += term;
    gradient.experiments[e].theta(s, t) += term;
    gradient.experiments[e].spot(s) += term;

    gradient.phi_p(g, t) += p * (r + k) - k;
  }
}

void Model::finalize_gradient(Model &gradient) const {
  LOG(verbose) << "Finalizing gradient";

  if (parameters.targeted(Target::global)) {
    const double a = parameters.hyperparameters.phi_r_1;
    const double b = parameters.hyperparameters.phi_r_2;
#pragma omp parallel for if (DO_PARALLEL)
    for (size_t g = 0; g < G; ++g)
      for (size_t t = 0; t < T; ++t)
        gradient.phi_r(g, t) += (a - 1) - phi_r(g, t) * b;
  }

  if (parameters.targeted(Target::variance)) {
    const double a = parameters.hyperparameters.phi_p_1;
    const double b = parameters.hyperparameters.phi_p_2;
#pragma omp parallel for if (DO_PARALLEL)
    for (size_t g = 0; g < G; ++g)
      for (size_t t = 0; t < T; ++t) {
        const double no = phi_p(g, t);
        const double p = neg_odds_to_prob(no);
        gradient.phi_p(g, t) += (a + b - 2) * p - a + 1;
      }
  }

  if (parameters.targeted(Target::local)) {
    const double a = parameters.hyperparameters.local_phi_r_1;
    const double b = parameters.hyperparameters.local_phi_r_2;
    for (auto &coord_sys : coordinate_systems)
      for (auto e : coord_sys.members)
#pragma omp parallel for if (DO_PARALLEL)
        for (size_t g = 0; g < G; ++g)
          for (size_t t = 0; t < T; ++t)
            gradient.experiments[e].phi_l(g, t)
                += (a - 1) - experiments[e].phi_l(g, t) * b;
  }

  if (parameters.targeted(Target::baseline)) {
    const double a = parameters.hyperparameters.baseline_1;
    const double b = parameters.hyperparameters.baseline_2;
    for (auto &coord_sys : coordinate_systems)
      for (auto e : coord_sys.members)
#pragma omp parallel for if (DO_PARALLEL)
        for (size_t g = 0; g < G; ++g)
          gradient.experiments[e].phi_b(g)
              += (a - 1) - experiments[e].phi_b(g) * b;
  }

  if (parameters.targeted(Target::theta))
    for (auto &coord_sys : coordinate_systems)
      for (auto e : coord_sys.members)
#pragma omp parallel for if (DO_PARALLEL)
        for (size_t s = 0; s < experiments[e].S; ++s)
          for (size_t t = 0; t < T; ++t) {
            const double a = experiments[e].field(s, t) * mix_prior.r(t);
            const double b = mix_prior.p(t);
            gradient.experiments[e].theta(s, t)
                += (a - 1) - experiments[e].theta(s, t) * b;
          }

  if (parameters.targeted(Target::spot)) {
    const double a = parameters.hyperparameters.spot_a;
    const double b = parameters.hyperparameters.spot_b;
    for (auto &coord_sys : coordinate_systems)
      for (auto e : coord_sys.members)
#pragma omp parallel for if (DO_PARALLEL)
        for (size_t s = 0; s < experiments[e].S; ++s)
          gradient.experiments[e].spot(s)
              += (a - 1) - experiments[e].spot(s) * b;
  }

  if (parameters.targeted(Target::theta_prior)) {
    gradient.mix_prior.r.setZero();
    for (auto &coord_sys : coordinate_systems)
      for (auto e : coord_sys.members)
        for (size_t t = 0; t < T; ++t) {
          const double r = mix_prior.r(t);
          const double no = mix_prior.p(t);
          const double p = neg_odds_to_prob(no);
          const double log_no = log(no);

          double x = 0;
          double y = 0;
#pragma omp parallel for if (DO_PARALLEL)
          for (size_t s = 0; s < experiments[e].S; ++s) {
            const double prod = r * experiments[e].field(s, t);
            x += (log(experiments[e].theta(s, t)) + log_no - digamma(prod))
                 * prod;
            y += (1 - p) * (p * experiments[e].theta(s, t) - prod);
          }
          gradient.mix_prior.r(t) += x;
          gradient.mix_prior.p(t) += y;
        }

    {
      const double a = parameters.hyperparameters.theta_r_1;
      const double b = parameters.hyperparameters.theta_r_2;
      for (size_t t = 0; t < T; ++t)
        gradient.mix_prior.r(t) += a - 1 - b * mix_prior.r(t);
    }

    {
      const double a = parameters.hyperparameters.theta_p_1;
      const double b = parameters.hyperparameters.theta_p_2;
      for (size_t t = 0; t < T; ++t) {
        const double no = mix_prior.p(t);
        const double p = neg_odds_to_prob(no);
        gradient.mix_prior.p(t) += (a + b - 2) * p - a + 1;
      }
    }
  }
}

// calculate parameter's likelihood
double Model::param_likel() const {
  double score = 0;
  if (parameters.targeted(Target::global)) {
    const double a = parameters.hyperparameters.phi_r_1;
    const double b = parameters.hyperparameters.phi_r_2;
#pragma omp parallel for if (DO_PARALLEL)
    for (size_t g = 0; g < G; ++g)
      for (size_t t = 0; t < T; ++t)
        score += log_gamma_rate(phi_r(g, t), a, b);
  }

  if (parameters.targeted(Target::local)) {
    const double a = parameters.hyperparameters.local_phi_r_1;
    const double b = parameters.hyperparameters.local_phi_r_2;
    for (auto &coord_sys : coordinate_systems)
      for (auto e : coord_sys.members)
#pragma omp parallel for if (DO_PARALLEL)
        for (size_t g = 0; g < G; ++g)
          for (size_t t = 0; t < T; ++t)
            score += log_gamma_rate(experiments[e].phi_l(g, t), a, b);
  }

  if (parameters.targeted(Target::baseline)) {
    const double a = parameters.hyperparameters.baseline_1;
    const double b = parameters.hyperparameters.baseline_2;
    for (auto &coord_sys : coordinate_systems)
      for (auto e : coord_sys.members)
#pragma omp parallel for if (DO_PARALLEL)
        for (size_t g = 0; g < G; ++g)
          score += log_gamma_rate(experiments[e].phi_b(g), a, b);
  }

  if (parameters.targeted(Target::theta))
    for (auto &coord_sys : coordinate_systems)
      for (auto e : coord_sys.members)
#pragma omp parallel for if (DO_PARALLEL)
        for (size_t s = 0; s < experiments[e].S; ++s)
          for (size_t t = 0; t < T; ++t) {
            const double a = experiments[e].field(s, t) * mix_prior.r(t);
            const double b = mix_prior.p(t);
            score += log_gamma_rate(experiments[e].theta(s, t), a, b);
          }

  if (parameters.targeted(Target::spot)) {
    const double a = parameters.hyperparameters.spot_a;
    const double b = parameters.hyperparameters.spot_b;
    for (auto &coord_sys : coordinate_systems)
      for (auto e : coord_sys.members)
#pragma omp parallel for if (DO_PARALLEL)
        for (size_t s = 0; s < experiments[e].S; ++s)
          score += log_gamma_rate(experiments[e].spot(s), a, b);
  }

  if (parameters.targeted(Target::variance)) {
    const double a = parameters.hyperparameters.phi_p_1;
    const double b = parameters.hyperparameters.phi_p_2;
#pragma omp parallel for if (DO_PARALLEL)
    for (size_t g = 0; g < G; ++g)
      for (size_t t = 0; t < T; ++t) {
        const double no = phi_p(g, t);
        score += log_beta_neg_odds(no, a, b);
      }
  }

  if (parameters.targeted(Target::theta_prior)) {
    {
      const double a = parameters.hyperparameters.theta_r_1;
      const double b = parameters.hyperparameters.theta_r_2;
      for (size_t t = 0; t < T; ++t)
        score += log_gamma_rate(mix_prior.r(t), a, b);
    }
    {
      const double a = parameters.hyperparameters.theta_p_1;
      const double b = parameters.hyperparameters.theta_p_2;
      for (size_t t = 0; t < T; ++t)
        score += log_beta_neg_odds(mix_prior.p(t), a, b);
    }
  }

  return score;
}

void Model::gradient_update() {
  LOG(verbose) << "Performing gradient update iteration";

  size_t iter_cnt = 0;

  auto fnc = [&](const Vector &x, Vector &grad) {
    if (((iter_cnt++) % parameters.report_interval) == 0) {
      const size_t iteration_num_digits
          = 1 + floor(log(parameters.grad_iterations) / log(10));
      store("iter" + to_string_embedded(iter_cnt - 1, iteration_num_digits)
            + "/");
    }

    from_log_vector(begin(x));
    double score = 0;
    Model model_grad = compute_gradient(score);
    grad = model_grad.vectorize();
    contributions_gene_type = model_grad.contributions_gene_type;
    for (size_t e = 0; e < E; ++e) {
      experiments[e].contributions_spot_type
          = model_grad.experiments[e].contributions_spot_type;
      experiments[e].contributions_gene_type
          = model_grad.experiments[e].contributions_gene_type;
    }

    if (parameters.optim_method == Optimize::Method::lBFGS) {
      // as for lBFGS we want to minimize, we have to negate
      score = -score;
    }

    LOG(info) << "Iteration " << iter_cnt << ", score: " << score;
    LOG(verbose) << "x: " << endl << Stats::summary(x);
    LOG(verbose) << "grad: " << endl << Stats::summary(grad);

    return score;
  };

  Vector x = vectorize().array().log();

  double fx;
  switch (parameters.optim_method) {
    case Optimize::Method::RPROP: {
      Vector grad;
      Vector prev_sign(Vector::Zero(x.size()));
      Vector rates(x.size());
      rates.fill(parameters.grad_alpha);
      for (size_t iter = 0; iter < parameters.grad_iterations; ++iter) {
        fx = fnc(x, grad);
        rprop_update(grad, prev_sign, rates, x);
      }
    } break;
    case Optimize::Method::Gradient: {
      double alpha = parameters.grad_alpha;
      for (size_t iter = 0; iter < parameters.grad_iterations; ++iter) {
        Vector grad;
        fx = fnc(x, grad);
        x = x + alpha * grad;
        LOG(verbose) << "iter " << iter << " alpha: " << alpha;
        LOG(verbose) << "iter " << iter << " fx: " << fx;
        LOG(verbose) << "iter " << iter << " x: " << endl << Stats::summary(x);

        alpha *= parameters.grad_anneal;
      }
    } break;
    case Optimize::Method::lBFGS: {
      using namespace LBFGSpp;
      LBFGSParam<double> param;
      param.epsilon = parameters.lbfgs_epsilon;
      param.max_iterations = parameters.lbfgs_iter;
      // Create solver and function object
      LBFGSSolver<double> solver(param);

      int niter = solver.minimize(fnc, x, fx);

      LOG(verbose) << "LBFGS performed " << niter << " iterations";
    } break;
  }
  LOG(verbose) << "Final f(x) = " << fx;

  from_log_vector(begin(x));
}

double Model::field_gradient(const CoordinateSystem &coord_sys,
                             const Matrix &field, Matrix &grad) const {
  LOG(debug) << "field dim " << field.rows() << "x" << field.cols();
  double score = 0;

  Matrix fitness(coord_sys.S, T);
  size_t cumul = 0;
  for (auto member : coord_sys.members) {
    const size_t current_S = experiments[member].S;
    // no need to calculate fitness contribution to score
    // it is calculated in finalize_gradient()
    fitness.middleRows(cumul, current_S)
        = experiments[member].field_fitness_posterior_gradient();
    cumul += current_S;
  }

  LOG(debug) << "Fitness contribution to score: " << score;

  grad = Matrix::Zero(coord_sys.N, T);

  grad.topRows(coord_sys.S) = fitness;

  if (parameters.field_lambda_dirichlet != 0) {
    Matrix grad_dirichlet = Matrix::Zero(coord_sys.N, T);
    for (size_t t = 0; t < T; ++t) {
      grad_dirichlet.col(t)
          = coord_sys.mesh.grad_dirichlet_energy(Vector(field.col(t)));

      double s = coord_sys.mesh.sum_dirichlet_energy(Vector(field.col(t)));
      score -= s * parameters.field_lambda_dirichlet;
      LOG(debug) << "Smoothness contribution to score of factor " << t << ": "
                 << s;
    }
    grad -= grad_dirichlet * parameters.field_lambda_dirichlet;
  }

  if (parameters.field_lambda_laplace != 0) {
    Matrix grad_laplace = Matrix::Zero(coord_sys.N, T);
    for (size_t t = 0; t < T; ++t) {
      grad_laplace.col(t)
          = coord_sys.mesh.grad_sq_laplace_operator(Vector(field.col(t)));

      double s = coord_sys.mesh.sum_sq_laplace_operator(Vector(field.col(t)));
      score -= s * parameters.field_lambda_laplace;
      LOG(debug) << "Curvature contribution to score of factor " << t << ": "
                 << s;
    }
    grad -= grad_laplace * parameters.field_lambda_laplace;
  }

  LOG(debug) << "Field score: " << score;

  return score;
}

void Model::enforce_positive_parameters() {
  enforce_positive_and_warn("phi_r", phi_r);
  enforce_positive_and_warn("phi_p", phi_p);
  enforce_positive_and_warn("mix_prior_r", mix_prior.r);
  for (auto &coord_sys : coordinate_systems)
    enforce_positive_and_warn("field", coord_sys.field);
  enforce_positive_and_warn("mix_prior_p", mix_prior.p);
  enforce_positive_and_warn("phi_p", phi_p);
  for (auto &experiment : experiments)
    experiment.enforce_positive_parameters();
}

double Model::log_likelihood(const string &prefix) const {
  double l = 0;
  for (size_t e = 0; e < E; ++e) {
    Matrix m = experiments[e].log_likelihood();

    auto &gene_names = experiments[e].counts.row_names;
    auto &spot_names = experiments[e].counts.col_names;

    string exp_prefix = prefix + "experiment"
                        + to_string_embedded(e, EXPERIMENT_NUM_DIGITS) + "-";
    write_matrix(m, exp_prefix + "loglikelihood" + FILENAME_ENDING,
                 parameters.compression_mode, gene_names, spot_names);
    write_matrix(*experiments[e].counts.matrix,
                 exp_prefix + "loglikelihood-counts" + FILENAME_ENDING,
                 parameters.compression_mode, gene_names, spot_names);
    for (auto &x : m)
      l += x;
  }
  return l;
}

// computes a matrix M(g,t)
// with M(g,t) = phi(g,t) sum_e local_baseline_phi(e,g) local_phi(e,g,t) sum_s
// theta(e,s,t) sigma(e,s)
Matrix Model::expected_gene_type() const {
  Matrix m = Matrix::Zero(G, T);
  for (auto &experiment : experiments)
    m += experiment.expected_gene_type();
  return m;
}

void Model::update_experiment_fields() {
  LOG(verbose) << "Updating experiment fields";
  for (auto &coord_sys : coordinate_systems) {
    size_t cumul = 0;
    for (auto member : coord_sys.members) {
      experiments[member].field
          = coord_sys.field.middleRows(cumul, experiments[member].S);
      cumul += experiments[member].S;
    }
  }
}

void Model::update_contributions() {
  contributions_gene_type.setZero();
  contributions_gene.setZero();
  for (auto &experiment : experiments)
#pragma omp parallel for if (DO_PARALLEL)
    for (size_t g = 0; g < G; ++g) {
      contributions_gene(g) += experiment.contributions_gene(g);
      for (size_t t = 0; t < T; ++t)
        contributions_gene_type(g, t)
            += experiment.contributions_gene_type(g, t);
    }
}

void Model::initialize_coordinate_systems(double v) {
  size_t coord_sys_idx = 0;
  for (auto &coord_sys : coordinate_systems) {
    size_t num_additional = parameters.mesh_additional;
    coord_sys.S = 0;
    for (auto &member : coord_sys.members)
      coord_sys.S += experiments[member].S;
    if (coord_sys.S == 0)
      num_additional = 0;
    coord_sys.N = coord_sys.S + num_additional;
    coord_sys.T = T;
    coord_sys.field = Matrix(coord_sys.N, T);
    coord_sys.field.fill(v);

    using Point = Vector;
    size_t dim = experiments[coord_sys.members[0]].coords.cols();
    vector<Point> pts;
    {
      Point pt(dim);
      for (auto &member : coord_sys.members)
        for (size_t s = 0; s < experiments[member].S; ++s) {
          for (size_t i = 0; i < dim; ++i)
            pt[i] = experiments[member].coords(s, i);
          pts.push_back(pt);
        }

      if (num_additional > 0) {
        Point mi = pts[0];
        Point ma = pts[0];
        for (auto &p : pts)
          for (size_t d = 0; d < dim; ++d) {
            if (p[d] < mi[d])
              mi[d] = p[d];
            if (p[d] > ma[d])
              ma[d] = p[d];
          }
        if (parameters.mesh_hull_distance <= 0) {
          Point mid = (ma + mi) / 2;
          Point half_diff = (ma - mi) / 2;
          mi = mid - half_diff * parameters.mesh_hull_enlarge;
          ma = mid + half_diff * parameters.mesh_hull_enlarge;
        } else {
          mi.array() -= parameters.mesh_hull_distance;
          ma.array() += parameters.mesh_hull_distance;
        }
        for (size_t i = 0; i < num_additional; ++i) {
          // TODO improve this
          // e.g. enlarge area, use convex hull instead of bounding box, etc
          bool ok = false;
          while (not ok) {
            for (size_t d = 0; d < dim; ++d)
              pt[d] = mi[d]
                      + (ma[d] - mi[d])
                            * RandomDistribution::Uniform(EntropySource::rng);
            for (size_t s = 0; s < coord_sys.S; ++s)
              if ((pt - pts[s]).norm() < parameters.mesh_hull_distance) {
                ok = true;
                break;
              }
          }

          pts.push_back(pt);
        }
      }
    }
    coord_sys.mesh
        = Mesh(dim, pts,
               parameters.output_directory + "coordsys"
                   + to_string_embedded(coord_sys_idx, EXPERIMENT_NUM_DIGITS));
    coord_sys_idx++;
    S += coord_sys.S;
  }
}

void Model::add_experiment(const Counts &counts, size_t coord_sys) {
  experiments.push_back({this, counts, T, parameters});
  E++;
  while (coordinate_systems.size() <= coord_sys)
    coordinate_systems.push_back({});
  coordinate_systems[coord_sys].members.push_back(E - 1);
}

ostream &operator<<(ostream &os, const Model &model) {
  os << "Spatial Transcriptome Deconvoltuion "
     << "G = " << model.G << " "
     << "T = " << model.T << " "
     << "E = " << model.E << endl;

  /*
  if (verbosity >= Verbosity::debug) {
    print_matrix_head(os, model.features.matrix, "Φ");
    os << model.phi;
  }
  */
  for (auto &experiment : model.experiments)
    os << experiment;

  return os;
}

Model operator*(const Model &a, const Model &b) {
  Model model = a;

  model.contributions_gene_type.array() *= b.contributions_gene_type.array();
  model.contributions_gene.array() *= b.contributions_gene.array();
  model.phi_r.array() *= b.phi_r.array();
  model.phi_p.array() *= b.phi_p.array();
  for (size_t e = 0; e < model.E; ++e)
    model.experiments[e] = model.experiments[e] * b.experiments[e];

  return model;
}

Model operator+(const Model &a, const Model &b) {
  Model model = a;

  model.contributions_gene_type += b.contributions_gene_type;
  model.contributions_gene += b.contributions_gene;
  model.phi_r += b.phi_r;
  model.phi_p += b.phi_p;
  for (size_t e = 0; e < model.E; ++e)
    model.experiments[e] = model.experiments[e] + b.experiments[e];

  return model;
}

Model operator-(const Model &a, const Model &b) {
  Model model = a;

  model.contributions_gene_type -= b.contributions_gene_type;
  model.contributions_gene -= b.contributions_gene;
  model.phi_r -= b.phi_r;
  model.phi_p -= b.phi_p;
  for (size_t e = 0; e < model.E; ++e)
    model.experiments[e] = model.experiments[e] - b.experiments[e];

  return model;
}

Model operator*(const Model &a, double x) {
  Model model = a;

  model.contributions_gene_type *= x;
  model.contributions_gene *= x;
  model.phi_r *= x;
  model.phi_p *= x;
  for (auto &experiment : model.experiments)
    experiment = experiment * x;

  return model;
}

Model operator/(const Model &a, double x) {
  Model model = a;

  model.contributions_gene_type /= x;
  model.contributions_gene /= x;
  model.phi_r /= x;
  model.phi_p /= x;
  for (auto &experiment : model.experiments)
    experiment = experiment / x;

  return model;
}
}
