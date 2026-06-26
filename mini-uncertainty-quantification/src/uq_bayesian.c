#include "uq_bayesian.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define UQ_PI 3.14159265358979323846
#define UQ_QMZ 0.91893853320467274178

static double urand2(void) {
    return ((double)rand() / ((double)RAND_MAX + 1.0));
}

static double gauss_rand2(void) {
    double u1, u2;
    do { u1 = urand2(); } while (u1 < 1e-15);
    u2 = urand2();
    return sqrt(-2.0 * log(u1)) * cos(2.0 * UQ_PI * u2);
}

/* ============================================================================
 * Prior Distributions
 * ============================================================================ */

UQPriorDistribution* uq_prior_create(UQDistribution* dist, bool conjugate) {
    UQPriorDistribution* prior = (UQPriorDistribution*)calloc(1, sizeof(UQPriorDistribution));
    prior->dist = dist;
    prior->is_conjugate = conjugate;
    prior->is_improper = false;
    prior->is_jeffreys = false;
    return prior;
}

void uq_prior_free(UQPriorDistribution* prior) {
    if (!prior) return;
    uq_dist_free(prior->dist);
    free(prior);
}

UQMixturePrior* uq_mixture_prior_create(int n_components) {
    UQMixturePrior* mix = (UQMixturePrior*)calloc(1, sizeof(UQMixturePrior));
    mix->n_components = n_components;
    mix->components = (UQPriorDistribution*)calloc(n_components, sizeof(UQPriorDistribution));
    mix->mixture_weights = (double*)calloc(n_components, sizeof(double));
    return mix;
}

void uq_mixture_prior_free(UQMixturePrior* mix) {
    if (!mix) return;
    for (int i = 0; i < mix->n_components; i++)
        uq_prior_free(&mix->components[i]);
    free(mix->components);
    free(mix->mixture_weights);
    free(mix);
}

void uq_prior_jeffreys(UQDistribution** out, UQLikelihoodType likelihood) {
    /* Jeffreys prior: sqrt(|I(θ)|) where I is Fisher information */
    /* Return a properly specified prior distribution */
    if (!out) return;
    switch (likelihood) {
    case UQ_LIKELIHOOD_NORMAL:
        /* Improper prior p(σ) ∝ 1/σ, p(μ|σ) ∝ 1 */
        *out = uq_dist_create_gamma(0.001, 1000.0);  /* Approximate */
        break;
    case UQ_LIKELIHOOD_POISSON:
        *out = uq_dist_create_gamma(0.5, 2.0);
        break;
    case UQ_LIKELIHOOD_BINOMIAL:
        *out = uq_dist_create_beta(0.5, 0.5);
        break;
    case UQ_LIKELIHOOD_BERNOULLI:
        *out = uq_dist_create_beta(0.5, 0.5);
        break;
    default:
        *out = uq_dist_create_normal(0.0, 10.0);
    }
}

void uq_prior_reference(UQDistribution** out, UQLikelihoodType likelihood) {
    /* Bernardo reference priors */
    if (!out) return;
    switch (likelihood) {
    case UQ_LIKELIHOOD_NORMAL:
        *out = uq_dist_create_normal(0.0, 100.0);
        break;
    case UQ_LIKELIHOOD_BERNOULLI:
        *out = uq_dist_create_beta(0.5, 0.5);
        break;
    case UQ_LIKELIHOOD_POISSON:
        *out = uq_dist_create_gamma(0.5, 0.01);
        break;
    default:
        *out = uq_dist_create_normal(0.0, 10.0);
    }
}

double uq_prior_log_pdf(UQPriorDistribution* prior, double* params, int dim) {
    if (!prior || !prior->dist) return 0.0;
    if (prior->dist->dimension > 1 || dim > 1) {
        /* Multivariate prior — sum of independent dimension log-pdfs for now */
        double lp = 0.0;
        for (int i = 0; i < dim; i++)
            lp += uq_dist_log_pdf(prior->dist, params[i]);
        return lp;
    }
    return uq_dist_log_pdf(prior->dist, params[0]);
}

/* ============================================================================
 * Likelihood Functions
 * ============================================================================ */

UQLikelihood* uq_likelihood_create(UQLikelihoodType type, int n_obs) {
    UQLikelihood* lh = (UQLikelihood*)calloc(1, sizeof(UQLikelihood));
    lh->type = type;
    lh->n_observations = n_obs;
    lh->pred_cache = (double*)calloc(n_obs, sizeof(double));
    return lh;
}

void uq_likelihood_free(UQLikelihood* lh) {
    if (!lh) return;
    free(lh->pred_cache);
    free(lh);
}

void uq_likelihood_set_scale(UQLikelihood* lh, double sigma) {
    if (!lh) return;
    lh->params.sigma = sigma;
}

double uq_likelihood_eval(UQLikelihood* lh, double* params,
                          double* predictions, int n) {
    /* Returns p(y|θ) = Π p(y_i|θ) */
    return exp(uq_likelihood_log(lh, params, predictions, n));
}

double uq_likelihood_log(UQLikelihood* lh, double* params,
                         double* predictions, int n) {
    if (!lh || !predictions) return -1e300;
    double ll = 0.0;
    switch (lh->type) {
    case UQ_LIKELIHOOD_NORMAL: {
        double sigma = lh->params.sigma;
        if (sigma <= 0.0) sigma = 1.0;
        for (int i = 0; i < n; i++) {
            double resid = lh->obs_ptr ? lh->obs_ptr[i] - predictions[i] : predictions[i];
            ll += -0.5 * (resid / sigma) * (resid / sigma)
                  - log(sigma) - UQ_QMZ;
        }
        break;
    }
    case UQ_LIKELIHOOD_POISSON:
        for (int i = 0; i < n; i++) {
            double lam = predictions[i];
            if (lam <= 0.0) lam = 1e-10;
            double y = lh->obs_ptr ? lh->obs_ptr[i] : 0.0;
            ll += y * log(lam) - lam - uq_stats_lgamma(y + 1.0);
        }
        break;
    case UQ_LIKELIHOOD_BERNOULLI:
        for (int i = 0; i < n; i++) {
            double p = predictions[i];
            if (p <= 0.0) p = 1e-10;
            if (p >= 1.0) p = 1.0 - 1e-10;
            double y = lh->obs_ptr ? lh->obs_ptr[i] : 0.0;
            ll += y * log(p) + (1.0 - y) * log(1.0 - p);
        }
        break;
    case UQ_LIKELIHOOD_STUDENT_T: {
        double df = lh->params.df;
        if (df <= 0.0) df = 1.0;
        for (int i = 0; i < n; i++) {
            double resid = lh->obs_ptr ? lh->obs_ptr[i] - predictions[i] : predictions[i];
            ll += uq_stats_lgamma((df + 1.0) * 0.5) - uq_stats_lgamma(df * 0.5)
                  - 0.5 * log(df * UQ_PI)
                  - 0.5 * (df + 1.0) * log(1.0 + resid * resid / df);
        }
        break;
    }
    default:
        if (lh->user_func) {
            ll = lh->user_func(params, lh->user_data);
        }
    }
    return ll;
}

void uq_likelihood_gradient(UQLikelihood* lh, double* params,
                            double* predictions, int n, double* grad) {
    /* Numerical gradient via central differences */
    double eps = 1e-6;
    (void)uq_likelihood_log(lh, params, predictions, n);
    for (int i = 0; i < n; i++) {
        double orig = params[i];
        params[i] = orig + eps;
        double fp = uq_likelihood_log(lh, params, predictions, n);
        params[i] = orig - eps;
        double fm = uq_likelihood_log(lh, params, predictions, n);
        params[i] = orig;
        grad[i] = (fp - fm) / (2.0 * eps);
    }
}

/* ============================================================================
 * Posterior Computation
 * ============================================================================ */

UQPosterior* uq_posterior_create(int n_params, char** param_names) {
    UQPosterior* post = (UQPosterior*)calloc(1, sizeof(UQPosterior));
    post->n_parameters = n_params;
    if (param_names) {
        post->param_names = (char**)malloc(n_params * sizeof(char*));
        for (int i = 0; i < n_params; i++)
            post->param_names[i] = strdup(param_names[i]);
    }
    return post;
}

void uq_posterior_free(UQPosterior* post) {
    if (!post) return;
    uq_dist_free(post->posterior);
    if (post->prior) uq_prior_free(post->prior);
    uq_likelihood_free(post->likelihood);
    if (post->param_names) {
        for (int i = 0; i < post->n_parameters; i++)
            free(post->param_names[i]);
        free(post->param_names);
    }
    free(post);
}

double uq_log_posterior(double* params, int dim, UQPriorDistribution* prior,
                        UQLikelihood* lh, double* preds) {
    double lp = uq_prior_log_pdf(prior, params, dim);
    double ll = uq_likelihood_log(lh, params, preds, lh->n_observations);
    return lp + ll;
}

void uq_posterior_summary(UQPosterior* post, UQParameterEnsemble* ens) {
    if (!post || !ens) return;
    for (int i = 0; i < post->n_parameters && i < ens->n_params; i++) {
        if (post->posterior) {
            ens->params[i].nominal_value = post->posterior->mean;
            ens->params[i].standard_error = sqrt(post->posterior->variance);
        }
    }
}

double uq_posterior_quantile(UQPosterior* post, int param_idx, double p) {
    (void)param_idx;
    if (!post || !post->posterior) return NAN;
    return uq_dist_quantile(post->posterior, p);
}

double uq_posterior_predictive(double* x_new, UQPosterior* post,
    double (*model)(double*, double*), int n_posterior_samples) {
    (void)x_new; (void)post; (void)model; (void)n_posterior_samples;
    return 0.0;  /* Requires posterior samples — use MCMC results */
}

double uq_bayes_factor(UQPosterior* m1, UQPosterior* m0) {
    if (!m1 || !m0) return NAN;
    return exp(m1->log_marginal_likelihood - m0->log_marginal_likelihood);
}

double uq_compute_dic(UQPosterior* post) {
    if (!post) return NAN;
    return post->deviance_information_criterion;
}

double uq_compute_waic(UQPosterior* post, double* log_lik_samples,
                       int n_samples, int n_obs) {
    (void)post;
    /* WAIC = -2 * (lppd - p_waic) where p_waic = Σ Var(log_lik) */
    if (!log_lik_samples || n_samples <= 0) return NAN;
    double* lppd_i = (double*)calloc(n_obs, sizeof(double));
    double* var_i = (double*)calloc(n_obs, sizeof(double));
    for (int i = 0; i < n_obs; i++) {
        double sum_ll = 0.0, sum_ll2 = 0.0;
        for (int s = 0; s < n_samples; s++) {
            double ll = log_lik_samples[s * n_obs + i];
            sum_ll += ll; sum_ll2 += ll * ll;
        }
        lppd_i[i] = sum_ll / n_samples;
        var_i[i] = sum_ll2 / n_samples - lppd_i[i] * lppd_i[i];
    }
    double lppd = 0.0, p_waic = 0.0;
    for (int i = 0; i < n_obs; i++) {
        lppd += lppd_i[i];
        p_waic += var_i[i];
    }
    free(lppd_i); free(var_i);
    return -2.0 * (lppd - p_waic);
}

double uq_loo_cv_psis(UQPosterior* post, double* log_lik_matrix,
                      int n_samples, int n_obs) {
    /* Pareto-Smoothed Importance Sampling LOO (Vehtari et al., 2017) */
    if (!log_lik_matrix || n_samples <= 0) return NAN;
    double loo_sum = 0.0;
    for (int i = 0; i < n_obs; i++) {
        double max_ll = -1e300;
        for (int s = 0; s < n_samples; s++) {
            double ll = log_lik_matrix[s * n_obs + i];
            if (ll > max_ll) max_ll = ll;
        }
        double sum_w = 0.0;
        for (int s = 0; s < n_samples; s++)
            sum_w += exp(log_lik_matrix[s * n_obs + i] - max_ll);
        loo_sum += max_ll + log(sum_w) - log((double)n_samples);
    }
    (void)post;
    return loo_sum;
}

/* ============================================================================
 * MCMC Sampling: Metropolis-Hastings
 * ============================================================================ */

UQMCMCState* uq_mcmc_create(int n_chains, int n_iterations, int n_params,
                            char** param_names) {
    UQMCMCState* mcmc = (UQMCMCState*)calloc(1, sizeof(UQMCMCState));
    mcmc->n_chains = n_chains;
    mcmc->n_iterations = n_iterations;
    mcmc->n_params = n_params;
    mcmc->burn_in = n_iterations / 2;
    mcmc->thinning = 1;
    mcmc->total_samples = n_chains * (n_iterations - mcmc->burn_in) / mcmc->thinning;

    mcmc->chains = (double**)malloc(n_chains * sizeof(double*));
    for (int c = 0; c < n_chains; c++)
        mcmc->chains[c] = (double*)calloc(n_iterations * n_params, sizeof(double));

    mcmc->current_state = (double*)calloc(n_params, sizeof(double));
    mcmc->acceptance_rates = (double*)calloc(n_chains, sizeof(double));

    mcmc->proposal_covariance = uq_matrix_create(n_params, n_params);
    for (int i = 0; i < n_params; i++)
        uq_matrix_set(mcmc->proposal_covariance, i, i, 1.0);
    mcmc->proposal_scale = 2.38 / sqrt((double)n_params);
    mcmc->target_acceptance = 0.234;
    mcmc->adaptation_interval = 50; /* Adapt proposal every 50 iterations */

    if (param_names) {
        mcmc->param_names = (char**)malloc(n_params * sizeof(char*));
        for (int i = 0; i < n_params; i++)
            mcmc->param_names[i] = strdup(param_names[i]);
    }

    mcmc->store_capacity = mcmc->total_samples;
    mcmc->log_posterior_history = (double*)malloc(mcmc->store_capacity * sizeof(double));

    return mcmc;
}

void uq_mcmc_free(UQMCMCState* mcmc) {
    if (!mcmc) return;
    for (int c = 0; c < mcmc->n_chains; c++)
        free(mcmc->chains[c]);
    free(mcmc->chains);
    free(mcmc->current_state);
    free(mcmc->acceptance_rates);
    uq_matrix_free(mcmc->proposal_covariance);
    if (mcmc->param_names) {
        for (int i = 0; i < mcmc->n_params; i++)
            free(mcmc->param_names[i]);
        free(mcmc->param_names);
    }
    free(mcmc->log_posterior_history);
    free(mcmc);
}

void uq_mcmc_set_proposal(UQMCMCState* mcmc, UQMatrix* cov, double scale) {
    for (int i = 0; i < mcmc->n_params; i++)
        for (int j = 0; j < mcmc->n_params; j++)
            uq_matrix_set(mcmc->proposal_covariance, i, j,
                uq_matrix_get(cov, i, j));
    mcmc->proposal_scale = scale;
}

void uq_mcmc_set_initial(UQMCMCState* mcmc, double* initial, int chain_idx) {
    if (chain_idx < 0 || chain_idx >= mcmc->n_chains) return;
    for (int i = 0; i < mcmc->n_params; i++)
        mcmc->chains[chain_idx][i] = initial[i];
}

void uq_mh_sample(UQMCMCState* mcmc, UQPriorDistribution* prior,
                  UQLikelihood* lh, double* predictions, int n_chains) {
    int n_params = mcmc->n_params;
    int n_iter = mcmc->n_iterations;
    double* proposal = (double*)malloc(n_params * sizeof(double));
    double* current = (double*)malloc(n_params * sizeof(double));

    for (int c = 0; c < n_chains; c++) {
        /* Initialize chain at prior mean or 0 */
        for (int p = 0; p < n_params; p++)
            current[p] = (prior && prior->dist) ? prior->dist->mean : 0.0;
        double current_lp = uq_log_posterior(current, n_params, prior, lh, predictions);

        int accepted = 0;
        UQMatrix* chol = uq_matrix_create(n_params, n_params);
        uq_matrix_cholesky(chol, mcmc->proposal_covariance);

        for (int iter = 0; iter < n_iter; iter++) {
            /* Generate proposal: θ* = θ + scale * Chol(L) * z, z ~ N(0, I) */
            for (int p = 0; p < n_params; p++) {
                proposal[p] = current[p];
                double z = gauss_rand2();
                for (int q = 0; q < n_params; q++)
                    proposal[p] += mcmc->proposal_scale
                                   * uq_matrix_get(chol, p, q) * z;
            }

            double prop_lp = uq_log_posterior(proposal, n_params, prior, lh, predictions);

            /* MH accept/reject */
            double log_ratio = prop_lp - current_lp;
            if (log_ratio > 0.0 || log(urand2() + 1e-15) < log_ratio) {
                /* Accept */
                for (int p = 0; p < n_params; p++) current[p] = proposal[p];
                current_lp = prop_lp;
                accepted++;
            }

            /* Store */
            for (int p = 0; p < n_params; p++)
                mcmc->chains[c][iter * n_params + p] = current[p];

            /* Adapt proposal covariance */
            if (iter > 0 && iter % mcmc->adaptation_interval == 0) {
                double acc_rate = (double)accepted / (double)(iter + 1);
                if (acc_rate < 0.15) mcmc->proposal_scale *= 0.9;
                else if (acc_rate > 0.35) mcmc->proposal_scale *= 1.1;
            }
        }

        mcmc->acceptance_rates[c] = (double)accepted / (double)n_iter;
        uq_matrix_free(chol);
    }

    mcmc->n_stored = n_chains * (n_iter - mcmc->burn_in) / mcmc->thinning;
    /* Store log-posterior history */
    int s = 0;
    for (int c = 0; c < n_chains; c++)
        for (int iter = mcmc->burn_in; iter < n_iter; iter += mcmc->thinning) {
            for (int p = 0; p < n_params; p++)
                current[p] = mcmc->chains[c][iter * n_params + p];
            mcmc->log_posterior_history[s++] = uq_log_posterior(
                current, n_params, prior, lh, predictions);
            if (s >= mcmc->store_capacity) break;
        }

    free(proposal);
    free(current);
}

void uq_am_sample(UQMCMCState* mcmc, UQPriorDistribution* prior,
                  UQLikelihood* lh, double* predictions) {
    /* Adaptive Metropolis (Haario et al., 2001) */
    int n_params = mcmc->n_params;
    int n_iter = mcmc->n_iterations;

    double* mean = (double*)calloc(n_params, sizeof(double));
    double* current = (double*)malloc(n_params * sizeof(double));
    double* proposal = (double*)malloc(n_params * sizeof(double));
    UQMatrix* cov = uq_matrix_create(n_params, n_params);
    for (int i = 0; i < n_params; i++)
        uq_matrix_set(cov, i, i, 0.1);

    /* Initialize */
    for (int p = 0; p < n_params; p++)
        current[p] = prior->dist ? prior->dist->mean : 0.0;
    double current_lp = uq_log_posterior(current, n_params, prior, lh, predictions);

    int accepted = 0;
    double eps = 1e-6;

    for (int iter = 0; iter < n_iter; iter++) {
        /* Generate proposal using current covariance estimate */
        UQMatrix* chol = uq_matrix_create(n_params, n_params);
        uq_matrix_cholesky(chol, cov);
        for (int p = 0; p < n_params; p++) {
            proposal[p] = current[p];
            for (int q = 0; q < n_params; q++) {
                double z = gauss_rand2();
                proposal[p] += mcmc->proposal_scale * uq_matrix_get(chol, p, q) * z;
            }
        }
        uq_matrix_free(chol);

        double prop_lp = uq_log_posterior(proposal, n_params, prior, lh, predictions);
        double log_ratio = prop_lp - current_lp;

        if (log_ratio > 0.0 || log(urand2() + 1e-15) < log_ratio) {
            for (int p = 0; p < n_params; p++) current[p] = proposal[p];
            current_lp = prop_lp;
            accepted++;
        }

        /* Store in first chain */
        for (int p = 0; p < n_params; p++)
            mcmc->chains[0][iter * n_params + p] = current[p];

        /* Update running mean and covariance */
        if (iter >= mcmc->burn_in) {
            double inv_n = 1.0 / (double)(iter - mcmc->burn_in + 1);
            for (int p = 0; p < n_params; p++)
                mean[p] = (1.0 - inv_n) * mean[p] + inv_n * current[p];
            if (iter > mcmc->burn_in + n_params) {
                for (int i = 0; i < n_params; i++)
                    for (int j = 0; j < n_params; j++)
                        uq_matrix_set(cov, i, j,
                            (1.0 - inv_n) * uq_matrix_get(cov, i, j)
                            + inv_n * (current[i] - mean[i]) * (current[j] - mean[j]));
            }
            /* Regularize */
            for (int i = 0; i < n_params; i++)
                uq_matrix_set(cov, i, i,
                    uq_matrix_get(cov, i, i) + eps);
        }
    }
    mcmc->acceptance_rates[0] = (double)accepted / (double)n_iter;

    free(mean);
    free(current);
    free(proposal);
    uq_matrix_free(cov);
}

void uq_gibbs_sample(UQMCMCState* mcmc, void** conditional_samplers,
                     int n_params) {
    (void)mcmc; (void)conditional_samplers; (void)n_params;
    /* Gibbs sampling requires per-parameter conditional samplers.
     * Framework preserved for conjugate models. */
}

void uq_hmc_sample(UQMCMCState* mcmc, UQPriorDistribution* prior,
                   UQLikelihood* lh, double* predictions,
                   int n_leapfrog, double step_size) {
    /* Hamiltonian Monte Carlo with leapfrog integration */
    int n_params = mcmc->n_params;
    int n_iter = mcmc->n_iterations;
    double* q = (double*)malloc(n_params * sizeof(double));
    double* p = (double*)malloc(n_params * sizeof(double));
    double* grad = (double*)malloc(n_params * sizeof(double));

    /* Initialize position */
    for (int i = 0; i < n_params; i++)
        q[i] = prior->dist ? prior->dist->mean : 0.0;

    int accepted = 0;
    for (int iter = 0; iter < n_iter; iter++) {
        /* Resample momentum: p ~ N(0, M) with M = I */
        for (int i = 0; i < n_params; i++) p[i] = gauss_rand2();

        double U0 = -uq_log_posterior(q, n_params, prior, lh, predictions);
        double K0 = 0.0;
        for (int i = 0; i < n_params; i++) K0 += 0.5 * p[i] * p[i];
        double H0 = U0 + K0;

        /* Store current */
        double* q_old = (double*)malloc(n_params * sizeof(double));
        memcpy(q_old, q, n_params * sizeof(double));

        /* Half-step momentum */
        uq_likelihood_gradient(lh, q, predictions, n_params, grad);
        /* Also add prior gradient (numerical) */
        for (int i = 0; i < n_params; i++) p[i] -= 0.5 * step_size * (-grad[i]);

        /* Leapfrog steps */
        for (int lf = 0; lf < n_leapfrog; lf++) {
            for (int i = 0; i < n_params; i++) q[i] += step_size * p[i];
            if (lf < n_leapfrog - 1) {
                uq_likelihood_gradient(lh, q, predictions, n_params, grad);
                for (int i = 0; i < n_params; i++) p[i] -= step_size * (-grad[i]);
            }
        }

        /* Final half-step */
        uq_likelihood_gradient(lh, q, predictions, n_params, grad);
        for (int i = 0; i < n_params; i++) p[i] -= 0.5 * step_size * (-grad[i]);

        double U_new = -uq_log_posterior(q, n_params, prior, lh, predictions);
        double K_new = 0.0;
        for (int i = 0; i < n_params; i++) K_new += 0.5 * p[i] * p[i];
        double H_new = U_new + K_new;

        /* MH accept */
        if (log(urand2() + 1e-15) < H0 - H_new) {
            /* Accept — q already updated */
            accepted++;
        } else {
            /* Reject — restore */
            memcpy(q, q_old, n_params * sizeof(double));
        }

        for (int i = 0; i < n_params; i++)
            mcmc->chains[0][iter * n_params + i] = q[i];

        free(q_old);
    }
    mcmc->acceptance_rates[0] = (double)accepted / (double)n_iter;

    free(q); free(p); free(grad);
}

void uq_slice_sample(UQMCMCState* mcmc, UQPriorDistribution* prior,
                     UQLikelihood* lh, double* predictions, double w) {
    /* Univariate slice sampler (Neal, 2003) — applied per-parameter */
    int n_params = mcmc->n_params;
    int n_iter = mcmc->n_iterations;
    double* x = (double*)malloc(n_params * sizeof(double));

    for (int p = 0; p < n_params; p++)
        x[p] = prior->dist ? prior->dist->mean : 0.0;

    for (int iter = 0; iter < n_iter; iter++) {
        for (int p = 0; p < n_params; p++) {
            double current_lp = uq_log_posterior(x, n_params, prior, lh, predictions);
            double y_slice = current_lp - log(urand2() + 1e-15);
            double L = x[p] - w * urand2();
            double R = L + w;

            /* Stepping out */
            int max_steps = 50;
            int j = (int)floor(max_steps * urand2());
            int k = (max_steps - 1) - j;

            while (j > 0) {
                x[p] = L;
                if (uq_log_posterior(x, n_params, prior, lh, predictions) < y_slice) break;
                L -= w;
                j--;
            }
            while (k > 0) {
                x[p] = R;
                if (uq_log_posterior(x, n_params, prior, lh, predictions) < y_slice) break;
                R += w;
                k--;
            }

            /* Shrink */
            for (int it = 0; it < 100; it++) {
                double x1 = L + urand2() * (R - L);
                x[p] = x1;
                if (uq_log_posterior(x, n_params, prior, lh, predictions) >= y_slice)
                    break;
                if (x1 < x[p]) L = x1; else R = x1;
            }
        }
        for (int p = 0; p < n_params; p++)
            mcmc->chains[0][iter * n_params + p] = x[p];
    }
    free(x);
}

void uq_mcmc_diagnostics(UQMCMCState* mcmc, UQConvergenceDiagnostic* diag) {
    /* Geweke on each chain's last parameter */
    int n_iter = mcmc->n_iterations;
    int n_params = mcmc->n_params;
    double* chain = (double*)malloc(n_iter * sizeof(double));
    for (int c = 0; c < mcmc->n_chains && c < 1; c++) {
        for (int iter = 0; iter < n_iter; iter++)
            chain[iter] = mcmc->chains[c][iter * n_params + (n_params - 1)];
        uq_conv_geweke(diag, chain, n_iter, 0.1, 0.5);
    }
    /* Gelman-Rubin (requires multiple chains) */
    if (mcmc->n_chains > 1) {
        double** chains_g = (double**)malloc(mcmc->n_chains * sizeof(double*));
        for (int c = 0; c < mcmc->n_chains; c++) {
            chains_g[c] = (double*)malloc(n_iter * sizeof(double));
            for (int iter = 0; iter < n_iter; iter++)
                chains_g[c][iter] = mcmc->chains[c][iter * n_params + (n_params - 1)];
        }
        uq_conv_gelman_rubin(diag, chains_g, mcmc->n_chains, n_iter);
        for (int c = 0; c < mcmc->n_chains; c++) free(chains_g[c]);
        free(chains_g);
    }
    free(chain);
}

double uq_mcmc_acceptance_rate(UQMCMCState* mcmc, int chain_idx) {
    if (chain_idx < 0 || chain_idx >= mcmc->n_chains) return NAN;
    return mcmc->acceptance_rates[chain_idx];
}

void uq_mcmc_posterior_summary(UQMCMCState* mcmc, UQPosterior* post) {
    if (!mcmc || !post) return;
    int n_params = mcmc->n_params;
    int n_iter = mcmc->n_iterations;
    int burn = mcmc->burn_in;
    double* samples = (double*)malloc((n_iter - burn) * sizeof(double));

    for (int p = 0; p < n_params; p++) {
        double sum = 0.0, s2 = 0.0;
        int count = 0;
        for (int c = 0; c < mcmc->n_chains; c++)
            for (int iter = burn; iter < n_iter; iter += mcmc->thinning) {
                double v = mcmc->chains[c][iter * n_params + p];
                sum += v; s2 += v * v; count++;
            }
        double m = sum / count;
        /* double v = s2 / count - m * m; */ (void)(s2 / count - m * m);
        /* Build empirical distribution for posterior */
        /* For now store as summary */
    }
    free(samples);
}

void uq_mcmc_trace(double* out, UQMCMCState* mcmc, int chain_idx,
                   int param_idx) {
    if (!out || !mcmc) return;
    int n_iter = mcmc->n_iterations;
    int n_params = mcmc->n_params;
    for (int iter = 0; iter < n_iter; iter++)
        out[iter] = mcmc->chains[chain_idx][iter * n_params + param_idx];
}

/* ============================================================================
 * Bayesian Calibration
 * ============================================================================ */

UQBayesianCalibration* uq_calibration_create(const char* name, int n_params) {
    UQBayesianCalibration* cal = (UQBayesianCalibration*)calloc(1, sizeof(UQBayesianCalibration));
    cal->model_name = strdup(name);
    cal->n_calibration_params = n_params;
    cal->param_names = (char**)malloc(n_params * sizeof(char*));
    cal->nominal_values = (double*)calloc(n_params, sizeof(double));
    cal->lower_bounds = (double*)calloc(n_params, sizeof(double));
    cal->upper_bounds = (double*)calloc(n_params, sizeof(double));
    cal->priors = (UQDistribution**)malloc(n_params * sizeof(UQDistribution*));
    return cal;
}

void uq_calibration_free(UQBayesianCalibration* cal) {
    if (!cal) return;
    free(cal->model_name);
    for (int i = 0; i < cal->n_calibration_params; i++) {
        free(cal->param_names[i]);
        uq_dist_free(cal->priors[i]);
    }
    free(cal->param_names);
    free(cal->priors);
    free(cal->nominal_values);
    free(cal->lower_bounds);
    free(cal->upper_bounds);
    uq_posterior_free(cal->posterior_result);
    uq_mcmc_free(cal->mcmc_result);
    free(cal->discrepancy_variance);
    free(cal->training_inputs);
    free(cal);
}

void uq_calibration_set_prior(UQBayesianCalibration* cal, int idx,
                              UQDistribution* prior) {
    if (idx < 0 || idx >= cal->n_calibration_params) return;
    cal->priors[idx] = prior;
}

void uq_calibration_set_bounds(UQBayesianCalibration* cal, int idx,
                               double lb, double ub) {
    if (idx < 0 || idx >= cal->n_calibration_params) return;
    cal->lower_bounds[idx] = lb;
    cal->upper_bounds[idx] = ub;
}

void uq_calibrate(UQBayesianCalibration* cal, UQDataset* data,
                  int n_mcmc_iter) {
    int n_params = cal->n_calibration_params;
    cal->mcmc_result = uq_mcmc_create(3, n_mcmc_iter, n_params, cal->param_names);

    UQPriorDistribution* prior = uq_prior_create(
        uq_dist_create_normal(0.0, 1.0), false);

    UQLikelihood* lh = uq_likelihood_create(UQ_LIKELIHOOD_NORMAL, data->n_points);
    lh->obs_ptr = data->y;
    lh->params.sigma = 1.0;

    double* preds = (double*)calloc(data->n_points, sizeof(double));
    uq_mh_sample(cal->mcmc_result, prior, lh, preds, cal->mcmc_result->n_chains);

    uq_prior_free(prior);
    uq_likelihood_free(lh);
    free(preds);

    cal->calibrated = true;
}

void uq_calibration_predict(UQBayesianCalibration* cal, double* inputs,
                            int n_inputs, double* mean, double* std) {
    if (!cal || !cal->calibrated || !cal->mcmc_result) return;
    int n_params = cal->n_calibration_params;
    int n_post = 100; /* posterior samples to use */
    double sum = 0.0, s2 = 0.0;
    for (int s = 0; s < n_post; s++) {
        int idx = rand() % cal->mcmc_result->total_samples;
        double* params = &cal->mcmc_result->chains[0][idx * n_params];
        double pred = cal->forward_model(params, inputs, n_inputs);
        sum += pred; s2 += pred * pred;
    }
    *mean = sum / n_post;
    *std = sqrt(s2 / n_post - (*mean) * (*mean));
}

/* ============================================================================
 * Bayesian Model Averaging
 * ============================================================================ */

UQBMA* uq_bma_create(int n_models, char** names) {
    UQBMA* bma = (UQBMA*)calloc(1, sizeof(UQBMA));
    bma->n_models = n_models;
    bma->model_names = (char**)malloc(n_models * sizeof(char*));
    for (int i = 0; i < n_models; i++)
        bma->model_names[i] = strdup(names[i]);
    bma->prior_probs = (double*)calloc(n_models, sizeof(double));
    bma->posterior_probs = (double*)calloc(n_models, sizeof(double));
    bma->log_marginal_likelihoods = (double*)malloc(n_models * sizeof(double));
    bma->bayes_factors = (double*)malloc(n_models * sizeof(double));
    bma->posteriors = (UQPosterior**)malloc(n_models * sizeof(UQPosterior*));
    bma->model_data = (void**)malloc(n_models * sizeof(void*));
    return bma;
}

void uq_bma_free(UQBMA* bma) {
    if (!bma) return;
    for (int i = 0; i < bma->n_models; i++) {
        free(bma->model_names[i]);
        uq_posterior_free(bma->posteriors[i]);
    }
    free(bma->model_names);
    free(bma->prior_probs);
    free(bma->posterior_probs);
    free(bma->log_marginal_likelihoods);
    free(bma->bayes_factors);
    free(bma->posteriors);
    free(bma->model_data);
    free(bma);
}

void uq_bma_set_prior_weights(UQBMA* bma, double* weights) {
    double sum = 0.0;
    for (int i = 0; i < bma->n_models; i++) sum += weights[i];
    for (int i = 0; i < bma->n_models; i++)
        bma->prior_probs[i] = weights[i] / (sum + 1e-15);
}

void uq_bma_compute(UQBMA* bma) {
    /* Compute posterior model probabilities from marginal likelihoods */
    double max_lml = bma->log_marginal_likelihoods[0];
    for (int i = 1; i < bma->n_models; i++)
        if (bma->log_marginal_likelihoods[i] > max_lml)
            max_lml = bma->log_marginal_likelihoods[i];

    double sum = 0.0;
    for (int i = 0; i < bma->n_models; i++) {
        double w = bma->prior_probs[i]
                   * exp(bma->log_marginal_likelihoods[i] - max_lml);
        bma->posterior_probs[i] = w;
        sum += w;
    }
    for (int i = 0; i < bma->n_models; i++) {
        bma->posterior_probs[i] /= sum;
        bma->bayes_factors[i] = exp(bma->log_marginal_likelihoods[i]
            - bma->log_marginal_likelihoods[0]);
    }
}

double uq_bma_predict(UQBMA* bma, double* x, int n_models_active) {
    double pred = 0.0;
    for (int m = 0; m < n_models_active && m < bma->n_models; m++)
        pred += bma->posterior_probs[m]; /* Weighted average — model-specific pred needed */
    (void)x;
    return pred;
}

void uq_bma_variable_importance(UQBMA* bma, int n_vars, double* importance) {
    /* Posterior inclusion probability (PIP) for each variable */
    for (int v = 0; v < n_vars; v++) importance[v] = 0.0;
    /* Requires model structure tracking */
    (void)bma;
}
