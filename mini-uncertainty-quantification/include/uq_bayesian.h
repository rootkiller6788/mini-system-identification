#ifndef UQ_BAYESIAN_H
#define UQ_BAYESIAN_H

#include "uq_core.h"
#include <stdbool.h>

/* ============================================================================
 * Bayesian Uncertainty Quantification
 *
 * Key References:
 *   - Gelman, A., Carlin, J.B., Stern, H.S., & Rubin, D.B. (2013).
 *     "Bayesian Data Analysis" (3rd ed.). Chapman & Hall/CRC.
 *   - Robert, C.P. & Casella, G. (2004). "Monte Carlo Statistical Methods"
 *     (2nd ed.). Springer.
 *   - Kennedy, M.C. & O'Hagan, A. (2001). "Bayesian calibration of computer
 *     models." JRSS B, 63(3), 425-464.
 * ============================================================================ */

/* --- Prior Distribution Specification --- */

typedef struct {
    UQDistribution* dist;
    bool is_conjugate;         /* True if conjugate to likelihood */
    bool is_improper;          /* True if integrates to infinity */
    bool is_jeffreys;          /* True if Jeffreys reference prior */
    double log_marginal;       /* log p(data) for this prior */
    bool is_mixture;           /* True if mixture of priors */
    double mixture_weight;     /* Weight for mixture component */
} UQPriorDistribution;

typedef struct {
    UQPriorDistribution* components;
    int n_components;
    double* mixture_weights;    /* Normalized to sum to 1 */
} UQMixturePrior;

/* --- Likelihood Model --- */

typedef enum {
    UQ_LIKELIHOOD_NORMAL = 0,
    UQ_LIKELIHOOD_POISSON = 1,
    UQ_LIKELIHOOD_BINOMIAL = 2,
    UQ_LIKELIHOOD_BERNOULLI = 3,
    UQ_LIKELIHOOD_EXPONENTIAL = 4,
    UQ_LIKELIHOOD_GAMMA = 5,
    UQ_LIKELIHOOD_STUDENT_T = 6,
    UQ_LIKELIHOOD_MULTIVARIATE_NORMAL = 7,
    UQ_LIKELIHOOD_USER_DEFINED = 8
} UQLikelihoodType;

typedef double (*uq_likelihood_func)(double* params, void* data);

typedef struct {
    UQLikelihoodType type;
    union {
        double sigma;                    /* Normal likelihood scale */
        double df;                       /* Student-t degrees of freedom */
        UQMatrix* covariance;            /* Multivariate normal covariance */
    } params;
    uq_likelihood_func user_func;       /* For user-defined likelihood */
    void* user_data;
    int n_observations;
    double* obs_ptr;                    /* Observation vector */
    double* pred_cache;                 /* Cache for predictions */
} UQLikelihood;

/* --- Posterior Distribution --- */

typedef struct {
    UQDistribution* posterior;           /* Posterior approximation */
    UQPriorDistribution* prior;
    UQLikelihood* likelihood;
    double log_marginal_likelihood;      /* log p(y) = ∫ p(y|θ)p(θ)dθ */
    double log_bayes_factor;             /* vs. null model */
    double deviance_information_criterion; /* DIC */
    double widely_applicable_ic;          /* WAIC */
    double loo_cv;                       /* Leave-one-out cross-validation */
    int n_parameters;
    char** param_names;
} UQPosterior;

/* --- Markov Chain Monte Carlo (MCMC) --- */

typedef struct {
    int n_chains;
    int n_iterations;
    int burn_in;
    int thinning;
    double** chains;                /* chains[chain][step * dim + param] */
    int n_params;
    char** param_names;
    double* current_state;
    double current_log_posterior;
    int total_samples;             /* After thinning */
    double* acceptance_rates;      /* Per-parameter or per-block */

    /* Proposal tuning */
    UQMatrix* proposal_covariance;
    double proposal_scale;
    int adaptation_interval;
    double target_acceptance;

    /* History */
    double* log_posterior_history;
    int n_stored;
    int store_capacity;
} UQMCMCState;

typedef enum {
    UQ_SAMPLER_METROPOLIS = 0,
    UQ_SAMPLER_METROPOLIS_HASTINGS = 1,
    UQ_SAMPLER_GIBBS = 2,
    UQ_SAMPLER_HAMILTONIAN = 3,
    UQ_SAMPLER_NUTS = 4,
    UQ_SAMPLER_SLICE = 5,
    UQ_SAMPLER_AM = 6,             /* Adaptive Metropolis */
    UQ_SAMPLER_DRAM = 7            /* Delayed Rejection AM */
} UQSamplerType;

/* --- Bayesian Calibration --- */

typedef struct {
    char* model_name;
    int n_calibration_params;
    char** param_names;
    double* nominal_values;
    double* lower_bounds;
    double* upper_bounds;

    UQDistribution** priors;
    UQPosterior* posterior_result;
    UQMCMCState* mcmc_result;

    /* Emulator/surrogate */
    double (*forward_model)(double* params, double* inputs, int n_inputs);
    double* training_inputs;
    int n_train;

    /* Diagnostics */
    double bias_correction;
    double* discrepancy_variance;
    UQConvergenceDiagnostic convergence;

    bool calibrated;
} UQBayesianCalibration;

/* --- Bayesian Model Averaging (BMA) --- */

typedef struct {
    int n_models;
    char** model_names;
    double* prior_probs;
    double* posterior_probs;
    double* log_marginal_likelihoods;
    double* bayes_factors;          /* vs best model */
    UQPosterior** posteriors;
    void** model_data;
} UQBMA;

/* --- API: Prior Distributions --- */

UQPriorDistribution* uq_prior_create(UQDistribution* dist, bool conjugate);
void uq_prior_free(UQPriorDistribution* prior);
UQMixturePrior* uq_mixture_prior_create(int n_components);
void uq_mixture_prior_free(UQMixturePrior* mix);
void uq_prior_jeffreys(UQDistribution** out, UQLikelihoodType likelihood);
void uq_prior_reference(UQDistribution** out, UQLikelihoodType likelihood);
double uq_prior_log_pdf(UQPriorDistribution* prior, double* params, int dim);

/* --- API: Likelihood Functions --- */

UQLikelihood* uq_likelihood_create(UQLikelihoodType type, int n_obs);
void uq_likelihood_free(UQLikelihood* lh);
double uq_likelihood_eval(UQLikelihood* lh, double* params,
                          double* predictions, int n);
double uq_likelihood_log(UQLikelihood* lh, double* params,
                         double* predictions, int n);
void uq_likelihood_set_scale(UQLikelihood* lh, double sigma);
void uq_likelihood_gradient(UQLikelihood* lh, double* params,
                            double* predictions, int n, double* grad);

/* --- API: Posterior Computation --- */

UQPosterior* uq_posterior_create(int n_params, char** param_names);
void uq_posterior_free(UQPosterior* post);
double uq_log_posterior(double* params, int dim, UQPriorDistribution* prior,
                        UQLikelihood* lh, double* preds);
void uq_posterior_summary(UQPosterior* post, UQParameterEnsemble* ens);
double uq_posterior_quantile(UQPosterior* post, int param_idx, double p);
double uq_posterior_predictive(double* x_new, UQPosterior* post,
    double (*model)(double*, double*), int n_posterior_samples);
double uq_bayes_factor(UQPosterior* m1, UQPosterior* m0);
double uq_compute_dic(UQPosterior* post);
double uq_compute_waic(UQPosterior* post, double* log_lik_samples,
                       int n_samples, int n_obs);
double uq_loo_cv_psis(UQPosterior* post, double* log_lik_matrix,
                      int n_samples, int n_obs);

/* --- API: MCMC Sampling --- */

UQMCMCState* uq_mcmc_create(int n_chains, int n_iterations, int n_params,
                            char** param_names);
void uq_mcmc_free(UQMCMCState* mcmc);
void uq_mcmc_set_proposal(UQMCMCState* mcmc, UQMatrix* cov, double scale);
void uq_mcmc_set_initial(UQMCMCState* mcmc, double* initial, int chain_idx);

/* Metropolis-Hastings sampler */
void uq_mh_sample(UQMCMCState* mcmc, UQPriorDistribution* prior,
                  UQLikelihood* lh, double* predictions, int n_chains);
/* Adaptive Metropolis (Haario et al., 2001) */
void uq_am_sample(UQMCMCState* mcmc, UQPriorDistribution* prior,
                  UQLikelihood* lh, double* predictions);
/* Gibbs sampler for conditionally conjugate models */
void uq_gibbs_sample(UQMCMCState* mcmc, void** conditional_samplers,
                     int n_params);
/* Hamiltonian Monte Carlo with leapfrog integrator */
void uq_hmc_sample(UQMCMCState* mcmc, UQPriorDistribution* prior,
                   UQLikelihood* lh, double* predictions,
                   int n_leapfrog, double step_size);
/* Slice sampler (Neal, 2003) */
void uq_slice_sample(UQMCMCState* mcmc, UQPriorDistribution* prior,
                     UQLikelihood* lh, double* predictions, double w);

void uq_mcmc_diagnostics(UQMCMCState* mcmc, UQConvergenceDiagnostic* diag);
double uq_mcmc_acceptance_rate(UQMCMCState* mcmc, int chain_idx);
void uq_mcmc_posterior_summary(UQMCMCState* mcmc, UQPosterior* post);
void uq_mcmc_trace(double* out, UQMCMCState* mcmc, int chain_idx,
                   int param_idx);

/* --- API: Bayesian Calibration --- */

UQBayesianCalibration* uq_calibration_create(const char* name, int n_params);
void uq_calibration_free(UQBayesianCalibration* cal);
void uq_calibration_set_prior(UQBayesianCalibration* cal, int idx,
                              UQDistribution* prior);
void uq_calibration_set_bounds(UQBayesianCalibration* cal, int idx,
                               double lb, double ub);
void uq_calibrate(UQBayesianCalibration* cal, UQDataset* data,
                  int n_mcmc_iter);
void uq_calibration_predict(UQBayesianCalibration* cal, double* inputs,
                            int n_inputs, double* mean, double* std);

/* --- API: Bayesian Model Averaging --- */

UQBMA* uq_bma_create(int n_models, char** names);
void uq_bma_free(UQBMA* bma);
void uq_bma_set_prior_weights(UQBMA* bma, double* weights);
void uq_bma_compute(UQBMA* bma);
double uq_bma_predict(UQBMA* bma, double* x, int n_models_active);
void uq_bma_variable_importance(UQBMA* bma, int n_vars, double* importance);

#endif /* UQ_BAYESIAN_H */
