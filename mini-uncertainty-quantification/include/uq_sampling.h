#ifndef UQ_SAMPLING_H
#define UQ_SAMPLING_H

#include "uq_core.h"
#include <stdbool.h>

/* ============================================================================
 * Monte Carlo and Resampling Methods for UQ
 *
 * Key References:
 *   - Robert, C.P. & Casella, G. (2004). "Monte Carlo Statistical Methods."
 *     Springer.
 *   - Efron, B. & Tibshirani, R.J. (1993). "An Introduction to the Bootstrap."
 *     Chapman & Hall.
 *   - McKay, M.D., Beckman, R.J., & Conover, W.J. (1979). "A comparison of
 *     three methods for selecting values of input variables in the analysis of
 *     output from a computer code." Technometrics, 21(2), 239-245.
 *   - Owen, A.B. (2013). "Monte Carlo theory, methods and examples."
 * ============================================================================ */

/* --- Sampling Strategies --- */

typedef enum {
    UQ_MC_SIMPLE = 0,            /* Simple Monte Carlo */
    UQ_MC_IMPORTANCE = 1,        /* Importance sampling */
    UQ_MC_REJECTION = 2,         /* Rejection sampling */
    UQ_MC_STRATIFIED = 3,        /* Stratified sampling */
    UQ_MC_SYSTEMATIC = 4,        /* Systematic sampling */
    UQ_MC_ANTITHETIC = 5,        /* Antithetic variates */
    UQ_MC_CONTROL_VARIATE = 6,   /* Control variate method */
    UQ_MC_LATIN_HYPERCUBE = 7,   /* Latin Hypercube Sampling (LHS) */
    UQ_MC_SOBOL = 8,             /* Sobol' quasi-Monte Carlo */
    UQ_MC_HALTON = 9,            /* Halton sequence */
    UQ_MC_HAMMERSLEY = 10,       /* Hammersley sequence */
    UQ_MC_FAURE = 11,            /* Faure sequence */
    UQ_MC_ORTHOGONAL = 12        /* Orthogonal array sampling */
} UQMCStrategy;

/* --- Bootstrap Variants --- */

typedef enum {
    UQ_BOOTSTRAP_STANDARD = 0,     /* Efron's nonparametric bootstrap */
    UQ_BOOTSTRAP_PARAMETRIC = 1,   /* Parametric bootstrap */
    UQ_BOOTSTRAP_RESIDUAL = 2,     /* Residual resampling */
    UQ_BOOTSTRAP_WILD = 3,         /* Wild bootstrap (heteroskedastic) */
    UQ_BOOTSTRAP_BLOCK = 4,        /* Block bootstrap (time series) */
    UQ_BOOTSTRAP_STATIONARY = 5,   /* Stationary bootstrap */
    UQ_BOOTSTRAP_MOVING_BLOCK = 6, /* Moving block bootstrap */
    UQ_BOOTSTRAP_PAIRS = 7,        /* Pairs bootstrap (regression) */
    UQ_BOOTSTRAP_CLUSTER = 8,      /* Cluster bootstrap */
    UQ_BOOTSTRAP_BAYESIAN = 9      /* Bayesian bootstrap (Rubin, 1981) */
} UQBootstrapMethod;

/* --- Sampler State --- */

typedef struct {
    UQMCStrategy strategy;
    int n_samples;
    int n_dimensions;
    double** samples;              /* samples[sample_idx][dim] */
    UQDistribution** marginals;    /* Per-dimension distribution */
    UQMatrix* correlation;         /* Correlation structure for Iman-Conover */
    bool is_correlated;

    /* Quasi-MC state */
    int* prime_bases;              /* For Halton/Hammersley */
    unsigned long long* sobol_state;
    int sobol_skip;
    double* sobol_direction_numbers;

    /* LHS state */
    int* lhs_permutation;
    int lhs_n_intervals;

    /* Performance */
    double integration_error_estimate;
    double effective_sample_size;
    double relative_efficiency;    /* vs simple MC */
} UQSampler;

/* --- Bootstrap State --- */

typedef struct {
    UQBootstrapMethod method;
    int n_original;
    int n_bootstrap;
    double* original_data;
    int n_dim;
    double** bootstrap_replicates; /* [bootstrap][n_original] */

    /* Block bootstrap parameters */
    int block_length;

    /* Results */
    double* statistic_replicates;  /* [n_bootstrap] */
    double original_statistic;
    double bootstrap_mean;
    double bootstrap_se;
    double bootstrap_bias;
    double* ci_percentile;
    double* ci_basic;              /* Basic bootstrap CI */
    double* ci_bca;                /* BCa interval */
    int n_ci;
} UQBootstrap;

/* --- Gaussian Process Emulator --- */

typedef enum {
    UQ_GP_KERNEL_SQUARED_EXPONENTIAL = 0,
    UQ_GP_KERNEL_MATERN32 = 1,
    UQ_GP_KERNEL_MATERN52 = 2,
    UQ_GP_KERNEL_EXPONENTIAL = 3,
    UQ_GP_KERNEL_RATIONAL_QUADRATIC = 4,
    UQ_GP_KERNEL_PERIODIC = 5,
    UQ_GP_KERNEL_LINEAR = 6,
    UQ_GP_KERNEL_NEURAL_NETWORK = 7
} UQGPKernelType;

typedef struct {
    UQGPKernelType kernel_type;
    int n_inputs;
    int n_train;
    double* X_train;              /* [n_train * n_inputs] */
    double* y_train;              /* [n_train] */
    double* kernel_params;        /* length_scale[n_inputs], sigma_f, sigma_n */

    double* K;                    /* Kernel matrix [n_train * n_train] */
    double* K_inv;                /* Inverse + Cholesky factor */
    double* alpha;                /* K^{-1} * y */

    double log_marginal_likelihood;
    bool trained;
} UQGaussianProcess;

/* --- API: Core Sampling --- */

UQSampler* uq_sampler_create(UQMCStrategy strategy, int n_samples, int n_dims);
void uq_sampler_free(UQSampler* sampler);
void uq_sampler_set_marginals(UQSampler* sampler, UQDistribution** dists);
void uq_sampler_set_correlation(UQSampler* sampler, UQMatrix* corr);
void uq_sampler_generate(UQSampler* sampler);
double* uq_sampler_get_sample(UQSampler* sampler, int idx);

/* Quasi-Monte Carlo sequences */
void uq_sobol_sequence(int n, int dim, unsigned long long skip, double** out);
void uq_halton_sequence(int n, int dim, double** out);
void uq_hammersley_sequence(int n, int dim, double** out);

/* Latin Hypercube Sampling */
void uq_lhs_generate(int n, int dim, double** out, bool randomize);
void uq_lhs_imam_conover(double** lhs, int n, int dim, UQMatrix* target_corr);
void uq_lhs_midpoint(int n, int dim, double** out);
void uq_orthogonal_array_lhs(int n_levels, int dim, double** out);

/* Importance sampling */
double uq_importance_sampling(double (*f)(double*, void*), void* f_data,
    UQDistribution* proposal, double (*target_log_pdf)(double*, void*),
    void* target_data, int n_samples, int dim, double* mc_error);

/* Rejection sampling */
int uq_rejection_sampling(double (*target_pdf)(double*, void*), void* target_data,
    UQDistribution* proposal, double M, int n_desired, int dim, double** out);

/* Stratified sampling */
void uq_stratified_sample(UQDistribution** strata, double* strata_weights,
    int n_strata, int* n_per_stratum, int dim, double** out);

/* Antithetic variates */
void uq_antithetic_sample(int n_pairs, int dim, double** out);

/* Control variates */
double uq_control_variate_estimate(double* f_vals, double* c_vals,
    double true_mean_c, int n);

/* --- API: Bootstrap Methods --- */

UQBootstrap* uq_bootstrap_create(UQBootstrapMethod method, int n_original,
                                 int n_bootstrap, int dim);
void uq_bootstrap_free(UQBootstrap* bs);
void uq_bootstrap_set_data(UQBootstrap* bs, double* data);
void uq_bootstrap_set_block_length(UQBootstrap* bs, int block_len);
void uq_bootstrap_generate_replicates(UQBootstrap* bs);
void uq_bootstrap_compute_statistics(UQBootstrap* bs,
    double (*statistic)(double*, int, void*), void* stat_data);
void uq_bootstrap_ci_percentile(UQBootstrap* bs, double confidence);
void uq_bootstrap_ci_basic(UQBootstrap* bs, double confidence);
void uq_bootstrap_ci_bca(UQBootstrap* bs, double confidence,
                         double (*jackknife_stat)(double*, int, int, void*),
                         void* jack_data);
void uq_bootstrap_ci_studentized(UQBootstrap* bs, double confidence,
    double (*stat)(double*, int, void*),
    double (*se_func)(double*, int, void*), void* aux_data);

/* Bayesian bootstrap (Rubin, 1981 — Dirichlet weights) */
void uq_bayesian_bootstrap(int n, int n_replicates, double** weights_out);

/* --- API: Gaussian Process Emulator --- */

UQGaussianProcess* uq_gp_create(UQGPKernelType kernel, int n_inputs,
                                int n_train);
void uq_gp_free(UQGaussianProcess* gp);
void uq_gp_set_data(UQGaussianProcess* gp, double* X, double* y);
void uq_gp_set_params(UQGaussianProcess* gp, double* params);
void uq_gp_train(UQGaussianProcess* gp);
double uq_gp_predict(UQGaussianProcess* gp, double* x_new, double* pred_var);
void uq_gp_predict_batch(UQGaussianProcess* gp, double* X_new, int n_new,
                          double* pred_mean, double* pred_var);
void uq_gp_optimize_params(UQGaussianProcess* gp, int max_iter);

/* Kernel evaluation */
double uq_gp_kernel_se(double r, double length_scale);
double uq_gp_kernel_matern32(double r, double length_scale);
double uq_gp_kernel_matern52(double r, double length_scale);
double uq_gp_kernel_rq(double r, double length_scale, double alpha);

#endif /* UQ_SAMPLING_H */
