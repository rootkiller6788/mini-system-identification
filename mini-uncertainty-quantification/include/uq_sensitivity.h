#ifndef UQ_SENSITIVITY_H
#define UQ_SENSITIVITY_H

#include "uq_core.h"
#include <stdbool.h>

/* ============================================================================
 * Sensitivity Analysis for Uncertainty Quantification
 *
 * Quantifies how uncertainty in model inputs contributes to uncertainty
 * in model outputs. Supports local (derivative-based) and global
 * (variance-based) methods.
 *
 * Key References:
 *   - Saltelli, A. et al. (2008). "Global Sensitivity Analysis: The Primer."
 *     Wiley.
 *   - Sobol', I.M. (1993). "Sensitivity estimates for nonlinear mathematical
 *     models." Mathematical Modelling and Computational Experiments, 1(4),
 *     407-414.
 *   - Morris, M.D. (1991). "Factorial sampling plans for preliminary
 *     computational experiments." Technometrics, 33(2), 161-174.
 *   - Borgonovo, E. (2007). "A new uncertainty importance measure."
 *     Reliability Engineering & System Safety, 92(6), 771-784.
 * ============================================================================ */

/* --- Sensitivity Indices --- */

typedef struct {
    char* variable_name;
    int variable_index;

    /* Sobol' indices (variance-based) */
    double sobol_main;            /* First-order (main effect) S_i */
    double sobol_total;           /* Total effect S_{Ti} */
    double sobol_second_order_sum; /* Sum of 2nd-order interactions for var i */

    /* Moment-independent delta index (Borgonovo, 2007) */
    double delta_index;

    /* Morris elementary effects */
    double morris_mu;             /* Mean of elementary effects */
    double morris_mu_star;        /* Mean of absolute elementary effects */
    double morris_sigma;          /* Standard deviation of effects */
    double morris_mu_star_conf;   /* Confidence bound for mu* */

    /* Shapley value */
    double shapley_value;         /* Fair attribution */

    /* Derivative-based */
    double derivative_sensitivity; /* Average ∂y/∂x_i */
    double sigma_normalized_derivative; /* σ_i * ∂y/∂x_i / σ_y */

    /* FAST (Fourier Amplitude Sensitivity Test) */
    double fast_main;             /* Main effect from Fourier decomposition */
    double fast_interaction;      /* Interaction effect */

    /* Rank information */
    int rank_by_total_effect;
    int rank_by_main_effect;

    double confidence_interval_half; /* Bootstrap CI half-width */
} UQSensitivityIndex;

/* --- Sensitivity Analysis State --- */

typedef struct {
    int n_variables;
    UQSensitivityIndex* indices;
    double* variable_names;
    int n_samples;               /* Base sample size N */

    /* Matrices for Saltelli estimator */
    double* A;                   /* [N × d] sample matrix A */
    double* B;                   /* [N × d] sample matrix B */
    double* A_B_i;               /* [N × d × d] A with column i from B */

    /* Model evaluations */
    double* f_A;                 /* f(A) for all N rows */
    double* f_B;                 /* f(B) for all N rows */
    double** f_A_B_i;            /* [d][N] — f(A_B^(i)) */

    /* Morris trajectories */
    int n_trajectories;          /* Number of Morris paths */
    double grid_levels;           /* Number of grid levels (p) */
    double delta_morris;          /* Grid step size */

    UQSensitivityMethod method;
    bool computed;
    double computation_cost;      /* Total model evaluations */

    /* Convergence diagnostics */
    bool* sobol_converged;        /* Per-index convergence */
    double* sobol_error_estimates;
} UQSensitivityAnalysis;

/* --- PAWN Density-Based --- */

typedef struct {
    int n_variables;
    int n_points;                  /* Sample points per CDF */
    int n_conditioning;           /* n_u — unconditional samples */
    double* kolmogorov_smirnov;   /* KS statistic per variable */
    double* pawn_index;           /* PAWN index per variable */
    double* cdf_unconditional;    /* [n_variables × n_points] */
    double** cdf_conditional;     /* [n_variables][n_cond × n_points] */
    bool computed;
} UQPAWNAnalysis;

/* --- Regional Sensitivity Analysis --- */

typedef struct {
    int n_variables;
    double threshold;              /* Output threshold for behavioral/non-behavioral */
    int n_behavioral;
    int n_non_behavioral;
    double* kolmogorov_smirnov;   /* KS statistic: sep between behavioral/non */
    double* area_between_cdfs;    /* CDF distance metric */
    double* t_statistics;         /* Two-sample t-test on means */
    bool computed;
} UQRegionalSA;

/* --- API: Sobol' Indices --- */

UQSensitivityAnalysis* uq_sa_create(int n_variables, char** var_names);
void uq_sa_free(UQSensitivityAnalysis* sa);
void uq_sa_set_sample_size(UQSensitivityAnalysis* sa, int n_base);
void uq_sa_generate_matrices(UQSensitivityAnalysis* sa);

/* Saltelli (2010) estimator for first-order and total Sobol' indices */
void uq_sobol_saltelli(UQSensitivityAnalysis* sa,
    double (*model)(double*, void*), void* model_data);
/* Jansen (1999) estimator (often more stable for total indices) */
void uq_sobol_jansen(UQSensitivityAnalysis* sa,
    double (*model)(double*, void*), void* model_data);
/* Janon/Monod estimator with better convergence for small N */
void uq_sobol_janon(UQSensitivityAnalysis* sa,
    double (*model)(double*, void*), void* model_data);
/* Bootstrap confidence intervals for Sobol' indices */
void uq_sobol_bootstrap_ci(UQSensitivityAnalysis* sa, int n_bootstrap);

double uq_sobol_main_effect(UQSensitivityAnalysis* sa, int var_idx);
double uq_sobol_total_effect(UQSensitivityAnalysis* sa, int var_idx);

/* --- API: Morris Method --- */

void uq_morris_setup(UQSensitivityAnalysis* sa, int n_trajectories,
                     int grid_levels);
void uq_morris_evaluate(UQSensitivityAnalysis* sa,
    double (*model)(double*, void*), void* model_data,
    double* lower_bounds, double* upper_bounds);
void uq_morris_mu_sigma(UQSensitivityAnalysis* sa,
                        double** mu_star, double** sigma);

/* --- API: FAST (Fourier Amplitude Sensitivity Test) --- */

void uq_fast_evaluate(UQSensitivityAnalysis* sa,
    double (*model)(double*, void*), void* model_data,
    int n_samples_fast, int interference_order);
void uq_fast_indices(UQSensitivityAnalysis* sa, double* omega_set);

/* --- API: Moment-Independent Delta (Borgonovo, 2007) --- */

void uq_delta_indices(UQSensitivityAnalysis* sa,
    double (*model)(double*, void*), void* model_data,
    int n_samples_delta);
/* PAWN: density-based sensitivity (Pianosi & Wagener, 2015) */
UQPAWNAnalysis* uq_pawn_create(int n_variables, int n_points, int n_cond);
void uq_pawn_free(UQPAWNAnalysis* pawn);
void uq_pawn_compute(UQPAWNAnalysis* pawn,
    double (*model)(double*, void*), void* model_data,
    int n_conditioning, int n_unconditional, double* bounds_lower,
    double* bounds_upper);

/* --- API: Shapley Value Attribution --- */

double uq_shapley_compute(int n_vars, int var_idx,
    double (*value_function)(int*, int, void*), void* vf_data);
void uq_shapley_all(int n_vars,
    double (*value_function)(int*, int, void*), void* vf_data,
    double* shapley_values);

/* --- API: Regional Sensitivity --- */

UQRegionalSA* uq_rsa_create(int n_variables, double threshold);
void uq_rsa_free(UQRegionalSA* rsa);
void uq_rsa_compute(UQRegionalSA* rsa,
    double (*model)(double*, void*), void* model_data,
    int n_samples, double* lower_bounds, double* upper_bounds);

/* --- API: Derivative-Based (local) --- */

void uq_local_sensitivity(double (*model)(double*, void*), void* model_data,
    double* nominal_params, int n_params, double* sensitivity,
    double perturbation);
void uq_local_sensitivity_normalized(double (*model)(double*, void*),
    void* model_data, double* nominal_params, double* param_std,
    int n_params, double* sigma_normalized);

/* --- API: Convergence Analysis for Sensitivity --- */

bool uq_sa_check_convergence(UQSensitivityAnalysis* sa, double tolerance);
void uq_sa_print_summary(UQSensitivityAnalysis* sa);

#endif /* UQ_SENSITIVITY_H */
