#include "../include/rls_core.h"
#include "../include/rls_solvers.h"
#include "../include/rls_models.h"
#include "../include/rls_validation.h"
#include "../include/rls_kernel.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* ============================================================================
 * Example 1: FIR System Identification with Ridge, LASSO, and Kernel methods
 *
 * Demonstrates:
 *  - Building FIR model from input-output data
 *  - Ridge regression with GCV lambda selection
 *  - LASSO for sparse impulse response recovery
 *  - Kernel-based (Stable Spline) regularization
 *  - Model validation and comparison
 *
 * Knowledge: L1 (FIR definition), L5 (ridge/lasso/kernel solvers),
 *            L6 (system ID canonical problem), L8 (kernel methods)
 * ============================================================================ */

int main(void) {
    printf("=== Example 1: FIR System Identification ===\n\n");

    /* Generate a 15-tap FIR system */
    const int nb_true = 15;
    double theta_true[15] = {0.0, 0.8, 0.5, 0.3, 0.2, 0.1, 0.05, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    const int N = 500;
    RLSData *data = rls_data_alloc(N);
    data->ts = 1.0;

    /* Generate PRBS excitation */
    rls_data_generate_prbs(data, 7, 1.0, 0.0);
    /* Simulate FIR system */
    for (int t = nb_true; t < N; t++) {
        double yh = 0.0;
        for (int j = 0; j < nb_true; j++)
            yh += theta_true[j] * data->u[t - 1 - j];
        data->y[t] = yh;
    }
    /* Add measurement noise (SNR ~ 25dB) */
    double sig_var = 0.0;
    for (int t = 0; t < N; t++) sig_var += data->y[t]*data->y[t];
    double noise_std = sqrt(sig_var/N) * 0.056;
    for (int t = 0; t < N; t++)
        data->y[t] += noise_std * ((double)rand()/RAND_MAX - 0.5)*2.0*sqrt(3.0);

    /* Build FIR regressor (estimate nb=20 for over-parameterization) */
    int nb_est = 20;
    int n_eff = data->N - nb_est;
    RLSMatrix *Phi = rls_matrix_alloc(n_eff, nb_est);
    RLSVector *y_vec = rls_vector_alloc(n_eff);
    rls_build_fir_regressor(Phi, y_vec, data, nb_est);
    RLSOptions opt = rls_options_default();
    opt.verbose = false;

    /* --- Method 1: Ridge with GCV --- */
    printf("--- Method 1: Ridge Regression with GCV ---\n");
    RLSLambdaSelection sel_gcv = rls_lambda_selection_default(RLS_LAMBDA_GCV, 0.1);
    sel_gcv.lambda_min = 1e-4; sel_gcv.lambda_max = 10.0; sel_gcv.n_lambda = 30;
    rls_select_lambda(Phi, y_vec, &sel_gcv, RLS_REG_RIDGE, 0.0, &opt);
    RLSEstimate *est_ridge = rls_solve_ridge(Phi, y_vec, sel_gcv.lambda_opt, &opt);
    if (est_ridge) {
        printf("  lambda_gcv = %.6e\n", sel_gcv.lambda_opt);
        printf("  MSE = %.6f, R2 = %.4f, df = %.2f\n",
               est_ridge->mse, est_ridge->r2, est_ridge->effective_df);
        printf("  First 5 taps: ");
        for (int j=0; j<5; j++) printf("%.4f ", est_ridge->theta[j]);
        printf("\n");
    }

    /* --- Method 2: LASSO for sparse recovery --- */
    printf("\n--- Method 2: LASSO (Coordinate Descent) ---\n");
    RLSSolverConfig cd_cfg = rls_solver_config_default(RLS_SOLVER_CD);
    cd_cfg.cd_max_iter = 10000;
    cd_cfg.cd_tol = 1e-5;
    double lambda_lasso = 0.05;
    RLSEstimate *est_lasso = rls_solve_lasso_cd(Phi, y_vec, lambda_lasso, &opt, &cd_cfg);
    if (est_lasso) {
        printf("  lambda = %.4f\n", lambda_lasso);
        printf("  MSE = %.6f, Non-zeros = ", est_lasso->mse);
        int nnz = 0;
        for (int j = 0; j < nb_est; j++)
            if (fabs(est_lasso->theta[j]) > 1e-6) nnz++;
        printf("%d\n", nnz);
        printf("  Taps: ");
        for (int j=0; j<10; j++) printf("%.4f ", est_lasso->theta[j]);
        printf("\n");
    }

    /* --- Method 3: Kernel-based (Stable Spline) --- */
    printf("\n--- Method 3: Kernel-Based (Stable Spline) ---\n");
    RLSKernel kernel = rls_kernel_default_ss(nb_est, 0.88);
    RLSEstimate *est_kernel = rls_kernel_fir_identify(data, &kernel, 0.5, &opt);
    if (est_kernel) {
        printf("  beta = %.4f\n", kernel.beta);
        printf("  MSE = %.6f, R2 = %.4f\n", est_kernel->mse, est_kernel->r2);
        printf("  Taps: ");
        for (int j=0; j<10; j++) printf("%.4f ", est_kernel->theta[j]);
        printf("\n");
    }

    /* --- Method 4: OLS (no regularization) for comparison --- */
    printf("\n--- Method 4: OLS (no regularization) ---\n");
    RLSEstimate *est_ols = rls_solve_ridge(Phi, y_vec, 0.0, &opt);
    if (est_ols) {
        printf("  MSE = %.6f, R2 = %.4f, Cond = %.2e\n",
               est_ols->mse, est_ols->r2, est_ols->cond_number);
    }

    /* --- Comparison --- */
    printf("\n--- Comparison ---\n");
    printf("  True impulse response (first 8): ");
    for (int j=0; j<8; j++) printf("%.4f ", theta_true[j]);
    printf("\n  Ridge estimate:                    ");
    if (est_ridge) for (int j=0; j<8; j++) printf("%.4f ", est_ridge->theta[j]);
    printf("\n  LASSO estimate:                    ");
    if (est_lasso) for (int j=0; j<8; j++) printf("%.4f ", est_lasso->theta[j]);
    printf("\n  Kernel estimate:                   ");
    if (est_kernel) for (int j=0; j<8; j++) printf("%.4f ", est_kernel->theta[j]);
    printf("\n  OLS estimate:                      ");
    if (est_ols) for (int j=0; j<8; j++) printf("%.4f ", est_ols->theta[j]);
    printf("\n");

    /* Simulate and compute fit */
    if (est_ridge) {
        RLSVector *y_sim = rls_vector_alloc(N);
        RLSVector theta_v; theta_v.dim = nb_est; theta_v.capacity = 0;
        theta_v.data = est_ridge->theta;
        rls_simulate_fir(y_sim, data, &theta_v, nb_est);
        RLSVector yv; yv.dim = N; yv.capacity = 0; yv.data = data->y;
        double fit = rls_fit_percent(&yv, y_sim);
        printf("\n  Ridge model fit on full data: %.2f%%\n", fit);
        rls_vector_free(y_sim);
    }

    /* Cleanup */
    rls_data_free(data);
    rls_matrix_free(Phi); rls_vector_free(y_vec);
    if (est_ridge) rls_estimate_free(est_ridge);
    if (est_lasso) rls_estimate_free(est_lasso);
    if (est_kernel) rls_estimate_free(est_kernel);
    if (est_ols) rls_estimate_free(est_ols);
    free(sel_gcv.lambda_grid); free(sel_gcv.cv_scores);

    printf("\n=== Example 1 complete ===\n");
    return 0;
}
