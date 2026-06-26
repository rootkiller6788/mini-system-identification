#include "../include/rls_core.h"
#include "../include/rls_solvers.h"
#include "../include/rls_models.h"
#include "../include/rls_validation.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* ============================================================================
 * Example 2: ARX Model Identification with Lambda Selection Methods
 *
 * Demonstrates:
 *  - ARX model regressor construction
 *  - Multiple lambda selection criteria (GCV, L-curve, K-fold CV, AICc, Empirical Bayes)
 *  - Comparison of regularization methods (Ridge, LASSO, Elastic Net)
 *  - Model validation (fit, residual analysis)
 *
 * True system: y(t) - 1.5y(t-1) + 0.7y(t-2) = 1.0*u(t-1) + 0.5*u(t-2) + e(t)
 *              a1=-1.5, a2=0.7, b1=1.0, b2=0.5
 *
 * Knowledge: L2 (bias-variance, overfitting), L5 (lambda selection methods),
 *            L6 (ARX identification canonical problem), L7 (process analogy)
 * ============================================================================ */

int main(void) {
    printf("=== Example 2: ARX Model Identification ===\n\n");

    /* Generate data from ARX(2,2,1) system */
    const int N = 400;
    RLSData *data = rls_data_alloc(N);
    data->ts = 1.0;
    /* PRBS-like input (alternating steps) */
    for (int t = 0; t < N; t++)
        data->u[t] = ((t / 15) % 2 == 0) ? 1.0 : -1.0;

    /* Simulate ARX system */
    double a1_true = -1.5, a2_true = 0.7;
    double b1_true = 1.0, b2_true = 0.5;
    for (int t = 0; t < N; t++) {
        if (t < 2) { data->y[t] = 0.0; continue; }
        /* y(t) = -a1*y(t-1) - a2*y(t-2) + b1*u(t-1) + b2*u(t-2) + noise */
        data->y[t] = 1.5*data->y[t-1] - 0.7*data->y[t-2]
                    + 1.0*data->u[t-1] + 0.5*data->u[t-2];
    }
    /* Add noise (SNR ~ 30dB) */
    double sig_rms = 0.0;
    for (int t = 0; t < N; t++) sig_rms += data->y[t]*data->y[t];
    sig_rms = sqrt(sig_rms/N);
    for (int t = 0; t < N; t++)
        data->y[t] += sig_rms * 0.0316 * ((double)rand()/RAND_MAX-0.5)*2.0*sqrt(3.0);

    /* Build ARX model (estimate correct order) */
    RLSModelOrder order = {RLS_MODEL_ARX, 2, 2, 0, 0, 0, 1, 0, 1.0};
    int np = 4;
    int max_delay = 2;
    int n_eff = N - max_delay;
    RLSMatrix *Phi = rls_matrix_alloc(n_eff, np);
    RLSVector *y_vec = rls_vector_alloc(n_eff);
    rls_build_arx_regressor(Phi, y_vec, data, &order);
    RLSOptions opt = rls_options_default();
    opt.verbose = false;

    /* --- Compare lambda selection methods --- */
    printf("--- Lambda Selection Methods Comparison ---\n");
    const char *method_names[] = {"GCV", "K-fold CV", "AICc", "Empirical Bayes", "L-curve"};
    RLSLambdaMethod methods[] = {RLS_LAMBDA_GCV, RLS_LAMBDA_KFOLD_CV,
                                  RLS_LAMBDA_AICC, RLS_LAMBDA_ML, RLS_LAMBDA_LCURVE};
    double best_lambdas[5];
    for (int m = 0; m < 5; m++) {
        RLSLambdaSelection sel = rls_lambda_selection_default(methods[m], 0.1);
        sel.lambda_min = 1e-4; sel.lambda_max = 50.0; sel.n_lambda = 30;
        sel.k_folds = 5;
        rls_select_lambda(Phi, y_vec, &sel, RLS_REG_RIDGE, 0.0, &opt);
        best_lambdas[m] = sel.lambda_opt;
        printf("  %-16s: lambda_opt = %.6e\n", method_names[m], sel.lambda_opt);
        free(sel.lambda_grid); free(sel.cv_scores);
    }

    /* --- Ridge with best lambda (median of selections) --- */
    double lam_ridge = best_lambdas[2]; /* use AICc */
    RLSEstimate *est_ridge = rls_solve_ridge(Phi, y_vec, lam_ridge, &opt);
    if (est_ridge) {
        printf("\n--- Ridge Estimate (lambda=%.4e) ---\n", lam_ridge);
        printf("  theta = [%.4f, %.4f, %.4f, %.4f]\n",
               est_ridge->theta[0], est_ridge->theta[1],
               est_ridge->theta[2], est_ridge->theta[3]);
        printf("  True  = [%.4f, %.4f, %.4f, %.4f]\n", -a1_true, -a2_true, b1_true, b2_true);
        printf("  MSE=%.6f, R2=%.4f, AICc=%.2f\n",
               est_ridge->mse, est_ridge->r2, est_ridge->aic);
    }

    /* --- LASSO --- */
    RLSSolverConfig cd_cfg = rls_solver_config_default(RLS_SOLVER_CD);
    cd_cfg.cd_max_iter = 10000;
    RLSEstimate *est_lasso = rls_solve_lasso_cd(Phi, y_vec, 0.02, &opt, &cd_cfg);
    if (est_lasso) {
        printf("\n--- LASSO Estimate (lambda=0.02) ---\n");
        printf("  theta = [%.4f, %.4f, %.4f, %.4f]\n",
               est_lasso->theta[0], est_lasso->theta[1],
               est_lasso->theta[2], est_lasso->theta[3]);
        printf("  MSE=%.6f\n", est_lasso->mse);
    }

    /* --- Elastic Net --- */
    RLSEstimate *est_enet = rls_solve_elasticnet_cd(Phi, y_vec, 0.5, 0.05, &opt, &cd_cfg);
    if (est_enet) {
        printf("\n--- Elastic Net Estimate (alpha=0.5, lambda=0.05) ---\n");
        printf("  theta = [%.4f, %.4f, %.4f, %.4f]\n",
               est_enet->theta[0], est_enet->theta[1],
               est_enet->theta[2], est_enet->theta[3]);
        printf("  MSE=%.6f\n", est_enet->mse);
    }

    /* --- Model Validation --- */
    if (est_ridge) {
        printf("\n--- Model Validation ---\n");
        RLSVector *y_sim = rls_vector_alloc(N);
        RLSVector theta_v; theta_v.dim = np; theta_v.capacity = 0;
        theta_v.data = est_ridge->theta;
        rls_simulate_arx(y_sim, data, &theta_v, &order);
        RLSVector yv; yv.dim = N; yv.capacity = 0; yv.data = data->y;
        double fit = rls_fit_percent(&yv, y_sim);
        printf("  Fit: %.2f%%\n", fit);

        /* Residual analysis */
        RLSVector *res = rls_vector_alloc(n_eff);
        rls_compute_residual(res, y_vec, Phi, &theta_v);
        double p_white = rls_whiteness_test(res, 10);
        double p_indep = rls_independence_test(res, y_vec, 5);
        printf("  Whiteness test p-value: %.4f (>0.05 = white)\n", p_white);
        printf("  Independence test p-value: %.4f (>0.05 = independent)\n", p_indep);
        rls_vector_free(y_sim); rls_vector_free(res);
    }

    /* Cleanup */
    rls_data_free(data);
    rls_matrix_free(Phi); rls_vector_free(y_vec);
    if (est_ridge) rls_estimate_free(est_ridge);
    if (est_lasso) rls_estimate_free(est_lasso);
    if (est_enet) rls_estimate_free(est_enet);

    printf("\n=== Example 2 complete ===\n");
    return 0;
}
