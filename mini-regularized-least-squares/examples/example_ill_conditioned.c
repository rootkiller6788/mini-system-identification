#include "../include/rls_core.h"
#include "../include/rls_solvers.h"
#include "../include/rls_models.h"
#include "../include/rls_validation.h"
#include "../include/rls_regularizers.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* ============================================================================
 * Example 3: Ill-Conditioned Regression and Regularization Benefit
 *
 * Demonstrates:
 *  - Ill-conditioned design matrix (near-collinear columns)
 *  - OLS failure: extreme variance, wrong signs
 *  - Ridge stabilization
 *  - LASSO variable selection in correlated setting
 *  - L-curve visualization (text-based)
 *  - SVD analysis of singular value shrinkage
 *
 * This example illustrates the fundamental bias-variance tradeoff (L2/L4).
 * ============================================================================ */

int main(void) {
    printf("=== Example 3: Ill-Conditioned Regression ===\n\n");

    /* Construct ill-conditioned design matrix: nearly collinear columns */
    const int n = 50, p = 10;
    RLSMatrix *Phi = rls_matrix_alloc(n, p);
    RLSVector *theta_true = rls_vector_alloc(p);
    /* True parameters: only first 3 non-zero */
    theta_true->data[0] = 2.0;
    theta_true->data[1] = -1.0;
    theta_true->data[2] = 1.5;
    /* Remaining are 0 */

    /* Generate correlated columns */
    for (int i = 0; i < n; i++) {
        double base = ((double)i - n/2.0) / (n/4.0);
        for (int j = 0; j < p; j++) {
            /* Each column is base + small perturbation */
            double val = base + 0.05 * j + 0.01 * sin((double)(i*j));
            rls_matrix_set(Phi, i, j, val);
        }
    }

    /* Generate y with noise */
    RLSVector *y = rls_vector_alloc(n);
    rls_matrix_vector_mul(y, Phi, theta_true);
    double noise_std = 0.1;
    for (int i = 0; i < n; i++)
        y->data[i] += noise_std * ((double)rand()/RAND_MAX - 0.5)*2.0*sqrt(3.0);

    /* Condition number */
    double cond = rls_cond_estimate(Phi, 100, 1e-6);
    printf("Design matrix condition number: %.2e\n\n", cond);

    RLSOptions opt = rls_options_default();
    opt.verbose = false;

    /* --- OLS (lambda=0) --- */
    printf("--- OLS (no regularization) ---\n");
    RLSEstimate *est_ols = rls_solve_ridge(Phi, y, 0.0, &opt);
    if (est_ols) {
        printf("  theta = ");
        for (int j=0; j<p; j++) printf("%.4f ", est_ols->theta[j]);
        printf("\n  true  = ");
        for (int j=0; j<p; j++) printf("%.4f ", theta_true->data[j]);
        printf("\n  MSE=%.6f, Cond=%.2e\n", est_ols->mse, est_ols->cond_number);
    }

    /* --- Ridge with various lambda --- */
    printf("\n--- Ridge Regression (various lambda) ---\n");
    double lambdas[] = {0.001, 0.01, 0.1, 1.0, 10.0};
    for (int li = 0; li < 5; li++) {
        RLSEstimate *est = rls_solve_ridge(Phi, y, lambdas[li], &opt);
        if (est) {
            double param_err = 0.0;
            for (int j=0; j<p; j++) {
                double d = est->theta[j] - theta_true->data[j];
                param_err += d*d;
            }
            param_err = sqrt(param_err);
            printf("  lambda=%-8.4f  MSE=%.6f  param_err=%.4f  df=%.2f\n",
                   lambdas[li], est->mse, param_err, est->effective_df);
            rls_estimate_free(est);
        }
    }

    /* --- GCV lambda selection --- */
    printf("\n--- GCV Lambda Selection ---\n");
    RLSLambdaSelection sel = rls_lambda_selection_default(RLS_LAMBDA_GCV, 0.1);
    sel.lambda_min = 1e-4; sel.lambda_max = 100.0; sel.n_lambda = 40;
    rls_select_lambda(Phi, y, &sel, RLS_REG_RIDGE, 0.0, &opt);
    printf("  GCV optimal lambda = %.6e\n", sel.lambda_opt);
    RLSEstimate *est_gcv = rls_solve_ridge(Phi, y, sel.lambda_opt, &opt);
    if (est_gcv) {
        printf("  theta = ");
        for (int j=0; j<p; j++) printf("%.4f ", est_gcv->theta[j]);
        printf("\n  MSE=%.6f, df=%.2f\n", est_gcv->mse, est_gcv->effective_df);
    }

    /* --- L-Curve Analysis --- */
    printf("\n--- L-Curve Analysis ---\n");
    double *lam_grid = rls_lambda_path(100.0, 1e-3, 30);
    double *log_rho = (double *)malloc(30 * sizeof(double));
    double *log_eta = (double *)malloc(30 * sizeof(double));
    rls_lcurve_compute(Phi, y, lam_grid, 30, RLS_REG_RIDGE, 0.0, &opt, log_rho, log_eta);
    double lam_lc = rls_lcurve_corner(log_rho, log_eta, 30, lam_grid);
    printf("  L-curve lambda = %.6e\n", lam_lc);
    printf("  L-curve (first 10 points):\n");
    for (int i = 0; i < 10; i++)
        printf("    lam=%.2e: log(rho)=%.4f, log(eta)=%.4f\n",
               lam_grid[i], log_rho[i], log_eta[i]);

    /* --- LASSO in correlated setting --- */
    printf("\n--- LASSO (sparse recovery in correlated setting) ---\n");
    RLSSolverConfig cd_cfg = rls_solver_config_default(RLS_SOLVER_CD);
    cd_cfg.cd_max_iter = 10000;
    double lam_lasso_max = rls_lambda_max_lasso(Phi, y, n);
    RLSEstimate *est_lasso = rls_solve_lasso_cd(Phi, y, lam_lasso_max*0.1, &opt, &cd_cfg);
    if (est_lasso) {
        printf("  theta = ");
        for (int j=0; j<p; j++) printf("%.4f ", est_lasso->theta[j]);
        printf("\n  Non-zero count: ");
        int nnz = 0;
        for (int j=0; j<p; j++) if (fabs(est_lasso->theta[j]) > 1e-6) nnz++;
        printf("%d\n", nnz);
    }

    /* --- SVD Analysis: singular value shrinkage --- */
    printf("\n--- SVD Analysis: Singular Value Shrinkage ---\n");
    int k = (n < p) ? n : p;
    RLSMatrix *U = rls_matrix_alloc(n, k);
    RLSVector *S = rls_vector_alloc(k);
    RLSMatrix *Vt = rls_matrix_alloc(k, p);
    if (rls_svd_decompose(U, S, Vt, Phi) == 0) {
        printf("  Singular values: ");
        for (int i = 0; i < (k<8?k:8); i++) printf("%.4f ", S->data[i]);
        printf("\n  Ridge shrinkage (lambda=%.4f):\n", sel.lambda_opt);
        for (int i = 0; i < (k<8?k:8); i++) {
            double si = S->data[i];
            double shrink = si*si / (si*si + sel.lambda_opt);
            printf("    s[%d]=%.4f -> shrinkage=%.4f\n", i, si, shrink);
        }
    }

    /* Cleanup */
    rls_matrix_free(Phi); rls_vector_free(y); rls_vector_free(theta_true);
    if (est_ols) rls_estimate_free(est_ols);
    if (est_gcv) rls_estimate_free(est_gcv);
    if (est_lasso) rls_estimate_free(est_lasso);
    rls_matrix_free(U); rls_vector_free(S); rls_matrix_free(Vt);
    free(lam_grid); free(log_rho); free(log_eta);
    free(sel.lambda_grid); free(sel.cv_scores);

    printf("\n=== Example 3 complete ===\n");
    return 0;
}
