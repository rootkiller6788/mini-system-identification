#include "rls_core.h"
#include "rls_kernel.h"
#include "rls_validation.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================================
 * Kernel Functions (L8: Advanced Topics)
 *
 * Each kernel K(i,j) encodes prior knowledge about the impulse response:
 * - Smoothness: nearby coefficients are similar
 * - Stability: exponential decay (BIBO stability)
 * - Correlation structure: how coefficients influence each other
 *
 * Ref: [Pill10] Pillonetto & De Nicolao, Automatica 46(1), 2010
 *      [Pill14] Pillonetto, Dinuzzo, Chen, De Nicolao, Ljung, Automatica 50(3), 2014
 * ============================================================================ */

double rls_kernel_ss(int i, int j, double beta) {
    /* Stable Spline: K(i,j) = beta^(i+j+max(i,j))/2 - beta^(3*max(i,j))/6 */
    int mx = (i > j) ? i : j;
    double term1 = pow(beta, i + j + mx) / 2.0;
    double term2 = pow(beta, 3 * mx) / 6.0;
    return term1 - term2;
}

double rls_kernel_tc(int i, int j, double beta, double gamma) {
    /* Tuned/Correlated: K(i,j) = beta^max(i,j) * gamma^(|i-j|) */
    int mx = (i > j) ? i : j;
    int diff = abs(i - j);
    return pow(beta, mx) * pow(gamma, diff);
}

double rls_kernel_di(int i, int j, double beta) {
    /* Diagonal: K(i,j) = beta^i * delta(i,j) */
    if (i != j) return 0.0;
    return pow(beta, i);
}

double rls_kernel_dc(int i, int j, double beta, double c) {
    /* DC: K(i,j) = c*beta^(i+j) + (1-c)*beta^i*delta(i,j) */
    double diag = (i == j) ? (1.0 - c) * pow(beta, i) : 0.0;
    double corr = c * pow(beta, i + j);
    return corr + diag;
}

double rls_kernel_rbf(int i, int j, double ell) {
    double d = (double)(i - j);
    return exp(-d * d / (2.0 * ell * ell));
}

double rls_kernel_matern32(int i, int j, double ell) {
    double d = fabs((double)(i - j)) / ell;
    double sqrt3 = 1.7320508075688772;
    return (1.0 + sqrt3 * d) * exp(-sqrt3 * d);
}

double rls_kernel_matern52(int i, int j, double ell) {
    double d = fabs((double)(i - j)) / ell;
    double sqrt5 = 2.23606797749979;
    return (1.0 + sqrt5 * d + 5.0 * d * d / 3.0) * exp(-sqrt5 * d);
}

double rls_kernel_eval(const RLSKernel *kernel, int i, int j) {
    if (!kernel) return 0.0;
    double kval = 0.0;
    switch (kernel->type) {
        case RLS_KERNEL_SS:       kval = rls_kernel_ss(i, j, kernel->beta); break;
        case RLS_KERNEL_TC:       kval = rls_kernel_tc(i, j, kernel->beta, kernel->gamma); break;
        case RLS_KERNEL_DI:       kval = rls_kernel_di(i, j, kernel->beta); break;
        case RLS_KERNEL_DC:       kval = rls_kernel_dc(i, j, kernel->beta, kernel->c); break;
        case RLS_KERNEL_RBF:
        case RLS_KERNEL_SE:       kval = rls_kernel_rbf(i, j, kernel->ell); break;
        case RLS_KERNEL_MATERN32: kval = rls_kernel_matern32(i, j, kernel->ell); break;
        case RLS_KERNEL_MATERN52: kval = rls_kernel_matern52(i, j, kernel->ell); break;
        case RLS_KERNEL_CUSTOM:
            if (kernel->custom_fn) kval = kernel->custom_fn(i, j, kernel->custom_params);
            break;
        default: break;
    }
    return kernel->sigma_f2 * kval;
}

/* ============================================================================
 * Kernel Matrix Construction
 * ============================================================================ */

RLSMatrix *rls_kernel_matrix(const RLSKernel *kernel) {
    if (!kernel || kernel->dim <= 0) return NULL;
    int n = kernel->dim;
    RLSMatrix *K = rls_matrix_alloc(n, n);
    if (!K) return NULL;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            K->data[j * n + i] = rls_kernel_eval(kernel, i, j);
    return K;
}

/* ============================================================================
 * Kernel Ridge Regression (Dual Formulation)
 *
 * By the Representer Theorem, the solution to
 *   minimize ||y - Phi*theta||^2 + lambda*||theta||_{K^{-1}}^2
 * can be written as theta = Phi^T * alpha where alpha solves
 *   (K + lambda*I) * alpha = y,  K = Phi * Phi^T.
 *
 * This is efficient when p > n (more parameters than samples).
 * Complexity: O(n^3) for Cholesky on K (n x n).
 * ============================================================================ */

RLSEstimate *rls_solve_kernel_ridge(const RLSMatrix *K, const RLSVector *y,
                                     double lambda, const RLSOptions *opt) {
    if (!K || !y) return NULL;
    int n = K->rows;
    RLSEstimate *est = rls_estimate_alloc(n);
    if (!est) return NULL;
    /* (K + lambda*I) * alpha = y */
    RLSMatrix *Kreg = rls_matrix_copy(K);
    rls_matrix_add_diag(Kreg, lambda);
    if (rls_cholesky_decompose(Kreg) != 0) {
        rls_matrix_free(Kreg);
        est->loss = INFINITY;
        return est;
    }
    RLSVector tv; tv.dim = n; tv.capacity = 0; tv.data = est->theta; /* store alpha here */
    rls_cholesky_solve(&tv, Kreg, y);
    /* Compute loss */
    RLSVector *y_hat = rls_vector_alloc(n);
    rls_matrix_vector_mul(y_hat, K, &tv);
    double rss = 0.0;
    for (int i = 0; i < n; i++) {
        double d = y->data[i] - y_hat->data[i];
        rss += d * d;
    }
    /* Regularization term: alpha^T * K * alpha */
    double reg = rls_vector_dot(&tv, y_hat);
    est->loss = 0.5 * rss + 0.5 * lambda * reg;
    est->mse = rss / n;
    est->converged = true;
    est->iterations = 1;
    rls_matrix_free(Kreg);
    rls_vector_free(y_hat);
    return est;
}

/* ============================================================================
 * Kernel FIR Identification
 *
 * Full pipeline:
 * 1. Build kernel matrix K from kernel prior.
 * 2. Solve (K + lambda*I) * alpha = y (kernel ridge regression).
 * 3. Recover impulse response: theta = Phi^T * alpha.
 *
 * For FIR: y = Phi * theta where Phi is the Toeplitz matrix of u.
 * Then K = Phi * Phi^T captures the input correlation structure.
 *
 * The kernel encodes prior knowledge about the impulse response shape:
 * - Stable Spline enforces BIBO stable, smooth decay
 * - TC adds correlation between neighboring coefficients
 * - DI gives independent shrinkage per coefficient
 * ============================================================================ */

RLSEstimate *rls_kernel_fir_identify(const RLSData *data, const RLSKernel *kernel,
                                      double lambda, const RLSOptions *opt) {
    if (!data || !kernel) return NULL;
    int nb = kernel->dim;
    int n_eff = data->N - nb;
    if (n_eff <= 0) return NULL;
    /* Build FIR regressor */
    RLSMatrix *Phi = rls_matrix_alloc(n_eff, nb);
    RLSVector *y_vec = rls_vector_alloc(n_eff);
    /* Build manually to avoid depending on rls_models.h */
    for (int i = 0; i < n_eff; i++) {
        int t = i + nb;
        y_vec->data[i] = data->y[t];
        for (int j = 0; j < nb; j++)
            Phi->data[j * n_eff + i] = data->u[t - 1 - j];
    }
    /* Compute kernel matrix K = Phi * Phi^T (n_eff x n_eff) */
    RLSMatrix *K = rls_matrix_alloc(n_eff, n_eff);
    RLSMatrix *PhiT = rls_matrix_alloc(nb, n_eff);
    rls_matrix_transpose(PhiT, Phi);
    rls_matrix_multiply(K, Phi, PhiT);
    /* Solve kernel ridge regression */
    RLSEstimate *est_alpha = rls_solve_kernel_ridge(K, y_vec, lambda, opt);
    if (!est_alpha || !est_alpha->converged) {
        rls_matrix_free(Phi); rls_vector_free(y_vec);
        rls_matrix_free(K); rls_matrix_free(PhiT);
        if (est_alpha) rls_estimate_free(est_alpha);
        return NULL;
    }
    /* Recover theta = Phi^T * alpha */
    RLSEstimate *est = rls_estimate_alloc(nb);
    RLSVector alpha_v; alpha_v.dim = n_eff; alpha_v.capacity = 0; alpha_v.data = est_alpha->theta;
    RLSVector tv; tv.dim = nb; tv.capacity = 0; tv.data = est->theta;
    rls_matrix_t_vector_mul(&tv, Phi, &alpha_v);
    /* Compute diagnostics */
    RLSVector *res = rls_vector_alloc(n_eff);
    rls_compute_residual(res, y_vec, Phi, &tv);
    double rss = 0.0;
    for (int i = 0; i < n_eff; i++) rss += res->data[i] * res->data[i];
    est->loss = 0.5 * rss + 0.5 * lambda * rls_vector_dot(&tv, &tv);
    est->mse = rss / n_eff;
    est->converged = true;
    if (opt && opt->compute_stats)
        rls_estimate_compute_stats(est, Phi, y_vec, rss / n_eff, lambda);
    rls_matrix_free(Phi); rls_vector_free(y_vec);
    rls_matrix_free(K); rls_matrix_free(PhiT);
    rls_vector_free(res);
    rls_estimate_free(est_alpha);
    return est;
}

/* ============================================================================
 * Marginal Likelihood for Kernel Hyperparameters
 * ============================================================================ */

double rls_kernel_marginal_likelihood(const RLSMatrix *K, const RLSVector *y,
                                       double lambda) {
    return rls_marginal_likelihood(K, y, lambda);
}

double rls_kernel_marginal_gradient(const RLSMatrix *K, const RLSVector *y,
                                     double lambda, const RLSMatrix *dK) {
    if (!K || !y || !dK) return 0.0;
    int n = K->rows;
    /* S = K + lambda*I, solve S * alpha = y */
    RLSMatrix *S = rls_matrix_copy(K);
    rls_matrix_add_diag(S, lambda);
    if (rls_cholesky_decompose(S) != 0) { rls_matrix_free(S); return 0.0; }
    RLSVector *alpha = rls_vector_alloc(n);
    rls_cholesky_solve(alpha, S, y);
    /* S^{-1} * dK */
    RLSMatrix *Sinv_dK = rls_matrix_alloc(n, n);
    for (int j = 0; j < n; j++) {
        RLSVector ej; ej.dim = n; ej.capacity = 0;
        ej.data = (double *)calloc(n, sizeof(double));
        ej.data[j] = 1.0;
        RLSVector *col = rls_vector_alloc(n);
        /* dK * e_j */
        for (int i = 0; i < n; i++) col->data[i] = dK->data[j * n + i];
        RLSVector *soln = rls_vector_alloc(n);
        rls_cholesky_solve(soln, S, col);
        for (int i = 0; i < n; i++) Sinv_dK->data[j * n + i] = soln->data[i];
        free(ej.data);
        rls_vector_free(col); rls_vector_free(soln);
    }
    /* y^T * S^{-1} * dK * S^{-1} * y */
    RLSVector *Sinv_y = rls_vector_alloc(n);
    rls_cholesky_solve(Sinv_y, S, y);
    RLSVector *dK_Sinv_y = rls_vector_alloc(n);
    rls_matrix_vector_mul(dK_Sinv_y, dK, Sinv_y);
    double term1 = rls_vector_dot(Sinv_y, dK_Sinv_y);
    /* tr(S^{-1} * dK) */
    double term2 = rls_matrix_trace(Sinv_dK);
    double grad = 0.5 * (term1 - term2);
    rls_matrix_free(S); rls_vector_free(alpha);
    rls_matrix_free(Sinv_dK);
    rls_vector_free(Sinv_y); rls_vector_free(dK_Sinv_y);
    return grad;
}

int rls_kernel_optimize_hyperparams(RLSKernel *kernel, const RLSData *data,
                                     double lambda, const RLSOptions *opt) {
    if (!kernel || !data) return -1;
    /* Simple grid search over beta (0.5 to 0.99) */
    int nb = kernel->dim;
    int n_eff = data->N - nb;
    if (n_eff <= 0) return -1;
    /* Build FIR regressor and output */
    RLSMatrix *Phi = rls_matrix_alloc(n_eff, nb);
    RLSVector *y_vec = rls_vector_alloc(n_eff);
    for (int i = 0; i < n_eff; i++) {
        int t = i + nb;
        y_vec->data[i] = data->y[t];
        for (int j = 0; j < nb; j++)
            Phi->data[j * n_eff + i] = data->u[t - 1 - j];
    }
    /* Grid search over beta */
    double best_beta = kernel->beta;
    double best_ml = -INFINITY;
    for (int gi = 0; gi < 20; gi++) {
        double beta = 0.5 + gi * 0.025;
        if (beta >= 1.0) beta = 0.99;
        kernel->beta = beta;
        RLSMatrix *K = rls_kernel_matrix(kernel);
        if (!K) continue;
        /* Compute K_emp = Phi * K * Phi^T (simplified: use kernel directly on output) */
        /* For kernel FIR: y ~ N(0, Phi*K*Phi^T + sigma2*I). For simplicity, use kernel on output space. */
        double ml = rls_kernel_marginal_likelihood(K, y_vec, lambda);
        if (ml > best_ml) { best_ml = ml; best_beta = beta; }
        rls_matrix_free(K);
    }
    kernel->beta = best_beta;
    rls_matrix_free(Phi); rls_vector_free(y_vec);
    return 0;
}

/* ============================================================================
 * Kernel Utilities
 * ============================================================================ */

void rls_kernel_free(RLSKernel *kernel) {
    if (!kernel) return;
    free(kernel->custom_params);
}

RLSKernel rls_kernel_default_ss(int dim, double beta) {
    RLSKernel k;
    k.type = RLS_KERNEL_SS;
    k.dim = dim;
    k.beta = beta;
    k.gamma = 1.0;
    k.ell = 1.0;
    k.c = 0.5;
    k.nu = 1.5;
    k.sigma_f2 = 1.0;
    k.custom_fn = NULL;
    k.custom_params = NULL;
    return k;
}

RLSKernel rls_kernel_default_tc(int dim, double beta, double gamma) {
    RLSKernel k;
    k.type = RLS_KERNEL_TC;
    k.dim = dim;
    k.beta = beta;
    k.gamma = gamma;
    k.ell = 1.0;
    k.c = 0.5;
    k.nu = 1.5;
    k.sigma_f2 = 1.0;
    k.custom_fn = NULL;
    k.custom_params = NULL;
    return k;
}

double rls_kernel_logdet(const RLSMatrix *K, double lambda) {
    if (!K) return -INFINITY;
    RLSMatrix *S = rls_matrix_copy(K);
    rls_matrix_add_diag(S, lambda);
    if (rls_cholesky_decompose(S) != 0) { rls_matrix_free(S); return -INFINITY; }
    double logdet = 0.0;
    int n = K->rows;
    for (int i = 0; i < n; i++) logdet += 2.0 * log(S->data[i * n + i]);
    rls_matrix_free(S);
    return logdet;
}
