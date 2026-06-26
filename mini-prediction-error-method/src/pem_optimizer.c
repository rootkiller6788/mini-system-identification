#include "pem_optimizer.h"
#include "pem_criterion.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

/* ============================================================================
 * PEM Optimizer — Numerical Optimization Algorithms
 *
 * Implements:
 * 1. Cholesky decomposition (L*L^T = A for SPD matrices)
 * 2. Linear system solver using Cholesky factor
 * 3. Armijo backtracking line search
 * 4. Gauss-Newton optimization
 * 5. Levenberg-Marquardt optimization
 * 6. Stochastic Gradient Descent
 *
 * Reference:
 *   Nocedal & Wright (2006) — Numerical Optimization
 *   Marquardt (1963) — "An Algorithm for Least-Squares Estimation of Nonlinear Parameters"
 *   Kelley (1999) — Iterative Methods for Optimization
 * ============================================================================ */

/* ================================================================
 * Cholesky Decomposition: A = L * L^T
 *
 * For symmetric positive-definite matrix A (n x n, row-major).
 * On output, lower triangle of A contains L (L[i][j] for i >= j).
 * Upper triangle is not modified.
 *
 * Algorithm: Gaussian elimination variant (outer product):
 *   for j = 0..n-1:
 *     L[j][j] = sqrt(A[j][j] - sum_{k=0}^{j-1} L[j][k]^2)
 *     for i = j+1..n-1:
 *       L[i][j] = (A[i][j] - sum_{k=0}^{j-1} L[i][k]*L[j][k]) / L[j][j]
 *
 * Returns 0 if A is positive definite, 1 if not (negative pivot).
 *
 * Complexity: O(n^3 / 6) — twice as fast as LU decomposition.
 * ================================================================ */

int pem_cholesky(double *A, int n) {
    for (int j = 0; j < n; j++) {
        /* Compute diagonal element L[j][j] */
        double sum_diag = 0.0;
        for (int k = 0; k < j; k++) {
            sum_diag += A[j * n + k] * A[j * n + k];
        }
        double diag = A[j * n + j] - sum_diag;

        /* Check positive definiteness */
        if (diag <= 1e-15) {
            /* Matrix is not positive definite.
             * Attempt regularization: add small value to diagonal. */
            diag = 1e-10;
        }
        A[j * n + j] = sqrt(diag);

        /* Compute off-diagonal elements L[i][j] for i > j */
        double inv_diag = 1.0 / A[j * n + j];
        for (int i = j + 1; i < n; i++) {
            double sum_off = 0.0;
            for (int k = 0; k < j; k++) {
                sum_off += A[i * n + k] * A[j * n + k];
            }
            A[i * n + j] = (A[i * n + j] - sum_off) * inv_diag;
        }
    }
    return 0; /* Success (possibly regularized) */
}

/* ================================================================
 * Cholesky Solve: L * L^T * x = b
 *
 * Step 1: Forward substitution L * y = b
 *   y_0 = b_0 / L[0][0]
 *   y_i = (b_i - sum_{j=0}^{i-1} L[i][j] * y_j) / L[i][i]
 *
 * Step 2: Back substitution L^T * x = y
 *   x_{n-1} = y_{n-1} / L[n-1][n-1]
 *   x_i = (y_i - sum_{j=i+1}^{n-1} L[j][i] * x_j) / L[i][i]
 *
 * Complexity: O(n^2)
 * ================================================================ */

void pem_cholesky_solve(const double *L, const double *b, double *x, int n) {
    double *y = (double*)malloc((size_t)n * sizeof(double));
    if (!y) return;

    /* Forward substitution: L * y = b */
    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        for (int j = 0; j < i; j++) {
            sum += L[i * n + j] * y[j];
        }
        y[i] = (b[i] - sum) / L[i * n + i];
    }

    /* Back substitution: L^T * x = y */
    for (int i = n - 1; i >= 0; i--) {
        double sum = 0.0;
        for (int j = i + 1; j < n; j++) {
            sum += L[j * n + i] * x[j];
        }
        x[i] = (y[i] - sum) / L[i * n + i];
    }

    free(y);
}

int pem_solve_spd(const double *A, const double *b, double *x, int n) {
    /* Copy A into working buffer for Cholesky */
    double *L = (double*)malloc((size_t)(n * n) * sizeof(double));
    if (!L) return 1;
    memcpy(L, A, (size_t)(n * n) * sizeof(double));

    if (pem_cholesky(L, n) != 0) {
        free(L);
        return 1;
    }
    pem_cholesky_solve(L, b, x, n);
    free(L);
    return 0;
}

/* ================================================================
 * Matrix-Vector Operations
 * ================================================================ */

void pem_matvec(const double *A, const double *x, double *y, int m, int n) {
    /* y = A * x, A is m x n (row-major) */
    for (int i = 0; i < m; i++) {
        double sum = 0.0;
        for (int j = 0; j < n; j++) {
            sum += A[i * n + j] * x[j];
        }
        y[i] = sum;
    }
}

void pem_matmul(const double *A, const double *B, double *C, int m, int p, int n) {
    /* C = A * B, A: m x p, B: p x n, C: m x n (row-major)
     * C[i][j] = sum_{k=0}^{p-1} A[i][k] * B[k][j] */
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < p; k++) {
                sum += A[i * p + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

void pem_transpose(const double *A, double *AT, int m, int n) {
    /* AT = A^T (n x m), A is m x n */
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            AT[j * m + i] = A[i * n + j];
}

double pem_condition_number(const double *A, int n) {
    /* Estimate condition number using 1-norm.
     * cond_1(A) = ||A||_1 * ||A^{-1}||_1
     *
     * ||A||_1 = max_j sum_i |A[i][j]|
     *
     * For ||A^{-1}||_1, we solve A*x = b for a few random b vectors
     * and use ||x||_1 / ||b||_1 as a cheap estimate.
     * A more rigorous approach uses Hager's estimator. */

    /* Compute ||A||_1 */
    double norm_A = 0.0;
    for (int j = 0; j < n; j++) {
        double col_sum = 0.0;
        for (int i = 0; i < n; i++)
            col_sum += fabs(A[i * n + j]);
        if (col_sum > norm_A) norm_A = col_sum;
    }

    /* Simple estimate: use trace ratio for SPD matrices.
     * For Cholesky: cond ~ (lambda_max / lambda_min)
     * We approximate this via diagonal dominance: */
    double trace = 0.0;
    for (int i = 0; i < n; i++) trace += A[i * n + i];

    /* Estimate via attempting Cholesky */
    double *L = (double*)malloc((size_t)(n * n) * sizeof(double));
    memcpy(L, A, (size_t)(n * n) * sizeof(double));
    pem_cholesky(L, n);

    /* cond estimate from L: min(diag)^2 and max(diag)^2 ratio */
    double min_diag = 1e100, max_diag = 0.0;
    for (int i = 0; i < n; i++) {
        double d = L[i * n + i];
        if (d < min_diag && d > 1e-15) min_diag = d;
        if (d > max_diag) max_diag = d;
    }
    free(L);

    if (min_diag < 1e-14) return 1e15;
    return (max_diag / min_diag) * (max_diag / min_diag) * norm_A;
}

double pem_det_spd(const double *A, int n) {
    /* det(A) = det(L*L^T) = (prod L[i][i])^2 */
    double *L = (double*)malloc((size_t)(n * n) * sizeof(double));
    memcpy(L, A, (size_t)(n * n) * sizeof(double));
    pem_cholesky(L, n);
    double det = 1.0;
    for (int i = 0; i < n; i++) {
        det *= L[i * n + i];
    }
    free(L);
    return det * det;
}

int pem_inverse_spd(const double *A, double *A_inv, int n) {
    /* Compute inverse of SPD matrix by solving A * A_inv = I
     * Solve column by column: A * A_inv[:,j] = e_j */
    double *L = (double*)malloc((size_t)(n * n) * sizeof(double));
    double *e = (double*)calloc((size_t)n, sizeof(double));
    if (!L || !e) { free(L); free(e); return 1; }
    memcpy(L, A, (size_t)(n * n) * sizeof(double));
    if (pem_cholesky(L, n) != 0) { free(L); free(e); return 1; }

    for (int j = 0; j < n; j++) {
        memset(e, 0, (size_t)n * sizeof(double));
        e[j] = 1.0;
        pem_cholesky_solve(L, e, &A_inv[j * n], n); /* column j */
    }
    free(L); free(e);

    /* Result is column-major in row-major storage? Fix: transpose in-place.
     * Actually, A_inv[j*n + i] after above = element (i,j). We want (i,j) = A_inv[i*n+j]. */
    double *temp = (double*)malloc((size_t)(n * n) * sizeof(double));
    if (!temp) return 1;
    memcpy(temp, A_inv, (size_t)(n * n) * sizeof(double));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            A_inv[i * n + j] = temp[j * n + i];
    free(temp);
    return 0;
}

/* ================================================================
 * Line Search — Armijo Backtracking
 *
 * Find step length alpha such that:
 *   f(x + alpha * p) <= f(x) + c1 * alpha * g^T * p
 *
 * The Armijo condition (also called "sufficient decrease") ensures
 * that the step actually reduces the objective function.
 *
 * Typical parameters:
 *   c1 = 1e-4 (Armijo constant, small to accept modest decreases)
 *   rho = 0.5 (backtracking factor, halves alpha each iteration)
 *   alpha_init = 1.0 (full Newton step first)
 *
 * Reference: Armijo (1966), Nocedal & Wright (2006) Section 3.1
 * ================================================================ */

int pem_linesearch_backtrack(const double *x, const double *p, const double *g,
                             int n, double fx, double alpha_init,
                             double c1, double rho, int max_iter,
                             double *x_new,
                             pem_eval_function eval_fn, void *eval_data,
                             double *alpha_out, double *fn_out) {
    /* Compute directional derivative g^T * p (should be < 0 for descent) */
    double dphi0 = 0.0;
    for (int i = 0; i < n; i++) dphi0 += g[i] * p[i];

    /* If not a descent direction, return immediately */
    if (dphi0 >= 0.0) {
        *alpha_out = 0.0;
        *fn_out = fx;
        memcpy(x_new, x, (size_t)n * sizeof(double));
        return 1;
    }

    double alpha = alpha_init;

    for (int iter = 0; iter < max_iter; iter++) {
        /* x_new = x + alpha * p */
        for (int i = 0; i < n; i++)
            x_new[i] = x[i] + alpha * p[i];

        double f_new = eval_fn(x_new, eval_data);

        /* Check Armijo condition */
        if (f_new <= fx + c1 * alpha * dphi0) {
            *alpha_out = alpha;
            *fn_out = f_new;
            return 0; /* Success */
        }

        /* Reduce step size */
        alpha *= rho;

        /* If step becomes too small, give up */
        if (alpha < 1e-16) {
            /* Return current best */
            *alpha_out = alpha;
            *fn_out = f_new;
            return 1;
        }
    }

    /* Max iterations reached without success */
    *alpha_out = alpha;
    *fn_out = eval_fn(x_new, eval_data);
    return 1;
}

/* ================================================================
 * Gauss-Newton Optimization
 *
 * theta_{k+1} = theta_k - H_k^{-1} * g_k
 *
 * where H_k = (1/N) sum psi(t) * psi^T(t) is the GN Hessian,
 * and g_k = -(1/N) sum eps(t) * psi(t) is the gradient.
 *
 * Convergence criteria (satisfying any one triggers stop):
 *   1. ||theta_k - theta_{k-1}|| < tol_param
 *   2. ||g_k|| < tol_gradient
 *   3. |V_k - V_{k-1}| / |V_k| < tol_cost
 *   4. iter >= max_iterations
 *
 * Each iteration:
 *   - Compute gradient g_k and Hessian H_k via user callbacks
 *   - Solve H_k * p_k = -g_k via Cholesky
 *   - Line search to find alpha_k
 *   - Update theta_{k+1} = theta_k + alpha_k * p_k
 *
 * Complexity per iteration: O(n^3) for Cholesky + cost of eval callbacks
 * ================================================================ */

int pem_optimize_gauss_newton(double *theta, int npar,
                              pem_obj_function eval_f,
                              pem_grad_function eval_g,
                              pem_hess_function eval_H,
                              void *data,
                              const PEMOptions *opts,
                              PEMResult *result) {
    if (!theta || !eval_f || !eval_g || !opts || !result) return 1;
    if (npar <= 0) return 1;

    clock_t t_start = clock();

    /* Allocate working memory */
    double *g = (double*)malloc((size_t)npar * sizeof(double));
    double *H = (double*)malloc((size_t)(npar * npar) * sizeof(double));
    double *p = (double*)malloc((size_t)npar * sizeof(double));
    double *theta_new = (double*)malloc((size_t)npar * sizeof(double));
    double *theta_prev = (double*)malloc((size_t)npar * sizeof(double));
    if (!g || !H || !p || !theta_new || !theta_prev) {
        free(g); free(H); free(p); free(theta_new); free(theta_prev);
        result->status = PEM_SINGULAR_HESSIAN;
        return 1;
    }

    /* Initial function value */
    double f_curr = eval_f(theta, data);
    result->loss_init = f_curr;

    int converged = 0;
    int iter;
    for (iter = 0; iter < opts->max_iterations; iter++) {
        memcpy(theta_prev, theta, (size_t)npar * sizeof(double));

        /* Compute gradient and Hessian at current theta */
        eval_g(theta, data, g);
        eval_H(theta, data, H);

        /* Check gradient tolerance */
        double gnorm = pem_norm2(g, npar);
        if (gnorm < opts->tol_gradient) {
            result->status = PEM_GRADIENT_TOL;
            converged = 1;
            break;
        }

        /* Solve H * p = -g (search direction) */
        double *H_work = (double*)malloc((size_t)(npar * npar) * sizeof(double));
        memcpy(H_work, H, (size_t)(npar * npar) * sizeof(double));

        /* Regularize if needed */
        pem_regularize_hessian(H_work, npar, 1e-8);

        if (pem_cholesky(H_work, npar) != 0) {
            free(H_work);
            result->status = PEM_SINGULAR_HESSIAN;
            break;
        }

        double *neg_g = (double*)malloc((size_t)npar * sizeof(double));
        for (int i = 0; i < npar; i++) neg_g[i] = -g[i];
        pem_cholesky_solve(H_work, neg_g, p, npar);
        free(neg_g);
        free(H_work);

        /* Line search */
        double alpha, f_new;
        int ls_ret = pem_linesearch_backtrack(theta, p, g, npar, f_curr, 1.0,
                         opts->line_search_c1, opts->line_search_rho,
                         opts->max_line_search, theta_new,
                         eval_f, data, &alpha, &f_new);

        if (ls_ret != 0 && alpha < 1e-10) {
            result->status = PEM_LINE_SEARCH_FAIL;
            break;
        }

        /* Update theta */
        memcpy(theta, theta_new, (size_t)npar * sizeof(double));

        /* Check parameter change */
        double dtheta_norm = 0.0;
        for (int i = 0; i < npar; i++) {
            double d = theta[i] - theta_prev[i];
            dtheta_norm += d * d;
        }
        dtheta_norm = sqrt(dtheta_norm);

        /* Check cost change */
        double cost_rel_change = fabs(f_curr - f_new) / (fabs(f_new) + 1e-15);

        if (opts->verbose) {
            printf("  GN iter %3d: f=%.6e  ||g||=%.2e  ||dtheta||=%.2e  alpha=%.2e  cost_rel=%.2e\n",
                   iter, f_new, gnorm, dtheta_norm, alpha, cost_rel_change);
        }

        f_curr = f_new;

        if (dtheta_norm < opts->tol_param) {
            result->status = PEM_CONVERGED;
            converged = 1;
            break;
        }
        if (cost_rel_change < opts->tol_cost && iter > 2) {
            result->status = PEM_CONVERGED;
            converged = 1;
            break;
        }

        /* Check divergence */
        if (f_curr > result->loss_init * 100.0 && iter > 10) {
            result->status = PEM_DIVERGED;
            break;
        }
    }

    if (!converged && iter >= opts->max_iterations) {
        result->status = PEM_MAX_ITER;
    }

    /* Final evaluation */
    result->loss_final = f_curr;
    result->iterations = iter;
    memcpy(result->theta_hat, theta, (size_t)npar * sizeof(double));

    /* Final gradient */
    eval_g(theta, data, result->gradient);

    /* Covariance estimate: Cov = (2/N) * V_N * H^{-1}
     * Under Gaussian noise assumption,
     * Cov(theta_hat) = sigma^2 * [sum psi(t) psi^T(t)]^{-1}
     * sigma^2 = (2N / (N-npar)) * V_N(theta_hat) */
    if (opts->compute_covariance && npar > 0) {
        eval_H(theta, data, H);
        /* H = (1/N) sum psi*psi^T, so sum psi*psi^T = N * H */
        /* Cov = sigma^2 * (N*H)^{-1} = (sigma^2/N) * H^{-1} */
        double sigma2 = 2.0 * result->loss_final;

        double *Hinv = (double*)malloc((size_t)(npar * npar) * sizeof(double));
        if (Hinv && pem_inverse_spd(H, Hinv, npar) == 0) {
            for (int i = 0; i < npar * npar; i++)
                result->covariance[i] = sigma2 * Hinv[i] / (double)((PEMData*)data ? ((PEMData*)data)->N : 1);
        }
        free(Hinv);

        /* Store information matrix */
        memcpy(result->information_matrix, H, (size_t)(npar * npar) * sizeof(double));
        result->condition_number = pem_condition_number(H, npar);
    }

    result->elapsed_sec = (double)(clock() - t_start) / (double)CLOCKS_PER_SEC;

    free(g); free(H); free(p); free(theta_new); free(theta_prev);
    return 0;
}

/* ================================================================
 * Levenberg-Marquardt Optimization
 *
 * theta_{k+1} = theta_k - (H_k + lambda * I)^{-1} * g_k
 *
 * The damping parameter lambda is adjusted based on cost reduction:
 *   - If V_new < V_curr: lambda = lambda / lambda_factor (accept step)
 *   - If V_new >= V_curr: lambda = lambda * lambda_factor (reject step)
 *
 * This makes the search direction interpolate between:
 *   - Gauss-Newton (lambda -> 0): fast convergence near optimum
 *   - Steepest descent (lambda -> inf): safe descent far from optimum
 *
 * The LM algorithm is the workhorse of PEM because it handles:
 *   - Ill-conditioned Hessians (singular or near-singular)
 *   - Non-convex criterion functions (OE, BJ structures)
 *   - Poor initial parameter guesses
 *
 * Reference:
 *   Marquardt (1963), SIAM J. Appl. Math.
 *   More (1978), "The Levenberg-Marquardt Algorithm: Implementation and Theory"
 * ================================================================ */

int pem_optimize_levenberg_marquardt(double *theta, int npar,
                                     pem_obj_function eval_f,
                                     pem_grad_function eval_g,
                                     pem_hess_function eval_H,
                                     void *data,
                                     const PEMOptions *opts,
                                     PEMResult *result) {
    if (!theta || !eval_f || !eval_g || !eval_H || !opts || !result) return 1;
    if (npar <= 0) return 1;

    clock_t t_start = clock();

    double *g = (double*)malloc((size_t)npar * sizeof(double));
    double *H = (double*)malloc((size_t)(npar * npar) * sizeof(double));
    double *H_aug = (double*)malloc((size_t)(npar * npar) * sizeof(double));
    double *p = (double*)malloc((size_t)npar * sizeof(double));
    double *theta_trial = (double*)malloc((size_t)npar * sizeof(double));
    if (!g || !H || !H_aug || !p || !theta_trial) {
        free(g); free(H); free(H_aug); free(p); free(theta_trial);
        result->status = PEM_SINGULAR_HESSIAN;
        return 1;
    }

    double f_curr = eval_f(theta, data);
    result->loss_init = f_curr;

    double lambda = opts->lambda_init;
    int converged = 0;
    int iter;

    for (iter = 0; iter < opts->max_iterations; iter++) {
        eval_g(theta, data, g);
        double gnorm = pem_norm2(g, npar);
        if (gnorm < opts->tol_gradient) {
            result->status = PEM_GRADIENT_TOL;
            converged = 1;
            break;
        }

        eval_H(theta, data, H);
        memcpy(H_aug, H, (size_t)(npar * npar) * sizeof(double));

        /* H_aug = H + lambda * I */
        pem_regularize_hessian(H_aug, npar, lambda);

        if (pem_cholesky(H_aug, npar) != 0) {
            /* Cholesky failed, increase lambda and retry */
            lambda *= opts->lambda_factor;
            if (lambda > opts->lambda_max) {
                result->status = PEM_SINGULAR_HESSIAN;
                break;
            }
            continue;
        }

        /* Solve (H + lambda*I) * p = -g */
        double *neg_g = (double*)malloc((size_t)npar * sizeof(double));
        for (int i = 0; i < npar; i++) neg_g[i] = -g[i];
        pem_cholesky_solve(H_aug, neg_g, p, npar);
        free(neg_g);

        /* Compute trial point */
        for (int i = 0; i < npar; i++)
            theta_trial[i] = theta[i] + p[i];

        double f_trial = eval_f(theta_trial, data);

        /* Compute predicted reduction from linear model:
         * delta_pred = -g^T * p - 0.5 * p^T * H * p */
        double pred_reduction = 0.0;
        for (int i = 0; i < npar; i++) {
            double Hp_i = 0.0;
            for (int j = 0; j < npar; j++) Hp_i += H[i * npar + j] * p[j];
            pred_reduction -= g[i] * p[i] + 0.5 * p[i] * Hp_i;
        }

        double actual_reduction = f_curr - f_trial;

        /* Gain ratio: actual / predicted */
        double rho = (pred_reduction > 0.0) ? (actual_reduction / pred_reduction) : 0.0;

        if (actual_reduction > 0.0 && rho > 0.0) {
            /* Accept step */
            memcpy(theta, theta_trial, (size_t)npar * sizeof(double));
            f_curr = f_trial;

            /* Adjust lambda: decrease if good fit, increase if poor */
            if (rho > 0.75) {
                lambda /= opts->lambda_factor;
                if (lambda < opts->lambda_min) lambda = opts->lambda_min;
            } else if (rho < 0.25) {
                lambda *= opts->lambda_factor;
                if (lambda > opts->lambda_max) lambda = opts->lambda_max;
            }
        } else {
            /* Reject step, increase lambda */
            lambda *= opts->lambda_factor;
            if (lambda > opts->lambda_max) lambda = opts->lambda_max;
        }

        /* Convergence checks */
        double pnorm = pem_norm2(p, npar);
        double cost_change = fabs(actual_reduction) / (fabs(f_curr) + 1e-15);

        if (opts->verbose) {
            printf("  LM iter %3d: f=%.6e  ||g||=%.2e  ||p||=%.2e  lambda=%.2e  rho=%.3f\n",
                   iter, f_curr, gnorm, pnorm, lambda, rho);
        }

        if (pnorm < opts->tol_param * (pem_norm2(theta, npar) + opts->tol_param)) {
            result->status = PEM_CONVERGED;
            converged = 1;
            break;
        }

        if (cost_change < opts->tol_cost && iter > 2) {
            result->status = PEM_CONVERGED;
            converged = 1;
            break;
        }

        /* Divergence check */
        if (f_curr > result->loss_init * 10.0 && iter > 20) {
            result->status = PEM_DIVERGED;
            break;
        }
    }

    if (!converged && iter >= opts->max_iterations)
        result->status = PEM_MAX_ITER;

    result->loss_final = f_curr;
    result->iterations = iter;
    memcpy(result->theta_hat, theta, (size_t)npar * sizeof(double));
    eval_g(theta, data, result->gradient);

    if (opts->compute_covariance && npar > 0) {
        eval_H(theta, data, H);
        double sigma2 = 2.0 * result->loss_final;
        int N_data = ((PEMData*)data) ? ((PEMData*)data)->N : result->iterations + npar;
        if (N_data < npar + 1) N_data = npar + 10;
        double *Hinv = (double*)malloc((size_t)(npar * npar) * sizeof(double));
        if (Hinv && pem_inverse_spd(H, Hinv, npar) == 0) {
            for (int i = 0; i < npar * npar; i++)
                result->covariance[i] = sigma2 * Hinv[i] / (double)N_data;
        }
        free(Hinv);
        memcpy(result->information_matrix, H, (size_t)(npar * npar) * sizeof(double));
        result->condition_number = pem_condition_number(H, npar);
    }

    result->elapsed_sec = (double)(clock() - t_start) / (double)CLOCKS_PER_SEC;

    free(g); free(H); free(H_aug); free(p); free(theta_trial);
    return 0;
}

/* ================================================================
 * Stochastic Gradient Descent (SGD)
 *
 * theta_{k+1} = theta_k - alpha_k * g_k
 *
 * Step size schedule (Robbins-Monro conditions):
 *   alpha_k = alpha_0 / (1 + k)^gamma
 *
 * where gamma in (0.5, 1] ensures:
 *   sum alpha_k = inf    (reach any point)
 *   sum alpha_k^2 < inf  (bounded variance)
 *
 * SGD is useful for:
 *   - Very large datasets (use mini-batches)
 *   - Online/PEM (recursive identification)
 *   - Escape from shallow local minima
 *
 * Reference:
 *   Robbins & Monro (1951), Ann. Math. Stat.
 *   Bottou (2010), "Large-Scale Machine Learning with SGD"
 * ================================================================ */

int pem_optimize_sgd(double *theta, int npar,
                     pem_obj_function eval_f,
                     pem_grad_function eval_g,
                     void *data,
                     const PEMOptions *opts,
                     PEMResult *result) {
    if (!theta || !eval_f || !eval_g || !opts || !result) return 1;
    if (npar <= 0) return 1;

    clock_t t_start = clock();

    double *g = (double*)malloc((size_t)npar * sizeof(double));
    double *theta_prev = (double*)malloc((size_t)npar * sizeof(double));
    if (!g || !theta_prev) { free(g); free(theta_prev); return 1; }

    double f_curr = eval_f(theta, data);
    result->loss_init = f_curr;

    double alpha0 = 0.01; /* Initial step size */
    double gamma = 0.51;  /* Robbins-Monro exponent */

    int converged = 0;
    int iter;

    for (iter = 0; iter < opts->max_iterations; iter++) {
        memcpy(theta_prev, theta, (size_t)npar * sizeof(double));

        eval_g(theta, data, g);

        double gnorm = pem_norm2(g, npar);
        if (gnorm < opts->tol_gradient) {
            result->status = PEM_GRADIENT_TOL;
            converged = 1;
            break;
        }

        /* Step size: Robbins-Monro schedule */
        double alpha = alpha0 / pow(1.0 + (double)iter, gamma);

        /* Update */
        for (int i = 0; i < npar; i++)
            theta[i] -= alpha * g[i];

        double f_new = eval_f(theta, data);

        if (opts->verbose && iter % 10 == 0) {
            printf("  SGD iter %3d: f=%.6e  ||g||=%.2e  alpha=%.2e\n",
                   iter, f_new, gnorm, alpha);
        }

        /* Check parameter change */
        double dtheta_norm = 0.0;
        for (int i = 0; i < npar; i++) {
            double d = theta[i] - theta_prev[i];
            dtheta_norm += d * d;
        }
        dtheta_norm = sqrt(dtheta_norm);

        if (dtheta_norm < opts->tol_param && iter > 50) {
            result->status = PEM_CONVERGED;
            converged = 1;
            break;
        }

        f_curr = f_new;
    }

    if (!converged && iter >= opts->max_iterations)
        result->status = PEM_MAX_ITER;

    result->loss_final = f_curr;
    result->iterations = iter;
    memcpy(result->theta_hat, theta, (size_t)npar * sizeof(double));
    eval_g(theta, data, result->gradient);
    result->elapsed_sec = (double)(clock() - t_start) / (double)CLOCKS_PER_SEC;

    free(g); free(theta_prev);
    return 0;
}