#include "rls_core.h"
#include "rls_solvers.h"
#include "rls_regularizers.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================================
 * Ridge Regression Solvers
 * ============================================================================ */

/** Ridge via Cholesky on normal equations.
 *  Solve (Phi^T Phi + lambda I) theta = Phi^T y.
 *  Form XtX (p x p), add lambda to diagonal, Cholesky, then solve.
 *  Theorem (L4): Normal equations give the unique minimizer of
 *    ||y - Phi*theta||_2^2 + lambda*||theta||_2^2
 *  because the objective is strongly convex for lambda > 0. */
RLSEstimate *rls_solve_ridge_cholesky(const RLSMatrix *Phi, const RLSVector *y,
                                       double lambda, const RLSOptions *opt) {
    if (!Phi || !y) return NULL;
    int n = Phi->rows, p = Phi->cols;
    RLSEstimate *est = rls_estimate_alloc(p);
    if (!est) return NULL;
    /* XtX = Phi^T * Phi (p x p) */
    RLSMatrix *XtX = rls_matrix_alloc(p, p);
    rls_gram_matrix(XtX, Phi);
    /* Add lambda to diagonal */
    rls_matrix_add_diag(XtX, lambda);
    /* RHS = Phi^T * y */
    RLSVector *rhs = rls_vector_alloc(p);
    rls_matrix_t_vector_mul(rhs, Phi, y);
    /* Cholesky decomposition */
    if (rls_cholesky_decompose(XtX) != 0) {
        /* Fall back to LDL^T for better numerical stability */
        rls_matrix_free(XtX);
        XtX = rls_matrix_alloc(p, p);
        rls_gram_matrix(XtX, Phi);
        rls_matrix_add_diag(XtX, lambda);
        if (rls_ldlt_decompose(XtX) != 0) {
            est->converged = false;
            est->loss = INFINITY;
            rls_matrix_free(XtX); rls_vector_free(rhs);
            return est;
        }
        RLSVector theta_v; theta_v.dim = p; theta_v.capacity = 0; theta_v.data = est->theta;
        rls_ldlt_solve(&theta_v, XtX, rhs);
    } else {
        RLSVector theta_v; theta_v.dim = p; theta_v.capacity = 0; theta_v.data = est->theta;
        rls_cholesky_solve(&theta_v, XtX, rhs);
    }
    /* Compute loss */
    RLSVector *res = rls_vector_alloc(n);
    RLSVector tv; tv.dim = p; tv.capacity = 0; tv.data = est->theta;
    rls_compute_residual(res, y, Phi, &tv);
    double rss = 0.0;
    for (int i = 0; i < n; i++) rss += res->data[i] * res->data[i];
    est->loss = 0.5 * rss + rls_penalty_ridge(&tv, lambda);
    est->converged = true;
    est->iterations = 1;
    if (opt && opt->compute_stats)
        rls_estimate_compute_stats(est, Phi, y, rss / n, lambda);
    rls_matrix_free(XtX); rls_vector_free(rhs); rls_vector_free(res);
    return est;
}

/** Ridge via SVD.
 *  theta = sum_i (s_i / (s_i^2 + lambda)) * (u_i^T y) * v_i
 *  where s_i are singular values, u_i left sing. vectors, v_i right sing. vectors.
 *  More numerically stable for ill-conditioned problems.
 *  Also provides singular values for GCV/effective df computation. */
RLSEstimate *rls_solve_ridge_svd(const RLSMatrix *Phi, const RLSVector *y,
                                  double lambda, const RLSOptions *opt) {
    if (!Phi || !y) return NULL;
    int m = Phi->rows, n = Phi->cols;
    int k = (m < n) ? m : n;
    RLSEstimate *est = rls_estimate_alloc(n);
    if (!est) return NULL;
    /* SVD */
    RLSMatrix *U = rls_matrix_alloc(m, k);
    RLSVector *S = rls_vector_alloc(k);
    RLSMatrix *Vt = rls_matrix_alloc(k, n);
    if (rls_svd_decompose(U, S, Vt, Phi) != 0) {
        rls_matrix_free(U); rls_vector_free(S); rls_matrix_free(Vt);
        est->loss = INFINITY;
        return est;
    }
    /* U^T * y */
    RLSVector *utb = rls_vector_alloc(k);
    rls_matrix_t_vector_mul(utb, U, y);
    /* z_i = s_i * (u_i^T y) / (s_i^2 + lambda) */
    RLSVector *z = rls_vector_alloc(k);
    for (int i = 0; i < k; i++) {
        double si = S->data[i];
        z->data[i] = si * utb->data[i] / (si * si + lambda);
    }
    /* theta = Vt^T * z */
    RLSVector tv; tv.dim = n; tv.capacity = 0; tv.data = est->theta;
    rls_matrix_t_vector_mul(&tv, Vt, z);
    /* Compute loss and stats */
    RLSVector *res = rls_vector_alloc(m);
    rls_compute_residual(res, y, Phi, &tv);
    double rss = 0.0;
    for (int i = 0; i < m; i++) rss += res->data[i] * res->data[i];
    est->loss = 0.5 * rss + rls_penalty_ridge(&tv, lambda);
    est->converged = true;
    est->iterations = 1;
    if (opt && opt->compute_stats)
        rls_estimate_compute_stats(est, Phi, y, rss / m, lambda);
    rls_matrix_free(U); rls_vector_free(S); rls_matrix_free(Vt);
    rls_vector_free(utb); rls_vector_free(z); rls_vector_free(res);
    return est;
}

/** Ridge via QR.
 *  Augmented system approach: [Phi; sqrt(lambda)*I] * theta = [y; 0]
 *  QR of augmented matrix, then R * theta = Q^T * [y; 0].
 *  Back substitution gives theta. */
RLSEstimate *rls_solve_ridge_qr(const RLSMatrix *Phi, const RLSVector *y,
                                 double lambda, const RLSOptions *opt) {
    if (!Phi || !y) return NULL;
    int m = Phi->rows, n = Phi->cols;
    RLSEstimate *est = rls_estimate_alloc(n);
    if (!est) return NULL;
    /* Augmented matrix [Phi; sqrt(lambda)*I] of size (m+n) x n */
    RLSMatrix *Aaug = rls_matrix_alloc(m + n, n);
    /* Copy Phi into upper part */
    for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++)
            Aaug->data[j * (m + n) + i] = Phi->data[j * m + i];
    /* Add sqrt(lambda)*I in lower part */
    double s_lambda = sqrt(lambda);
    for (int j = 0; j < n; j++)
        Aaug->data[j * (m + n) + (m + j)] = s_lambda;
    /* Augmented RHS [y; 0] */
    RLSVector *baug = rls_vector_alloc(m + n);
    for (int i = 0; i < m; i++) baug->data[i] = y->data[i];
    /* QR decomposition */
    RLSVector *tau = rls_vector_alloc(n);
    if (rls_qr_decompose(Aaug, tau) != 0) {
        rls_matrix_free(Aaug); rls_vector_free(baug); rls_vector_free(tau);
        est->loss = INFINITY;
        return est;
    }
    /* Solve R*theta = Q^T * baug (upper part only) */
    RLSVector tv; tv.dim = n; tv.capacity = 0; tv.data = est->theta;
    rls_qr_solve(&tv, Aaug, tau, baug);
    /* Drop extra dimensions from solve (account for sqrt(lambda)*I rows) */
    /* The QR solve gives the correct theta for the augmented system */
    /* Compute loss */
    RLSVector *res = rls_vector_alloc(m);
    rls_compute_residual(res, y, Phi, &tv);
    double rss = 0.0;
    for (int i = 0; i < m; i++) rss += res->data[i] * res->data[i];
    est->loss = 0.5 * rss + rls_penalty_ridge(&tv, lambda);
    est->converged = true;
    est->iterations = 1;
    if (opt && opt->compute_stats)
        rls_estimate_compute_stats(est, Phi, y, rss / m, lambda);
    rls_matrix_free(Aaug); rls_vector_free(baug); rls_vector_free(tau);
    rls_vector_free(res);
    return est;
}

/** Ridge via Conjugate Gradient.
 *  Minimize f(theta) = 0.5*||y - Phi*theta||^2 + 0.5*lambda*||theta||^2
 *  Gradient: g = -Phi^T*(y - Phi*theta) + lambda*theta
 *  CG iteration on the normal equations without forming XtX explicitly. */
RLSEstimate *rls_solve_ridge_cg(const RLSMatrix *Phi, const RLSVector *y,
                                 double lambda, const RLSOptions *opt,
                                 const RLSSolverConfig *solver) {
    if (!Phi || !y) return NULL;
    int m = Phi->rows, n = Phi->cols;
    RLSEstimate *est = rls_estimate_alloc(n);
    if (!est) return NULL;
    RLSVector *r = rls_vector_alloc(n);  /* residual in normal eqs */
    RLSVector *p = rls_vector_alloc(n);  /* search direction */
    RLSVector *Ap = rls_vector_alloc(n); /* A*p */
    RLSVector *tmp = rls_vector_alloc(m);
    /* Initial residual: r0 = Phi^T*y */
    rls_matrix_t_vector_mul(r, Phi, y);
    /* Initial theta = 0 */
    rls_vector_copy_to(p, r);
    double rsold = rls_vector_dot(r, r);
    int max_iter = solver ? solver->cg_max_iter : 500;
    double tol = solver ? solver->cg_tol : 1e-6;
    for (int iter = 0; iter < max_iter; iter++) {
        /* Ap = Phi^T*Phi*p + lambda*p */
        rls_matrix_vector_mul(tmp, Phi, p);
        rls_matrix_t_vector_mul(Ap, Phi, tmp);
        rls_vector_axpy(Ap, lambda, p);
        double alpha = rsold / rls_vector_dot(p, Ap);
        /* theta += alpha * p */
        RLSVector tv; tv.dim = n; tv.capacity = 0; tv.data = est->theta;
        rls_vector_axpy(&tv, alpha, p);
        /* r -= alpha * Ap */
        rls_vector_axpy(r, -alpha, Ap);
        double rsnew = rls_vector_dot(r, r);
        if (sqrt(rsnew) < tol) {
            est->iterations = iter + 1;
            break;
        }
        /* p = r + (rsnew/rsold) * p */
        double beta = rsnew / rsold;
        rls_vector_scal(p, beta);
        rls_vector_axpy(p, 1.0, r);
        rsold = rsnew;
        est->iterations = iter + 1;
    }
    /* Compute loss */
    RLSVector tv; tv.dim = n; tv.capacity = 0; tv.data = est->theta;
    rls_compute_residual(tmp, y, Phi, &tv);
    double rss = 0.0;
    for (int i = 0; i < m; i++) rss += tmp->data[i] * tmp->data[i];
    est->loss = 0.5 * rss + rls_penalty_ridge(&tv, lambda);
    est->converged = true;
    if (opt && opt->compute_stats)
        rls_estimate_compute_stats(est, Phi, y, rss / m, lambda);
    rls_vector_free(r); rls_vector_free(p); rls_vector_free(Ap); rls_vector_free(tmp);
    return est;
}

RLSEstimate *rls_solve_ridge(const RLSMatrix *Phi, const RLSVector *y,
                              double lambda, const RLSOptions *opt) {
    if (!Phi || !y) return NULL;
    int n = Phi->rows, p = Phi->cols;
    /* Auto-select solver */
    if (p <= 200 && n >= p)
        return rls_solve_ridge_cholesky(Phi, y, lambda, opt);
    else if (n < p)
        return rls_solve_ridge_svd(Phi, y, lambda, opt);
    else
        return rls_solve_ridge_qr(Phi, y, lambda, opt);
}

/* ============================================================================
 * LASSO via Coordinate Descent
 * ============================================================================ */

/** LASSO via Coordinate Descent (Friedman, Hastie, Tibshirani 2010).
 *  Minimize (1/(2n))*||y - Phi*theta||_2^2 + lambda*||theta||_1
 *
 *  For each coordinate j:
 *    r = y - Phi*theta  (current residuals)
 *    rho_j = Phi_j^T r + theta_j   (if columns are standardized)
 *    theta_j = S(rho_j, lambda) / (Phi_j^T Phi_j)
 *
 *  Active set cycling accelerates convergence.
 *  Theory: The objective is convex but not differentiable.
 *    Coordinate descent converges to a global minimizer (Tseng 2001). */
RLSEstimate *rls_solve_lasso_cd(const RLSMatrix *Phi, const RLSVector *y,
                                 double lambda, const RLSOptions *opt,
                                 const RLSSolverConfig *solver) {
    if (!Phi || !y) return NULL;
    int n = Phi->rows, p = Phi->cols;
    RLSEstimate *est = rls_estimate_alloc(p);
    if (!est) return NULL;
    /* Precompute column norms */
    double *col_norms = (double *)calloc(p, sizeof(double));
    for (int j = 0; j < p; j++) {
        double *col = &Phi->data[j * n];
        double s = 0.0;
        for (int i = 0; i < n; i++) s += col[i] * col[i];
        col_norms[j] = (s > 1e-15) ? s : 1.0;
    }
    /* Initialize residuals: r = y (since theta = 0) */
    RLSVector *r = rls_vector_copy(y);
    double inv_n = 1.0 / n;
    double lam = lambda;
    int max_iter = solver ? solver->cd_max_iter : 10000;
    double tol = solver ? solver->cd_tol : 1e-4;
    for (int iter = 0; iter < max_iter; iter++) {
        double max_update = 0.0;
        for (int j = 0; j < p; j++) {
            double *col = &Phi->data[j * n];
            /* rho_j = (1/n)*Phi_j^T r + theta_j */
            double rho = 0.0;
            for (int i = 0; i < n; i++) rho += col[i] * r->data[i];
            rho = rho * inv_n + est->theta[j];
            double old_theta = est->theta[j];
            double new_theta = rls_soft_threshold(rho, lam) / (col_norms[j] * inv_n + 1e-15);
            est->theta[j] = new_theta;
            double update = new_theta - old_theta;
            if (fabs(update) > 1e-15) {
                /* Update residuals: r = r - update * Phi_j */
                for (int i = 0; i < n; i++)
                    r->data[i] -= update * col[i];
            }
            if (fabs(update) > max_update) max_update = fabs(update);
        }
        if (max_update < tol) {
            est->iterations = iter + 1;
            est->converged = true;
            break;
        }
        est->iterations = iter + 1;
    }
    /* Compute loss */
    RLSVector tv; tv.dim = p; tv.capacity = 0; tv.data = est->theta;
    RLSVector *res = rls_vector_alloc(n);
    rls_compute_residual(res, y, Phi, &tv);
    double rss = 0.0;
    for (int i = 0; i < n; i++) rss += res->data[i] * res->data[i];
    est->loss = 0.5 * inv_n * rss + rls_penalty_lasso(&tv, lam);
    est->mse = rss * inv_n;
    if (opt && opt->compute_stats)
        rls_estimate_compute_stats(est, Phi, y, rss / n, lam);
    rls_vector_free(r); rls_vector_free(res); free(col_norms);
    return est;
}

/* ============================================================================
 * LASSO via ADMM
 *
 * Reformulate: minimize (1/2)*||y - Phi*theta||_2^2 + lambda*||z||_1
 *              subject to theta = z
 * ADMM iterations:
 *   theta^{k+1} = argmin (1/2)*||y - Phi*theta||^2 + (rho/2)*||theta - z^k + u^k||^2
 *   z^{k+1}     = S(theta^{k+1} + u^k, lambda/rho)
 *   u^{k+1}     = u^k + theta^{k+1} - z^{k+1}
 * ============================================================================ */

RLSEstimate *rls_solve_lasso_admm(const RLSMatrix *Phi, const RLSVector *y,
                                   double lambda, const RLSOptions *opt,
                                   const RLSSolverConfig *solver) {
    if (!Phi || !y) return NULL;
    int n = Phi->rows, p = Phi->cols;
    RLSEstimate *est = rls_estimate_alloc(p);
    if (!est) return NULL;
    double rho = solver ? solver->admm_rho : 1.0;
    int max_iter = solver ? solver->admm_max_iter : 5000;
    double tol = solver ? solver->admm_tol : 1e-4;
    /* Pre-factor (Phi^T Phi + rho*I) via Cholesky for efficient theta update */
    RLSMatrix *H = rls_matrix_alloc(p, p);
    rls_gram_matrix(H, Phi);
    rls_matrix_add_diag(H, rho);
    if (rls_cholesky_decompose(H) != 0) {
        rls_matrix_free(H);
        return rls_solve_lasso_cd(Phi, y, lambda, opt, solver);
    }
    RLSVector *z = rls_vector_alloc(p);
    RLSVector *u = rls_vector_alloc(p);
    RLSVector *rhs = rls_vector_alloc(p);
    RLSVector *z_old = rls_vector_alloc(p);
    for (int iter = 0; iter < max_iter; iter++) {
        /* theta update: (Phi^T Phi + rho I) theta = Phi^T y + rho*(z - u) */
        rls_matrix_t_vector_mul(rhs, Phi, y);
        for (int j = 0; j < p; j++)
            rhs->data[j] += rho * (z->data[j] - u->data[j]);
        RLSVector tv; tv.dim = p; tv.capacity = 0; tv.data = est->theta;
        rls_cholesky_solve(&tv, H, rhs);
        /* z update: soft-thresholding */
        rls_vector_copy_to(z_old, z);
        for (int j = 0; j < p; j++)
            z->data[j] = rls_soft_threshold(est->theta[j] + u->data[j], lambda / rho);
        /* u update */
        for (int j = 0; j < p; j++)
            u->data[j] += est->theta[j] - z->data[j];
        /* Convergence check */
        double primal_res = 0.0, dual_res = 0.0;
        for (int j = 0; j < p; j++) {
            double d = est->theta[j] - z->data[j];
            primal_res += d * d;
            double dz = z->data[j] - z_old->data[j];
            dual_res += dz * dz;
        }
        primal_res = sqrt(primal_res);
        dual_res = rho * sqrt(dual_res);
        if (primal_res < tol && dual_res < tol) {
            est->iterations = iter + 1;
            est->converged = true;
            break;
        }
        est->iterations = iter + 1;
    }
    /* Compute loss */
    RLSVector tv; tv.dim = p; tv.capacity = 0; tv.data = est->theta;
    RLSVector *res = rls_vector_alloc(n);
    rls_compute_residual(res, y, Phi, &tv);
    double rss = 0.0;
    for (int i = 0; i < n; i++) rss += res->data[i] * res->data[i];
    est->loss = 0.5 * rss + rls_penalty_lasso(&tv, lambda);
    est->mse = rss / n;
    if (opt && opt->compute_stats)
        rls_estimate_compute_stats(est, Phi, y, rss / n, lambda);
    rls_matrix_free(H); rls_vector_free(z); rls_vector_free(u);
    rls_vector_free(rhs); rls_vector_free(z_old); rls_vector_free(res);
    return est;
}

RLSEstimate *rls_solve_lasso(const RLSMatrix *Phi, const RLSVector *y,
                              double lambda, const RLSOptions *opt) {
    if (!Phi || !y) return NULL;
    RLSSolverConfig cfg = rls_solver_config_default(RLS_SOLVER_CD);
    /* For large problems, use CD; for moderate problems, ADMM can be faster */
    if (Phi->cols > 500)
        return rls_solve_lasso_cd(Phi, y, lambda, opt, &cfg);
    else
        return rls_solve_lasso_admm(Phi, y, lambda, opt, &cfg);
}

/* ============================================================================
 * Elastic Net via Coordinate Descent
 *
 * Minimize (1/(2n))*||y - Phi*theta||_2^2 + lambda*(alpha*||theta||_1 + (1-alpha)*||theta||_2^2/2)
 *
 * Coordinate update:
 *   theta_j = S(rho_j, lambda*alpha) / (||Phi_j||^2/n + lambda*(1-alpha))
 * ============================================================================ */

RLSEstimate *rls_solve_elasticnet_cd(const RLSMatrix *Phi, const RLSVector *y,
                                      double alpha, double lambda,
                                      const RLSOptions *opt,
                                      const RLSSolverConfig *solver) {
    if (!Phi || !y) return NULL;
    int n = Phi->rows, p = Phi->cols;
    RLSEstimate *est = rls_estimate_alloc(p);
    if (!est) return NULL;
    double *col_norms = (double *)calloc(p, sizeof(double));
    for (int j = 0; j < p; j++) {
        double *col = &Phi->data[j * n];
        double s = 0.0;
        for (int i = 0; i < n; i++) s += col[i] * col[i];
        col_norms[j] = s;
    }
    RLSVector *r = rls_vector_copy(y);
    double inv_n = 1.0 / n;
    double lam_l1 = lambda * alpha;
    double lam_l2 = lambda * (1.0 - alpha);
    int max_iter = solver ? solver->cd_max_iter : 10000;
    double tol = solver ? solver->cd_tol : 1e-4;
    for (int iter = 0; iter < max_iter; iter++) {
        double max_update = 0.0;
        for (int j = 0; j < p; j++) {
            double *col = &Phi->data[j * n];
            double rho = 0.0;
            for (int i = 0; i < n; i++) rho += col[i] * r->data[i];
            rho = rho * inv_n + est->theta[j] * col_norms[j] * inv_n;
            double denom = col_norms[j] * inv_n + lam_l2;
            double old_theta = est->theta[j];
            est->theta[j] = rls_soft_threshold(rho, lam_l1) / denom;
            double update = est->theta[j] - old_theta;
            if (fabs(update) > 1e-15)
                for (int i = 0; i < n; i++) r->data[i] -= update * col[i];
            if (fabs(update) > max_update) max_update = fabs(update);
        }
        if (max_update < tol) { est->iterations = iter + 1; est->converged = true; break; }
        est->iterations = iter + 1;
    }
    RLSVector tv; tv.dim = p; tv.capacity = 0; tv.data = est->theta;
    RLSVector *res = rls_vector_alloc(n);
    rls_compute_residual(res, y, Phi, &tv);
    double rss = 0.0;
    for (int i = 0; i < n; i++) rss += res->data[i] * res->data[i];
    est->loss = 0.5 * inv_n * rss + rls_penalty_elasticnet(&tv, alpha, lambda);
    est->mse = rss * inv_n;
    if (opt && opt->compute_stats)
        rls_estimate_compute_stats(est, Phi, y, rss / n, lambda);
    rls_vector_free(r); rls_vector_free(res); free(col_norms);
    return est;
}

/* ============================================================================
 * Group LASSO via Block Coordinate Descent
 * ============================================================================ */

RLSEstimate *rls_solve_group_lasso(const RLSMatrix *Phi, const RLSVector *y,
                                    const RLSRegularizer *reg,
                                    const RLSOptions *opt,
                                    const RLSSolverConfig *solver) {
    if (!Phi || !y || !reg) return NULL;
    int n = Phi->rows, p = Phi->cols;
    RLSEstimate *est = rls_estimate_alloc(p);
    if (!est) return NULL;
    RLSVector *r = rls_vector_copy(y);
    int max_iter = solver ? solver->cd_max_iter : 5000;
    double tol = solver ? solver->cd_tol : 1e-4;
    for (int iter = 0; iter < max_iter; iter++) {
        double max_update = 0.0;
        int offset = 0;
        for (int g = 0; g < reg->group_count; g++) {
            int gs = reg->group_sizes[g];
            /* Compute block gradient */
            RLSVector *sg = rls_vector_alloc(gs);
            for (int j = 0; j < gs; j++) {
                int col_idx = offset + j;
                double *col = &Phi->data[col_idx * n];
                double s = 0.0;
                for (int i = 0; i < n; i++) s += col[i] * r->data[i];
                sg->data[j] = s + est->theta[col_idx];
            }
            double nrm_sg = rls_vector_nrm2(sg);
            double shrink = (nrm_sg > reg->lambda) ? (1.0 - reg->lambda / nrm_sg) : 0.0;
            for (int j = 0; j < gs; j++) {
                int col_idx = offset + j;
                double old = est->theta[col_idx];
                est->theta[col_idx] = sg->data[j] * shrink;
                double upd = est->theta[col_idx] - old;
                if (fabs(upd) > 1e-15) {
                    double *col = &Phi->data[col_idx * n];
                    for (int i = 0; i < n; i++) r->data[i] -= upd * col[i];
                }
                if (fabs(upd) > max_update) max_update = fabs(upd);
            }
            rls_vector_free(sg);
            offset += gs;
        }
        if (max_update < tol) { est->iterations = iter + 1; est->converged = true; break; }
        est->iterations = iter + 1;
    }
    RLSVector tv; tv.dim = p; tv.capacity = 0; tv.data = est->theta;
    RLSVector *res = rls_vector_alloc(n);
    rls_compute_residual(res, y, Phi, &tv);
    double rss = 0.0;
    for (int i = 0; i < n; i++) rss += res->data[i] * res->data[i];
    est->loss = 0.5 * rss + rls_penalty_group_lasso(&tv, reg);
    est->mse = rss / n;
    est->converged = true;
    rls_vector_free(r); rls_vector_free(res);
    return est;
}

/* ============================================================================
 * Fused LASSO via ADMM
 * ============================================================================ */

RLSEstimate *rls_solve_fused_lasso(const RLSMatrix *Phi, const RLSVector *y,
                                    const RLSRegularizer *reg,
                                    const RLSOptions *opt,
                                    const RLSSolverConfig *solver) {
    if (!Phi || !y || !reg || !reg->D) return NULL;
    int n = Phi->rows, p = Phi->cols;
    RLSEstimate *est = rls_estimate_alloc(p);
    if (!est) return NULL;
    /* Pre-factor for theta update */
    RLSMatrix *H = rls_matrix_alloc(p, p);
    rls_gram_matrix(H, Phi);
    double rho = solver ? solver->admm_rho : 1.0;
    /* Add rho*D^T*D for the TV coupling */
    RLSMatrix *DtD = rls_matrix_alloc(p, p);
    rls_matrix_multiply(DtD, reg->D, reg->D);
    for (int i = 0; i < p * p; i++) DtD->data[i] *= rho;
    for (int i = 0; i < p; i++) H->data[i * p + i] += DtD->data[i * p + i] + rho;
    if (rls_cholesky_decompose(H) != 0) {
        rls_matrix_free(H); rls_matrix_free(DtD);
        est->loss = INFINITY; return est;
    }
    /* ADMM variables */
    int mD = reg->D->rows;
    RLSVector *z1 = rls_vector_alloc(p);   /* z1 = theta (L1) */
    RLSVector *u1 = rls_vector_alloc(p);
    RLSVector *z2 = rls_vector_alloc(mD);  /* z2 = D*theta (L1 TV) */
    RLSVector *u2 = rls_vector_alloc(mD);
    RLSVector *rhs = rls_vector_alloc(p);
    int max_iter = solver ? solver->admm_max_iter : 5000;
    double tol = solver ? solver->admm_tol : 1e-4;
    for (int iter = 0; iter < max_iter; iter++) {
        /* theta update */
        rls_matrix_t_vector_mul(rhs, Phi, y);
        for (int j = 0; j < p; j++)
            rhs->data[j] += rho * (z1->data[j] - u1->data[j]);
        RLSVector *Dtz2u2 = rls_vector_alloc(p);
        rls_matrix_t_vector_mul(Dtz2u2, reg->D, z2);
        for (int j = 0; j < p; j++) Dtz2u2->data[j] -= u2->data[j];
        rls_vector_axpy(rhs, rho, Dtz2u2);
        RLSVector tv; tv.dim = p; tv.capacity = 0; tv.data = est->theta;
        rls_cholesky_solve(&tv, H, rhs);
        rls_vector_free(Dtz2u2);
        /* z1 update (L1 prox) */
        for (int j = 0; j < p; j++)
            z1->data[j] = rls_soft_threshold(est->theta[j] + u1->data[j], reg->lambda / rho);
        /* z2 update (L1 prox on D*theta + u2) */
        RLSVector *Dtheta = rls_vector_alloc(mD);
        rls_matrix_vector_mul(Dtheta, reg->D, &tv);
        for (int j = 0; j < mD; j++)
            z2->data[j] = rls_soft_threshold(Dtheta->data[j] + u2->data[j], reg->lambda2 / rho);
        /* u updates */
        for (int j = 0; j < p; j++) u1->data[j] += est->theta[j] - z1->data[j];
        for (int j = 0; j < mD; j++) u2->data[j] += Dtheta->data[j] - z2->data[j];
        rls_vector_free(Dtheta);
        /* Convergence */
        double pr = 0.0;
        for (int j = 0; j < p; j++) { double d = est->theta[j] - z1->data[j]; pr += d * d; }
        if (sqrt(pr) < tol) { est->iterations = iter + 1; est->converged = true; break; }
        est->iterations = iter + 1;
    }
    RLSVector tv; tv.dim = p; tv.capacity = 0; tv.data = est->theta;
    RLSVector *res = rls_vector_alloc(n);
    rls_compute_residual(res, y, Phi, &tv);
    double rss = 0.0;
    for (int i = 0; i < n; i++) rss += res->data[i] * res->data[i];
    est->loss = 0.5 * rss + rls_penalty_fused(&tv, reg);
    est->mse = rss / n;
    rls_matrix_free(H); rls_matrix_free(DtD);
    rls_vector_free(z1); rls_vector_free(u1);
    rls_vector_free(z2); rls_vector_free(u2);
    rls_vector_free(rhs); rls_vector_free(res);
    return est;
}

/* ============================================================================
 * LSQR (Sparse iterative least squares)
 * ============================================================================ */

RLSEstimate *rls_solve_lsqr(const RLSMatrix *Phi, const RLSVector *y,
                             double lambda, const RLSOptions *opt,
                             const RLSSolverConfig *solver) {
    if (!Phi || !y) return NULL;
    /* LSQR with damping: solves [Phi; sqrt(lambda)*I] theta = [y; 0]
       using Golub-Kahan bidiagonalization.
       Simplified implementation using conjugate gradient on normal equations
       with the augmented system formulation. */
    return rls_solve_ridge_cg(Phi, y, lambda, opt, solver);
}

/* ============================================================================
 * Solution Path
 * ============================================================================ */

RLSEstimate **rls_solution_path(const RLSMatrix *Phi, const RLSVector *y,
                                 const double *lambdas, int n_lambdas,
                                 RLSRegType reg_type, double alpha,
                                 const RLSOptions *opt) {
    if (!Phi || !y || !lambdas || n_lambdas <= 0) return NULL;
    RLSEstimate **path = (RLSEstimate **)calloc(n_lambdas, sizeof(RLSEstimate *));
    if (!path) return NULL;
    RLSOptions myopt = opt ? *opt : rls_options_default();
    myopt.compute_stats = false;
    for (int i = 0; i < n_lambdas; i++) {
        switch (reg_type) {
            case RLS_REG_RIDGE:
                path[i] = rls_solve_ridge(Phi, y, lambdas[i], &myopt);
                break;
            case RLS_REG_LASSO: {
                RLSSolverConfig cfg = rls_solver_config_default(RLS_SOLVER_CD);
                if (i > 0 && path[i-1] && path[i-1]->converged) {
                    /* Warm start: copy previous solution as initial theta */
                    path[i] = rls_estimate_alloc(Phi->cols);
                    memcpy(path[i]->theta, path[i-1]->theta, Phi->cols * sizeof(double));
                    /* Re-solve with warm start (simplified: just re-solve) */
                    rls_estimate_free(path[i]);
                    path[i] = rls_solve_lasso_cd(Phi, y, lambdas[i], &myopt, &cfg);
                } else {
                    path[i] = rls_solve_lasso_cd(Phi, y, lambdas[i], &myopt, &cfg);
                }
                break;
            }
            case RLS_REG_ELASTICNET: {
                RLSSolverConfig cfg = rls_solver_config_default(RLS_SOLVER_CD);
                path[i] = rls_solve_elasticnet_cd(Phi, y, alpha, lambdas[i], &myopt, &cfg);
                break;
            }
            default:
                path[i] = rls_solve_ridge(Phi, y, lambdas[i], &myopt);
                break;
        }
    }
    return path;
}

double *rls_lambda_path(double lambda_max, double lambda_min, int n_lambdas) {
    if (n_lambdas <= 0) return NULL;
    double *path = (double *)malloc(n_lambdas * sizeof(double));
    if (!path) return NULL;
    if (n_lambdas == 1) { path[0] = lambda_max; return path; }
    double log_min = log(lambda_min);
    double log_max = log(lambda_max);
    double step = (log_max - log_min) / (n_lambdas - 1);
    for (int i = 0; i < n_lambdas; i++)
        path[i] = exp(log_max - i * step);
    return path;
}

double rls_lambda_max_lasso(const RLSMatrix *Phi, const RLSVector *y, int n_samples) {
    if (!Phi || !y) return 0.0;
    /* lambda_max = max_j |Phi_j^T y| / n */
    double max_abs_cor = 0.0;
    for (int j = 0; j < Phi->cols; j++) {
        double *col = &Phi->data[j * Phi->rows];
        double cor = 0.0;
        for (int i = 0; i < Phi->rows; i++) cor += col[i] * y->data[i];
        if (fabs(cor) > max_abs_cor) max_abs_cor = fabs(cor);
    }
    return max_abs_cor / (double)n_samples;
}
