#include "rls_core.h"
#include "rls_validation.h"
#include "rls_solvers.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================================
 * K-Fold Cross-Validation (L5: Methods)
 *
 * Theory (L4): CV provides an approximately unbiased estimate of prediction
 * error. K=10 is standard ([Hastie, Tibshirani, Friedman 2009] ESL Chapter 7).
 * ============================================================================ */

int rls_kfold_cv(const RLSMatrix *Phi, const RLSVector *y,
                 RLSLambdaSelection *sel, RLSRegType reg_type,
                 double alpha, const RLSOptions *opt) {
    if (!Phi || !y || !sel) return -1;
    int n = Phi->rows, p = Phi->cols;
    int K = sel->k_folds;
    if (K < 2 || K > n) K = (n < 10) ? n : 10;
    if (!sel->lambda_grid) rls_generate_lambda_grid(sel);
    int nL = sel->n_lambda;
    if (!sel->cv_scores)
        sel->cv_scores = (double *)calloc(nL, sizeof(double));
    /* For each lambda, compute K-fold CV score */
    int fold_size = n / K;
    for (int li = 0; li < nL; li++) {
        double lam = sel->lambda_grid[li];
        double cv_sum = 0.0;
        int cv_count = 0;
        for (int k = 0; k < K; k++) {
            int val_start = k * fold_size;
            int val_end = (k == K - 1) ? n : val_start + fold_size;
            int val_n = val_end - val_start;
            int train_n = n - val_n;
            if (train_n < p || val_n < 1) continue;
            /* Build train/val matrices (simplified: copy rows) */
            RLSMatrix *Phi_train = rls_matrix_alloc(train_n, p);
            RLSVector *y_train = rls_vector_alloc(train_n);
            RLSMatrix *Phi_val = rls_matrix_alloc(val_n, p);
            RLSVector *y_val = rls_vector_alloc(val_n);
            int ti = 0, vi = 0;
            for (int i = 0; i < n; i++) {
                if (i >= val_start && i < val_end) {
                    for (int j = 0; j < p; j++)
                        Phi_val->data[j * val_n + vi] = Phi->data[j * n + i];
                    y_val->data[vi] = y->data[i];
                    vi++;
                } else {
                    for (int j = 0; j < p; j++)
                        Phi_train->data[j * train_n + ti] = Phi->data[j * n + i];
                    y_train->data[ti] = y->data[i];
                    ti++;
                }
            }
            double fold_err = rls_kfold_single_fold(Phi_train, y_train,
                                                      Phi_val, y_val, lam,
                                                      reg_type, alpha, opt);
            cv_sum += fold_err;
            cv_count++;
            rls_matrix_free(Phi_train); rls_vector_free(y_train);
            rls_matrix_free(Phi_val); rls_vector_free(y_val);
        }
        sel->cv_scores[li] = (cv_count > 0) ? cv_sum / cv_count : INFINITY;
    }
    /* Find optimal lambda */
    double best_score = INFINITY;
    int best_idx = 0;
    for (int li = 0; li < nL; li++) {
        if (sel->cv_scores[li] < best_score) {
            best_score = sel->cv_scores[li];
            best_idx = li;
        }
    }
    sel->lambda_opt = sel->lambda_grid[best_idx];
    sel->lambda_opt_score = best_score;
    return 0;
}

double rls_kfold_single_fold(const RLSMatrix *Phi_train, const RLSVector *y_train,
                              const RLSMatrix *Phi_val, const RLSVector *y_val,
                              double lambda, RLSRegType reg_type, double alpha,
                              const RLSOptions *opt) {
    RLSEstimate *est = NULL;
    RLSSolverConfig cfg = rls_solver_config_default(RLS_SOLVER_CD);
    switch (reg_type) {
        case RLS_REG_RIDGE:
            est = rls_solve_ridge(Phi_train, y_train, lambda, opt);
            break;
        case RLS_REG_LASSO:
            est = rls_solve_lasso_cd(Phi_train, y_train, lambda, opt, &cfg);
            break;
        case RLS_REG_ELASTICNET:
            est = rls_solve_elasticnet_cd(Phi_train, y_train, alpha, lambda, opt, &cfg);
            break;
        default:
            est = rls_solve_ridge(Phi_train, y_train, lambda, opt);
            break;
    }
    if (!est) return INFINITY;
    /* Validation error */
    RLSVector *y_pred = rls_vector_alloc(Phi_val->rows);
    RLSVector tv; tv.dim = est->n_params; tv.capacity = 0; tv.data = est->theta;
    rls_matrix_vector_mul(y_pred, Phi_val, &tv);
    double err = 0.0;
    for (int i = 0; i < Phi_val->rows; i++) {
        double d = y_val->data[i] - y_pred->data[i];
        err += d * d;
    }
    err /= Phi_val->rows;
    rls_estimate_free(est);
    rls_vector_free(y_pred);
    return err;
}

/* ============================================================================
 * Generalized Cross-Validation (GCV)
 *
 * GCV(lambda) = n * ||y - y_hat||^2 / (n - df(lambda))^2
 *
 * For ridge: df(lambda) = sum_i s_i^2 / (s_i^2 + lambda)
 * where s_i are singular values of Phi.
 * Ref: Golub, Heath, Wahba (1979)
 * ============================================================================ */

double rls_gcv_score(const RLSMatrix *Phi, const RLSVector *y,
                     double lambda, RLSRegType reg_type, double alpha,
                     const RLSOptions *opt) {
    if (!Phi || !y) return INFINITY;
    RLSOptions myopt = opt ? *opt : rls_options_default();
    myopt.compute_stats = false;
    RLSEstimate *est = rls_solve_ridge(Phi, y, lambda, &myopt);
    if (!est) return INFINITY;
    /* Compute RSS */
    RLSVector *y_hat = rls_vector_alloc(Phi->rows);
    RLSVector tv; tv.dim = Phi->cols; tv.capacity = 0; tv.data = est->theta;
    rls_matrix_vector_mul(y_hat, Phi, &tv);
    double rss = 0.0;
    for (int i = 0; i < Phi->rows; i++) {
        double d = y->data[i] - y_hat->data[i];
        rss += d * d;
    }
    int n = Phi->rows;
    double df = rls_effective_df(Phi, lambda);
    double gcv = (n * rss) / ((n - df) * (n - df) + 1e-15);
    rls_estimate_free(est);
    rls_vector_free(y_hat);
    return gcv;
}

int rls_gcv_optimize(const RLSMatrix *Phi, const RLSVector *y,
                     RLSLambdaSelection *sel, RLSRegType reg_type,
                     double alpha, const RLSOptions *opt) {
    if (!Phi || !y || !sel) return -1;
    if (!sel->lambda_grid) rls_generate_lambda_grid(sel);
    double best_gcv = INFINITY;
    int best_idx = 0;
    for (int li = 0; li < sel->n_lambda; li++) {
        double gcv = rls_gcv_score(Phi, y, sel->lambda_grid[li], reg_type, alpha, opt);
        if (gcv < best_gcv) { best_gcv = gcv; best_idx = li; }
    }
    sel->lambda_opt = sel->lambda_grid[best_idx];
    sel->lambda_opt_score = best_gcv;
    return 0;
}

/* ============================================================================
 * L-Curve Criterion
 *
 * Plots log||residual|| vs log||solution_norm|| for various lambda.
 * Optimal lambda at the point of maximum curvature.
 * Ref: Hansen (1992) "Analysis of Discrete Ill-Posed Problems"
 * ============================================================================ */

void rls_lcurve_compute(const RLSMatrix *Phi, const RLSVector *y,
                        const double *lambdas, int n_lambdas,
                        RLSRegType reg_type, double alpha,
                        const RLSOptions *opt,
                        double *log_rho, double *log_eta) {
    if (!Phi || !y || !lambdas || !log_rho || !log_eta) return;
    RLSOptions myopt = opt ? *opt : rls_options_default();
    myopt.compute_stats = false;
    for (int li = 0; li < n_lambdas; li++) {
        RLSEstimate *est = rls_solve_ridge(Phi, y, lambdas[li], &myopt);
        if (!est) { log_rho[li] = 0.0; log_eta[li] = 0.0; continue; }
        RLSVector *res = rls_vector_alloc(Phi->rows);
        RLSVector tv; tv.dim = Phi->cols; tv.capacity = 0; tv.data = est->theta;
        rls_compute_residual(res, y, Phi, &tv);
        double rho = rls_vector_nrm2(res);
        double eta = 0.0;
        for (int j = 0; j < est->n_params; j++) eta += est->theta[j] * est->theta[j];
        eta = sqrt(eta);
        log_rho[li] = log(rho + 1e-15);
        log_eta[li] = log(eta + 1e-15);
        rls_estimate_free(est);
        rls_vector_free(res);
    }
}

double rls_lcurve_corner(const double *log_rho, const double *log_eta,
                          int n, const double *lambdas) {
    if (n < 4) return lambdas[n/2];
    /* Curvature via central differences */
    double max_curv = -1.0;
    int best_idx = 0;
    for (int i = 1; i < n - 1; i++) {
        double rho_p = (log_rho[i+1] - log_rho[i-1]) / 2.0;
        double eta_p = (log_eta[i+1] - log_eta[i-1]) / 2.0;
        double rho_pp = log_rho[i+1] - 2.0 * log_rho[i] + log_rho[i-1];
        double eta_pp = log_eta[i+1] - 2.0 * log_eta[i] + log_eta[i-1];
        double denom = pow(rho_p * rho_p + eta_p * eta_p, 1.5);
        if (denom < 1e-15) continue;
        double curv = fabs(rho_p * eta_pp - rho_pp * eta_p) / denom;
        if (curv > max_curv) { max_curv = curv; best_idx = i; }
    }
    return lambdas[best_idx];
}

int rls_lcurve_optimize(const RLSMatrix *Phi, const RLSVector *y,
                         RLSLambdaSelection *sel, RLSRegType reg_type,
                         double alpha, const RLSOptions *opt) {
    if (!Phi || !y || !sel) return -1;
    if (!sel->lambda_grid) rls_generate_lambda_grid(sel);
    int nL = sel->n_lambda;
    double *log_rho = (double *)malloc(nL * sizeof(double));
    double *log_eta = (double *)malloc(nL * sizeof(double));
    rls_lcurve_compute(Phi, y, sel->lambda_grid, nL, reg_type, alpha, opt,
                        log_rho, log_eta);
    sel->lambda_opt = rls_lcurve_corner(log_rho, log_eta, nL, sel->lambda_grid);
    sel->lambda_opt_score = 0.0;
    free(log_rho); free(log_eta);
    return 0;
}

/* ============================================================================
 * Information Criteria: AICc and BIC
 * ============================================================================ */

double rls_aicc_score(const RLSMatrix *Phi, const RLSVector *y,
                      double lambda, RLSRegType reg_type, double alpha,
                      const RLSOptions *opt) {
    if (!Phi || !y) return INFINITY;
    RLSOptions myopt = opt ? *opt : rls_options_default();
    myopt.compute_stats = true;
    RLSEstimate *est = rls_solve_ridge(Phi, y, lambda, &myopt);
    if (!est) return INFINITY;
    double aicc = est->aic;
    rls_estimate_free(est);
    return aicc;
}

double rls_bic_score(const RLSMatrix *Phi, const RLSVector *y,
                     double lambda, RLSRegType reg_type, double alpha,
                     const RLSOptions *opt) {
    if (!Phi || !y) return INFINITY;
    RLSOptions myopt = opt ? *opt : rls_options_default();
    myopt.compute_stats = true;
    RLSEstimate *est = rls_solve_ridge(Phi, y, lambda, &myopt);
    if (!est) return INFINITY;
    double bic = est->bic;
    rls_estimate_free(est);
    return bic;
}

int rls_aicc_optimize(const RLSMatrix *Phi, const RLSVector *y,
                      RLSLambdaSelection *sel, RLSRegType reg_type,
                      double alpha, const RLSOptions *opt) {
    if (!Phi || !y || !sel) return -1;
    if (!sel->lambda_grid) rls_generate_lambda_grid(sel);
    double best = INFINITY;
    int best_idx = 0;
    for (int li = 0; li < sel->n_lambda; li++) {
        double aicc = rls_aicc_score(Phi, y, sel->lambda_grid[li], reg_type, alpha, opt);
        if (aicc < best) { best = aicc; best_idx = li; }
    }
    sel->lambda_opt = sel->lambda_grid[best_idx];
    sel->lambda_opt_score = best;
    return 0;
}

/* ============================================================================
 * SURE (Stein's Unbiased Risk Estimate)
 *
 * SURE(lambda) = ||y - y_hat||^2 + 2*sigma^2*df(lambda) - n*sigma^2
 * Requires knowledge of noise variance sigma^2.
 * Ref: Stein (1981) "Estimation of the Mean of a Multivariate Normal Distribution"
 * ============================================================================ */

double rls_sure_score(const RLSMatrix *Phi, const RLSVector *y,
                      double lambda, double sigma2, const RLSOptions *opt) {
    if (!Phi || !y) return INFINITY;
    RLSOptions myopt = opt ? *opt : rls_options_default();
    myopt.compute_stats = false;
    RLSEstimate *est = rls_solve_ridge(Phi, y, lambda, &myopt);
    if (!est) return INFINITY;
    RLSVector *res = rls_vector_alloc(Phi->rows);
    RLSVector tv; tv.dim = Phi->cols; tv.capacity = 0; tv.data = est->theta;
    rls_compute_residual(res, y, Phi, &tv);
    double rss = 0.0;
    for (int i = 0; i < Phi->rows; i++) rss += res->data[i] * res->data[i];
    double df = rls_effective_df(Phi, lambda);
    double sure = rss + 2.0 * sigma2 * df - Phi->rows * sigma2;
    rls_estimate_free(est);
    rls_vector_free(res);
    return sure;
}

int rls_sure_optimize(const RLSMatrix *Phi, const RLSVector *y,
                      double sigma2, RLSLambdaSelection *sel,
                      const RLSOptions *opt) {
    if (!Phi || !y || !sel) return -1;
    if (!sel->lambda_grid) rls_generate_lambda_grid(sel);
    double best = INFINITY;
    int best_idx = 0;
    for (int li = 0; li < sel->n_lambda; li++) {
        double sure = rls_sure_score(Phi, y, sel->lambda_grid[li], sigma2, opt);
        if (sure < best) { best = sure; best_idx = li; }
    }
    sel->lambda_opt = sel->lambda_grid[best_idx];
    sel->lambda_opt_score = best;
    return 0;
}

/* ============================================================================
 * Empirical Bayes / Marginal Likelihood
 *
 * log p(y|lambda) = -0.5 * y^T * (K + lambda*I)^{-1} * y - 0.5 * log|K+lambda*I| - n/2*log(2*pi)
 * where K = Phi * Phi^T.
 * Maximizing this yields the optimal lambda under a Gaussian prior theta ~ N(0, lambda^{-1} I).
 * ============================================================================ */

double rls_marginal_likelihood(const RLSMatrix *Phi, const RLSVector *y,
                                double lambda) {
    if (!Phi || !y || lambda <= 0) return -INFINITY;
    int n = Phi->rows, p = Phi->cols;
    /* For n > p: use Woodbury to compute y^T*(K+lambda*I)^{-1}*y efficiently */
    if (n > p) {
        /* (Phi*Phi^T + lambda*I)^{-1} = lambda^{-1}*I - lambda^{-2}*Phi*(I + lambda^{-1}*Phi^T*Phi)^{-1}*Phi^T */
        RLSMatrix *XtX = rls_matrix_alloc(p, p);
        rls_gram_matrix(XtX, Phi);
        for (int i = 0; i < p; i++) XtX->data[i*p+i] = XtX->data[i*p+i]/lambda + 1.0;
        /* Cholesky of I + lambda^{-1}*Phi^T*Phi */
        if (rls_cholesky_decompose(XtX) != 0) { rls_matrix_free(XtX); return -INFINITY; }
        RLSVector *Phity = rls_vector_alloc(p);
        rls_matrix_t_vector_mul(Phity, Phi, y);
        RLSVector *z = rls_vector_alloc(p);
        rls_cholesky_solve(z, XtX, Phity);
        double y_Sinv_y = (rls_vector_dot(y, y) - rls_vector_dot(Phity, z)) / lambda;
        /* log det(K + lambda*I) = (n-p)*log(lambda) + sum log(di) where di are eigenvalues of I + lambda^{-1}*XtX */
        double logdet = (n - p) * log(lambda);
        for (int i = 0; i < p; i++) {
            double di = XtX->data[i*p+i];
            logdet += 2.0 * log(di);
        }
        logdet += p * log(lambda);
        rls_matrix_free(XtX); rls_vector_free(Phity); rls_vector_free(z);
        return -0.5 * (y_Sinv_y + logdet + n * log(2.0 * M_PI));
    } else {
        /* Direct Cholesky on K + lambda*I (n x n) */
        RLSMatrix *K = rls_matrix_alloc(n, n);
        RLSMatrix *PhiT = rls_matrix_alloc(p, n);
        rls_matrix_transpose(PhiT, Phi);
        rls_matrix_multiply(K, Phi, PhiT);
        rls_matrix_add_diag(K, lambda);
        if (rls_cholesky_decompose(K) != 0) {
            rls_matrix_free(K); rls_matrix_free(PhiT);
            return -INFINITY;
        }
        RLSVector *z = rls_vector_alloc(n);
        rls_cholesky_solve(z, K, y);
        double y_Sinv_y = rls_vector_dot(y, z);
        double logdet = 0.0;
        for (int i = 0; i < n; i++) logdet += 2.0 * log(K->data[i*n+i]);
        rls_matrix_free(K); rls_matrix_free(PhiT); rls_vector_free(z);
        return -0.5 * (y_Sinv_y + logdet + n * log(2.0 * M_PI));
    }
}

int rls_empirical_bayes_optimize(const RLSMatrix *Phi, const RLSVector *y,
                                  RLSLambdaSelection *sel,
                                  const RLSOptions *opt) {
    if (!Phi || !y || !sel) return -1;
    if (!sel->lambda_grid) rls_generate_lambda_grid(sel);
    double best_ml = -INFINITY;
    int best_idx = 0;
    for (int li = 0; li < sel->n_lambda; li++) {
        double ml = rls_marginal_likelihood(Phi, y, sel->lambda_grid[li]);
        if (ml > best_ml) { best_ml = ml; best_idx = li; }
    }
    sel->lambda_opt = sel->lambda_grid[best_idx];
    sel->lambda_opt_score = best_ml;
    return 0;
}

/* ============================================================================
 * Unified Lambda Selection
 * ============================================================================ */

int rls_select_lambda(const RLSMatrix *Phi, const RLSVector *y,
                      RLSLambdaSelection *sel, RLSRegType reg_type,
                      double alpha, const RLSOptions *opt) {
    if (!Phi || !y || !sel) return -1;
    switch (sel->method) {
        case RLS_LAMBDA_FIXED:  return 0;
        case RLS_LAMBDA_GCV:    return rls_gcv_optimize(Phi, y, sel, reg_type, alpha, opt);
        case RLS_LAMBDA_LCURVE: return rls_lcurve_optimize(Phi, y, sel, reg_type, alpha, opt);
        case RLS_LAMBDA_KFOLD_CV: return rls_kfold_cv(Phi, y, sel, reg_type, alpha, opt);
        case RLS_LAMBDA_AICC:   return rls_aicc_optimize(Phi, y, sel, reg_type, alpha, opt);
        case RLS_LAMBDA_ML:     return rls_empirical_bayes_optimize(Phi, y, sel, opt);
        case RLS_LAMBDA_STEIN: {
            double sigma2 = 1.0; /* Need estimate; use RSS/(n-p) from OLS */
            RLSOptions myopt = opt ? *opt : rls_options_default();
            RLSEstimate *ols = rls_solve_ridge(Phi, y, 0.0, &myopt);
            if (ols) { sigma2 = ols->mse; rls_estimate_free(ols); }
            return rls_sure_optimize(Phi, y, sigma2, sel, opt);
        }
        default: return -1;
    }
}

void rls_generate_lambda_grid(RLSLambdaSelection *sel) {
    if (!sel || sel->n_lambda <= 0) return;
    if (sel->lambda_grid) free(sel->lambda_grid);
    sel->lambda_grid = rls_lambda_path(sel->lambda_max, sel->lambda_min, sel->n_lambda);
}
