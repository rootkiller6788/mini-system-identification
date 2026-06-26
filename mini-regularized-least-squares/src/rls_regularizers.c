#include "rls_core.h"
#include "rls_regularizers.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================================
 * Penalty Evaluation
 * ============================================================================ */

double rls_penalty_eval(const RLSVector *theta, const RLSRegularizer *reg) {
    if (!theta || !reg) return 0.0;
    switch (reg->type) {
        case RLS_REG_RIDGE:       return rls_penalty_ridge(theta, reg->lambda);
        case RLS_REG_LASSO:       return rls_penalty_lasso(theta, reg->lambda);
        case RLS_REG_ELASTICNET:  return rls_penalty_elasticnet(theta, reg->alpha, reg->lambda);
        case RLS_REG_GROUP_LASSO: return rls_penalty_group_lasso(theta, reg);
        case RLS_REG_FUSED:       return rls_penalty_fused(theta, reg);
        default: return 0.0;
    }
}

double rls_penalty_ridge(const RLSVector *theta, double lambda) {
    if (!theta) return 0.0;
    double s = 0.0;
    for (int i = 0; i < theta->dim; i++) s += theta->data[i] * theta->data[i];
    return 0.5 * lambda * s;
}

double rls_penalty_lasso(const RLSVector *theta, double lambda) {
    if (!theta) return 0.0;
    double s = 0.0;
    for (int i = 0; i < theta->dim; i++) s += fabs(theta->data[i]);
    return lambda * s;
}

double rls_penalty_elasticnet(const RLSVector *theta, double alpha, double lambda) {
    if (!theta) return 0.0;
    double l1 = 0.0, l2 = 0.0;
    for (int i = 0; i < theta->dim; i++) {
        double ti = theta->data[i];
        l1 += fabs(ti);
        l2 += ti * ti;
    }
    return lambda * (alpha * l1 + 0.5 * (1.0 - alpha) * l2);
}

double rls_penalty_nuclear(const RLSMatrix *Theta, double lambda) {
    if (!Theta) return 0.0;
    int m = Theta->rows, n = Theta->cols;
    int k = (m < n) ? m : n;
    RLSMatrix *U = rls_matrix_alloc(m, k);
    RLSVector *S = rls_vector_alloc(k);
    RLSMatrix *Vt = rls_matrix_alloc(k, n);
    if (rls_svd_decompose(U, S, Vt, Theta) != 0) {
        rls_matrix_free(U); rls_vector_free(S); rls_matrix_free(Vt);
        return INFINITY;
    }
    double sum_s = 0.0;
    for (int i = 0; i < k; i++) sum_s += S->data[i];
    rls_matrix_free(U); rls_vector_free(S); rls_matrix_free(Vt);
    return lambda * sum_s;
}

double rls_penalty_group_lasso(const RLSVector *theta, const RLSRegularizer *reg) {
    if (!theta || !reg) return 0.0;
    double penalty = 0.0;
    int offset = 0;
    for (int g = 0; g < reg->group_count; g++) {
        int gs = reg->group_sizes[g];
        double nrm = 0.0;
        for (int j = 0; j < gs; j++) {
            double tj = theta->data[offset + j];
            nrm += tj * tj;
        }
        penalty += sqrt(nrm);
        offset += gs;
    }
    return reg->lambda * penalty;
}

double rls_penalty_fused(const RLSVector *theta, const RLSRegularizer *reg) {
    if (!theta || !reg || !reg->D) return 0.0;
    double l1 = rls_penalty_lasso(theta, reg->lambda);
    /* TV penalty: lambda2 * ||D*theta||_1 */
    RLSVector *Dtheta = rls_vector_alloc(reg->D->rows);
    rls_matrix_vector_mul(Dtheta, reg->D, theta);
    double tv = 0.0;
    for (int i = 0; i < Dtheta->dim; i++) tv += fabs(Dtheta->data[i]);
    rls_vector_free(Dtheta);
    return l1 + reg->lambda2 * tv;
}

/* ============================================================================
 * Gradient / Subgradient
 * ============================================================================ */

void rls_gradient_ridge(RLSVector *grad, const RLSVector *theta, double lambda) {
    if (!grad || !theta) return;
    for (int i = 0; i < theta->dim; i++)
        grad->data[i] = lambda * theta->data[i];
}

void rls_subgradient_lasso(RLSVector *subg, const RLSVector *theta, double lambda) {
    if (!subg || !theta) return;
    for (int i = 0; i < theta->dim; i++) {
        if (theta->data[i] > 1e-15)
            subg->data[i] = lambda;
        else if (theta->data[i] < -1e-15)
            subg->data[i] = -lambda;
        else
            subg->data[i] = 0.0; /* Any value in [-lambda, lambda]; choose 0 */
    }
}

void rls_gradient_elasticnet(RLSVector *grad, const RLSVector *theta,
                              double alpha, double lambda) {
    if (!grad || !theta) return;
    for (int i = 0; i < theta->dim; i++) {
        grad->data[i] = lambda * ((1.0 - alpha) * theta->data[i]);
        /* L1 subgradient component added by caller */
    }
}

/* ============================================================================
 * Proximal Operators
 * ============================================================================ */

double rls_soft_threshold(double x, double lambda) {
    if (x > lambda) return x - lambda;
    if (x < -lambda) return x + lambda;
    return 0.0;
}

void rls_prox_lasso(RLSVector *out, const RLSVector *theta, double lambda) {
    if (!out || !theta) return;
    for (int i = 0; i < theta->dim; i++)
        out->data[i] = rls_soft_threshold(theta->data[i], lambda);
}

void rls_prox_ridge(RLSVector *out, const RLSVector *theta, double lambda) {
    if (!out || !theta) return;
    double denom = 1.0 + lambda;
    for (int i = 0; i < theta->dim; i++)
        out->data[i] = theta->data[i] / denom;
}

void rls_prox_elasticnet(RLSVector *out, const RLSVector *theta,
                          double alpha, double lambda) {
    if (!out || !theta) return;
    double l1 = lambda * alpha;
    double l2_denom = 1.0 + lambda * (1.0 - alpha);
    for (int i = 0; i < theta->dim; i++) {
        double x = theta->data[i] / l2_denom;
        out->data[i] = rls_soft_threshold(x, l1 / l2_denom);
    }
}

void rls_prox_group_lasso(RLSVector *out, const RLSVector *theta,
                           const RLSRegularizer *reg) {
    if (!out || !theta || !reg) return;
    int offset = 0;
    for (int g = 0; g < reg->group_count; g++) {
        int gs = reg->group_sizes[g];
        double nrm = 0.0;
        for (int j = 0; j < gs; j++)
            nrm += theta->data[offset + j] * theta->data[offset + j];
        nrm = sqrt(nrm);
        double shrink = (nrm > reg->lambda) ? (1.0 - reg->lambda / nrm) : 0.0;
        for (int j = 0; j < gs; j++)
            out->data[offset + j] = theta->data[offset + j] * shrink;
        offset += gs;
    }
}

void rls_prox_nuclear(RLSMatrix *out, const RLSMatrix *Theta, double lambda) {
    if (!out || !Theta) return;
    int m = Theta->rows, n = Theta->cols, k = (m < n) ? m : n;
    RLSMatrix *U = rls_matrix_alloc(m, k);
    RLSVector *S = rls_vector_alloc(k);
    RLSMatrix *Vt = rls_matrix_alloc(k, n);
    if (rls_svd_decompose(U, S, Vt, Theta) != 0) {
        rls_matrix_free(U); rls_vector_free(S); rls_matrix_free(Vt);
        return;
    }
    /* Soft-threshold singular values */
    for (int i = 0; i < k; i++)
        S->data[i] = rls_soft_threshold(S->data[i], lambda);
    /* Reconstruct: out = U * diag(S) * Vt */
    rls_matrix_zero(out);
    for (int l = 0; l < k; l++) {
        if (S->data[l] <= 0.0) continue;
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                out->data[j * m + i] += U->data[l * m + i] * S->data[l] *
                                         Vt->data[j * k + l];
    }
    rls_matrix_free(U); rls_vector_free(S); rls_matrix_free(Vt);
}

void rls_prox_fused(RLSVector *out, const RLSVector *theta,
                     const RLSRegularizer *reg, double rho) {
    if (!out || !theta || !reg) return;
    /* Simplified: apply soft-thresholding in D-transform domain.
       Full ADMM would iterate the fused lasso prox. */
    int n = theta->dim;
    RLSVector *w = rls_vector_alloc(n);
    rls_vector_copy_to(w, theta);
    /* Iterative soft-thresholding with TV penalty */
    for (int iter = 0; iter < 100; iter++) {
        /* Gradient step on data fidelity */
        /* Prox of TV via taut-string algorithm would go here.
           This simplified version applies per-coordinate soft-thresholding
           with neighbor differences penalized. */
        for (int i = 0; i < n; i++) {
            double x = w->data[i];
            double grad_tv = 0.0;
            if (i > 0) grad_tv += (x - w->data[i-1]);
            if (i < n - 1) grad_tv += (x - w->data[i+1]);
            w->data[i] = rls_soft_threshold(x - 0.01 * grad_tv * reg->lambda2,
                                             reg->lambda * 0.01);
        }
    }
    rls_vector_copy_to(out, w);
    rls_vector_free(w);
}

/* ============================================================================
 * Effective Degrees of Freedom
 * ============================================================================ */

double rls_df_ridge(const RLSMatrix *Phi, double lambda) {
    return rls_effective_df(Phi, lambda);
}

double rls_df_lasso(const RLSMatrix *Phi, const RLSVector *theta_hat) {
    /* df = number of non-zero coefficients (Zou, Hastie, Tibshirani 2007) */
    if (!theta_hat) return 0.0;
    int nnz = 0;
    for (int i = 0; i < theta_hat->dim; i++)
        if (fabs(theta_hat->data[i]) > 1e-12) nnz++;
    return (double)nnz;
}

double rls_df_elasticnet(const RLSMatrix *Phi, const RLSVector *theta_hat,
                          double alpha, double lambda) {
    if (!Phi || !theta_hat) return 0.0;
    /* df = tr(Phi_A * (Phi_A^T Phi_A + lambda*(1-alpha)*I)^{-1} * Phi_A^T)
       where Phi_A selects active columns. Simplified: use active set + ridge formula. */
    int p = theta_hat->dim;
    int nnz = 0;
    for (int i = 0; i < p; i++)
        if (fabs(theta_hat->data[i]) > 1e-12) nnz++;
    if (nnz == 0) return 0.0;
    /* Conservative estimate: between nnz and ridge df */
    double ridge_df = rls_effective_df(Phi, lambda * (1.0 - alpha));
    return 0.5 * ((double)nnz + ridge_df);
}

/* ============================================================================
 * Dual Norms & KKT Checks
 * ============================================================================ */

double rls_dual_norm_lasso(const RLSVector *v) {
    return rls_vector_norm_inf(v);
}

double rls_dual_norm_group_lasso(const RLSVector *v, const RLSRegularizer *reg) {
    if (!v || !reg) return 0.0;
    double max_nrm = 0.0;
    int offset = 0;
    for (int g = 0; g < reg->group_count; g++) {
        int gs = reg->group_sizes[g];
        double nrm = 0.0;
        for (int j = 0; j < gs; j++)
            nrm += v->data[offset + j] * v->data[offset + j];
        nrm = sqrt(nrm);
        if (nrm > max_nrm) max_nrm = nrm;
        offset += gs;
    }
    return max_nrm;
}

bool rls_kkt_check_lasso(const RLSVector *theta, const RLSVector *grad_loss,
                          double lambda, double tol) {
    if (!theta || !grad_loss) return false;
    for (int i = 0; i < theta->dim; i++) {
        double gi = grad_loss->data[i];
        double ti = theta->data[i];
        if (ti > tol) {
            if (fabs(gi + lambda) > tol) return false;
        } else if (ti < -tol) {
            if (fabs(gi - lambda) > tol) return false;
        } else {
            if (fabs(gi) > lambda + tol) return false;
        }
    }
    return true;
}

bool rls_kkt_check_elasticnet(const RLSVector *theta, const RLSVector *grad_loss,
                               double alpha, double lambda, double tol) {
    if (!theta || !grad_loss) return false;
    double l2_pen = lambda * (1.0 - alpha);
    for (int i = 0; i < theta->dim; i++) {
        double gi = grad_loss->data[i] + l2_pen * theta->data[i];
        double ti = theta->data[i];
        double l1 = lambda * alpha;
        if (ti > tol) {
            if (fabs(gi + l1) > tol) return false;
        } else if (ti < -tol) {
            if (fabs(gi - l1) > tol) return false;
        } else {
            if (fabs(gi) > l1 + tol) return false;
        }
    }
    return true;
}
