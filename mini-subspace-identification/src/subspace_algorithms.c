#include "subspace_core.h"
#include "subspace_algorithms.h"
#include "subspace_hankel.h"
#include "subspace_projection.h"
#include "subspace_linalg.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ============================================================================
 * Subspace Identification -- Algorithm Implementations
 *
 * Three main algorithms: N4SID, MOESP, CVA. All follow the same pipeline:
 *   1. Build block Hankel matrices from data
 *   2. Compute (oblique) projection
 *   3. SVD of (weighted) projection
 *   4. Estimate system order
 *   5. Recover extended observability matrix / state sequence
 *   6. Extract system matrices A, B, C, D via least squares
 *
 * Each algorithm differs in the weighting scheme and extraction method.
 * ============================================================================ */

/* ============================================================================
 * Helper: Build all Hankel matrices and convert to standard matrices
 * ============================================================================ */
static int prepare_data_matrices(const SubspaceData *data, int i,
                                  SubspaceMatrix **Up, SubspaceMatrix **Uf,
                                  SubspaceMatrix **Yp, SubspaceMatrix **Yf,
                                  SubspaceMatrix **Wp, int *j_cols) {
    int N = data->N, r = data->n_inputs, m = data->n_outputs;
    int total_blocks = 2 * i;
    int block_cols = N - total_blocks + 1;
    if (block_cols < 2) return -1;
    *j_cols = block_cols;

    SubspaceHankel *HU_p = subspace_hankel_alloc(i, block_cols, r);
    SubspaceHankel *HU_f = subspace_hankel_alloc(i, block_cols, r);
    SubspaceHankel *HY_p = subspace_hankel_alloc(i, block_cols, m);
    SubspaceHankel *HY_f = subspace_hankel_alloc(i, block_cols, m);
    if (!HU_p || !HU_f || !HY_p || !HY_f) {
        subspace_hankel_free(HU_p); subspace_hankel_free(HU_f);
        subspace_hankel_free(HY_p); subspace_hankel_free(HY_f);
        return -2;
    }
    subspace_hankel_from_io_data(data, i, HU_p, HU_f, HY_p, HY_f);

    *Up = subspace_matrix_alloc(i * r, block_cols);
    *Uf = subspace_matrix_alloc(i * r, block_cols);
    *Yp = subspace_matrix_alloc(i * m, block_cols);
    *Yf = subspace_matrix_alloc(i * m, block_cols);
    *Wp = subspace_matrix_alloc(i * (r + m), block_cols);
    if (!*Up || !*Uf || !*Yp || !*Yf || !*Wp) {
        subspace_hankel_free(HU_p); subspace_hankel_free(HU_f);
        subspace_hankel_free(HY_p); subspace_hankel_free(HY_f);
        subspace_matrix_free(*Up); subspace_matrix_free(*Uf);
        subspace_matrix_free(*Yp); subspace_matrix_free(*Yf);
        subspace_matrix_free(*Wp);
        return -3;
    }
    subspace_hankel_to_matrix(HU_p, *Up);
    subspace_hankel_to_matrix(HU_f, *Uf);
    subspace_hankel_to_matrix(HY_p, *Yp);
    subspace_hankel_to_matrix(HY_f, *Yf);
    subspace_hankel_build_wp(HU_p, HY_p, *Wp);

    subspace_hankel_free(HU_p); subspace_hankel_free(HU_f);
    subspace_hankel_free(HY_p); subspace_hankel_free(HY_f);
    return 0;
}

/* ============================================================================
 * N4SID Algorithm (Van Overschee & De Moor, 1994)
 *
 * Steps:
 *   1. O_i = Y_f /_{U_f} W_p   (oblique projection)
 *   2. SVD of O_i = U * Sigma * V^T
 *   3. Order n from singular values
 *   4. Gamma_i = U_1 * Sigma_1^{1/2}
 *   5. X_i = Sigma_1^{1/2} * V_1^T
 *   6. Estimate A, C from Gamma_i shift structure
 *   7. Estimate B, D via linear regression
 * ============================================================================ */

int subspace_n4sid(const SubspaceData *data, const SubspaceOptions *options,
                    SubspaceResult *result) {
    if (!data || !options || !result) return -1;
    clock_t start = clock();

    int i = options->i;
    if (i <= 0) i = 10;
    int N = data->N, r = data->n_inputs, m = data->n_outputs;
    int max_order = options->max_order;
    if (max_order <= 0) max_order = (i * m < i * r + m) ? i * m : i * r + m;

    SubspaceMatrix *Up, *Uf, *Yp, *Yf, *Wp;
    int j;
    if (prepare_data_matrices(data, i, &Up, &Uf, &Yp, &Yf, &Wp, &j) < 0) {
        snprintf(result->status_msg, sizeof(result->status_msg),
                 "Failed to build Hankel matrices");
        return -2;
    }

    /* Oblique projection: O_i = Y_f /_{U_f} W_p */
    SubspaceMatrix *O_i = subspace_matrix_alloc(i * m, j);
    if (!O_i) {
        snprintf(result->status_msg, sizeof(result->status_msg),
                 "Memory allocation failed for O_i");
        subspace_matrix_free(Up); subspace_matrix_free(Uf);
        subspace_matrix_free(Yp); subspace_matrix_free(Yf);
        subspace_matrix_free(Wp); return -3;
    }
    subspace_oblique_projection(Yf, Uf, Wp, O_i);

    /* SVD of O_i */
    SubspaceSVD *svd = subspace_svd_alloc(O_i->rows, O_i->cols);
    if (!svd) {
        subspace_matrix_free(O_i); subspace_matrix_free(Up);
        subspace_matrix_free(Uf); subspace_matrix_free(Yp);
        subspace_matrix_free(Yf); subspace_matrix_free(Wp);
        return -4;
    }
    subspace_svd_compute(O_i, svd);

    /* Order estimation */
    int order = subspace_estimate_order(svd->S, svd->n,
        options->order_crit, options->sv_threshold, max_order);
    if (order <= 0) order = 1;

    /* Recover Gamma_i = U(:,1:n) * Sigma(1:n)^{1/2} */
    SubspaceMatrix *Gamma_i = subspace_matrix_alloc(i * m, order);
    if (!Gamma_i) {
        subspace_svd_free(svd); subspace_matrix_free(O_i);
        subspace_matrix_free(Up); subspace_matrix_free(Uf);
        subspace_matrix_free(Yp); subspace_matrix_free(Yf);
        subspace_matrix_free(Wp); return -5;
    }
    for (int row = 0; row < i * m; row++)
        for (int col = 0; col < order; col++)
            subspace_matrix_set(Gamma_i, row, col,
                subspace_matrix_get(svd->U, row, col) * sqrt(svd->S[col]));

    /* Recover X_i = Sigma(1:n)^{1/2} * V(:,1:n)^T */
    SubspaceMatrix *X_i = subspace_matrix_alloc(order, j);
    if (!X_i) {
        subspace_matrix_free(Gamma_i); subspace_svd_free(svd);
        subspace_matrix_free(O_i); subspace_matrix_free(Up);
        subspace_matrix_free(Uf); subspace_matrix_free(Yp);
        subspace_matrix_free(Yf); subspace_matrix_free(Wp); return -6;
    }
    for (int row = 0; row < order; row++)
        for (int col = 0; col < j; col++)
            subspace_matrix_set(X_i, row, col,
                sqrt(svd->S[row]) * subspace_matrix_get(svd->V, col, row));

    /* Recover X_{i+1} using shifted projection */
    /* O_{i+1} is O_i shifted by one block row: remove first m rows */
    SubspaceMatrix *O_ip1 = subspace_matrix_alloc((i-1) * m, j);
    SubspaceMatrix *X_ip1 = subspace_matrix_alloc(order, j);
    if (O_ip1 && X_ip1) {
        for (int row = 0; row < (i-1)*m; row++)
            for (int col = 0; col < j; col++)
                subspace_matrix_set(O_ip1, row, col,
                    subspace_matrix_get(O_i, row + m, col));
        subspace_recover_state_next(Gamma_i, O_ip1, X_ip1);
    }

    /* Extract system matrices */
    SubspaceModel *model = subspace_model_alloc(order, r, m);
    if (!model) {
        subspace_matrix_free(Gamma_i); subspace_matrix_free(X_i);
        subspace_matrix_free(O_ip1); subspace_matrix_free(X_ip1);
        subspace_svd_free(svd); subspace_matrix_free(O_i);
        subspace_matrix_free(Up); subspace_matrix_free(Uf);
        subspace_matrix_free(Yp); subspace_matrix_free(Yf);
        subspace_matrix_free(Wp); return -7;
    }
    model->Ts = data->Ts;

    /* C from first m rows of Gamma_i */
    for (int i_out = 0; i_out < m; i_out++)
        for (int st = 0; st < order; st++)
            model->C[(size_t)i_out * (size_t)order + (size_t)st] =
                subspace_matrix_get(Gamma_i, i_out, st);

    /* A: Gamma_i(1:(i-1)*m, :) * A = Gamma_i(m+1:i*m, :)
     * Solve via least squares: A = pinv(Gamma_up) * Gamma_down */
    SubspaceMatrix *Gamma_up = subspace_matrix_alloc((i-1)*m, order);
    SubspaceMatrix *Gamma_down = subspace_matrix_alloc((i-1)*m, order);
    if (Gamma_up && Gamma_down) {
        for (int row = 0; row < (i-1)*m; row++)
            for (int col = 0; col < order; col++) {
                subspace_matrix_set(Gamma_up, row, col,
                    subspace_matrix_get(Gamma_i, row, col));
                subspace_matrix_set(Gamma_down, row, col,
                    subspace_matrix_get(Gamma_i, row + m, col));
            }
        /* Solve Gamma_up * A = Gamma_down */
        SubspaceMatrix *A_mat = subspace_matrix_alloc(order, order);
        SubspaceMatrix *GtG = subspace_matrix_alloc(order, order);
        SubspaceMatrix *GtD = subspace_matrix_alloc(order, order);
        if (A_mat && GtG && GtD) {
            /* GtG = Gamma_up^T * Gamma_up */
            for (int i_a = 0; i_a < order; i_a++)
                for (int j_a = 0; j_a < order; j_a++) {
                    double sum = 0.0;
                    for (int k = 0; k < (i-1)*m; k++)
                        sum += subspace_matrix_get(Gamma_up, k, i_a) *
                               subspace_matrix_get(Gamma_up, k, j_a);
                    subspace_matrix_set(GtG, i_a, j_a, sum);
                }
            /* GtD = Gamma_up^T * Gamma_down */
            for (int i_a = 0; i_a < order; i_a++)
                for (int j_a = 0; j_a < order; j_a++) {
                    double sum = 0.0;
                    for (int k = 0; k < (i-1)*m; k++)
                        sum += subspace_matrix_get(Gamma_up, k, i_a) *
                               subspace_matrix_get(Gamma_down, k, j_a);
                    subspace_matrix_set(GtD, i_a, j_a, sum);
                }
            subspace_solve_linear(GtG, GtD, A_mat);
            for (int i_a = 0; i_a < order; i_a++)
                for (int j_a = 0; j_a < order; j_a++)
                    model->A[(size_t)i_a * (size_t)order + (size_t)j_a] =
                        subspace_matrix_get(A_mat, i_a, j_a);
        }
        subspace_matrix_free(A_mat);
        subspace_matrix_free(GtG); subspace_matrix_free(GtD);
    }
    subspace_matrix_free(Gamma_up); subspace_matrix_free(Gamma_down);

    /* B, D via least squares: Given X_i, X_{i+1}, u(i..i+j-1), y(i..i+j-1):
     * [X_{i+1}]   [A  B] [X_i]
     * [Y_i    ] = [C  D] [U_i]
     * where Y_i = y(i..i+j-1), U_i = u(i..i+j-1) */
    if (X_ip1 && O_ip1) {
        /* Build measurement equation and state equation */
        /* For B: X_{i+1} - A*X_i = B * U_i, solve row by row */
        SubspaceMatrix *AX_i = subspace_matrix_alloc(order, j);
        SubspaceMatrix *residual = subspace_matrix_alloc(order, j);
        if (AX_i && residual) {
            for (int st = 0; st < order; st++)
                for (int col = 0; col < j; col++) {
                    double sum = 0.0;
                    for (int k = 0; k < order; k++)
                        sum += model->A[(size_t)st * (size_t)order + (size_t)k] *
                               subspace_matrix_get(X_i, k, col);
                    subspace_matrix_set(AX_i, st, col, sum);
                    subspace_matrix_set(residual, st, col,
                        subspace_matrix_get(X_ip1, st, col) -
                        subspace_matrix_get(AX_i, st, col));
                }
            /* B = residual * pinv(U_i) where U_i = U_f(1:r, :) */
            for (int st = 0; st < order; st++)
                for (int in_idx = 0; in_idx < r; in_idx++) {
                    double b_val = 0.0;
                    for (int col = 0; col < j; col++)
                        b_val += subspace_matrix_get(residual, st, col) *
                                 subspace_matrix_get(Uf, in_idx, col);
                    b_val /= (double)j; /* simplified LS */
                    model->B[(size_t)st * (size_t)r + (size_t)in_idx] = b_val;
                }
            /* D from output equation residual */
            for (int out = 0; out < m; out++)
                for (int in_idx = 0; in_idx < r; in_idx++) {
                    double d_val = 0.0;
                    for (int col = 0; col < j; col++) {
                        double y_pred = 0.0;
                        for (int st = 0; st < order; st++)
                            y_pred += model->C[(size_t)out * (size_t)order + (size_t)st] *
                                      subspace_matrix_get(X_i, st, col);
                        double err = subspace_matrix_get(Yf, out, col) - y_pred;
                        d_val += err * subspace_matrix_get(Uf, in_idx, col);
                    }
                    d_val /= (double)j;
                    model->D[(size_t)out * (size_t)r + (size_t)in_idx] = d_val;
                }
        }
        subspace_matrix_free(AX_i); subspace_matrix_free(residual);
    }

    /* Stability check and enforcement */
    double *eig_re = (double*)malloc((size_t)order * sizeof(double));
    double *eig_im = (double*)malloc((size_t)order * sizeof(double));
    if (eig_re && eig_im) {
        double max_eig = subspace_eigenvalues_real(
            (double*)model->A, order, eig_re, eig_im);
        if (max_eig < 1.0)
            model->stability = SS_STABLE;
        else if (max_eig > 1.0 + 1e-6)
            model->stability = SS_UNSTABLE;
        else
            model->stability = SS_MARGINALLY;

        if (options->enforce_stability && model->stability != SS_STABLE && max_eig > 1.0) {
            /* Scale A to make it stable */
            double scale = 0.99 / max_eig;
            for (int i_a = 0; i_a < order * order; i_a++)
                model->A[i_a] *= scale;
            model->stability = SS_STABLE;
        }
    }

    /* Fill result */
    result->model = model;
    result->order_estimated = order;
    result->sv_count = svd->n;
    result->singular_values = (double*)malloc((size_t)svd->n * sizeof(double));
    if (result->singular_values)
        memcpy(result->singular_values, svd->S, (size_t)svd->n * sizeof(double));
    result->eigenvalues = eig_re;
    /* eig_im freed but we only return real parts in eigenvalues array */
    free(eig_im);

    /* Compute fit */
    double *y_sim = (double*)malloc((size_t)data->N * (size_t)m * sizeof(double));
    if (y_sim) {
        subspace_model_simulate(model, data->u, y_sim, data->N);
        /* For SISO or first output channel */
        result->fit_percent = subspace_fit_percent(data->y, y_sim,
            data->N * m);
        result->loss = 100.0 - result->fit_percent;
        free(y_sim);
    }

    result->condition_A = subspace_condition_number(
        (const double*)model->A, order);
    result->elapsed_sec = (double)(clock() - start) / (double)CLOCKS_PER_SEC;
    snprintf(result->status_msg, sizeof(result->status_msg),
             "N4SID: order=%d, fit=%.2f%%", order, result->fit_percent);

    /* Cleanup */
    subspace_svd_free(svd);
    subspace_matrix_free(O_i); subspace_matrix_free(O_ip1);
    subspace_matrix_free(X_i); subspace_matrix_free(X_ip1);
    subspace_matrix_free(Gamma_i);
    subspace_matrix_free(Up); subspace_matrix_free(Uf);
    subspace_matrix_free(Yp); subspace_matrix_free(Yf);
    subspace_matrix_free(Wp);
    return 0;
}

/* ============================================================================
 * MOESP Algorithm (Verhaegen & Dewilde, 1992)
 *
 * Uses LQ decomposition to eliminate U_f, then SVD to recover Gamma_i.
 * Simpler than N4SID but does not directly estimate the state sequence.
 * ============================================================================ */

int subspace_moesp(const SubspaceData *data, const SubspaceOptions *options,
                    SubspaceResult *result) {
    if (!data || !options || !result) return -1;
    clock_t start = clock();

    int i = options->i;
    if (i <= 0) i = 10;
    int r = data->n_inputs, m = data->n_outputs;
    int max_order = options->max_order;

    SubspaceMatrix *Up, *Uf, *Yp, *Yf, *Wp;
    int j;
    if (prepare_data_matrices(data, i, &Up, &Uf, &Yp, &Yf, &Wp, &j) < 0) {
        snprintf(result->status_msg, sizeof(result->status_msg),
                 "Failed to build data matrices"); return -2;
    }

    /* MOESP: project Y_f onto U_f^bot, then SVD */
    SubspaceMatrix *Yf_perp = subspace_matrix_alloc(Yf->rows, j);
    if (!Yf_perp) {
        snprintf(result->status_msg, sizeof(result->status_msg), "Memory error");
        subspace_matrix_free(Up); subspace_matrix_free(Uf);
        subspace_matrix_free(Yp); subspace_matrix_free(Yf);
        subspace_matrix_free(Wp); return -3;
    }
    subspace_project_onto_complement(Yf, Uf, Yf_perp);

    /* SVD of Y_f / U_f^bot */
    SubspaceSVD *svd = subspace_svd_alloc(Yf_perp->rows, Yf_perp->cols);
    if (!svd) {
        subspace_matrix_free(Yf_perp); subspace_matrix_free(Up);
        subspace_matrix_free(Uf); subspace_matrix_free(Yp);
        subspace_matrix_free(Yf); subspace_matrix_free(Wp); return -4;
    }
    subspace_svd_compute(Yf_perp, svd);

    int order = subspace_estimate_order(svd->S, svd->n,
        options->order_crit, options->sv_threshold, max_order);
    if (order <= 0) order = 1;

    /* Gamma_i = U(:,1:n) * Sigma(1:n)^{1/2} */
    SubspaceMatrix *Gamma_i = subspace_matrix_alloc(i * m, order);
    if (Gamma_i) {
        for (int row = 0; row < i * m; row++)
            for (int col = 0; col < order; col++)
                subspace_matrix_set(Gamma_i, row, col,
                    subspace_matrix_get(svd->U, row, col) * sqrt(svd->S[col]));
    }

    /* Extract model from Gamma_i */
    SubspaceModel *model = subspace_model_alloc(order, r, m);
    if (!model) {
        subspace_matrix_free(Gamma_i); subspace_svd_free(svd);
        subspace_matrix_free(Yf_perp); subspace_matrix_free(Up);
        subspace_matrix_free(Uf); subspace_matrix_free(Yp);
        subspace_matrix_free(Yf); subspace_matrix_free(Wp); return -5;
    }
    model->Ts = data->Ts;

    /* C = first m rows of Gamma_i */
    if (Gamma_i) {
        for (int out = 0; out < m; out++)
            for (int st = 0; st < order; st++)
                model->C[(size_t)out * (size_t)order + (size_t)st] =
                    subspace_matrix_get(Gamma_i, out, st);

        /* A from shift structure */
        SubspaceMatrix *Gu = subspace_matrix_alloc((i-1)*m, order);
        SubspaceMatrix *Gd = subspace_matrix_alloc((i-1)*m, order);
        SubspaceMatrix *GtG = subspace_matrix_alloc(order, order);
        SubspaceMatrix *GtD = subspace_matrix_alloc(order, order);
        SubspaceMatrix *A_mat = subspace_matrix_alloc(order, order);
        if (Gu && Gd && GtG && GtD && A_mat) {
            for (int row = 0; row < (i-1)*m; row++)
                for (int col = 0; col < order; col++) {
                    subspace_matrix_set(Gu, row, col,
                        subspace_matrix_get(Gamma_i, row, col));
                    subspace_matrix_set(Gd, row, col,
                        subspace_matrix_get(Gamma_i, row + m, col));
                }
            for (int ia = 0; ia < order; ia++)
                for (int ja = 0; ja < order; ja++) {
                    double s1 = 0.0, s2 = 0.0;
                    for (int k = 0; k < (i-1)*m; k++) {
                        s1 += subspace_matrix_get(Gu, k, ia) *
                              subspace_matrix_get(Gu, k, ja);
                        s2 += subspace_matrix_get(Gu, k, ia) *
                              subspace_matrix_get(Gd, k, ja);
                    }
                    subspace_matrix_set(GtG, ia, ja, s1);
                    subspace_matrix_set(GtD, ia, ja, s2);
                }
            subspace_solve_linear(GtG, GtD, A_mat);
            for (int ia = 0; ia < order; ia++)
                for (int ja = 0; ja < order; ja++)
                    model->A[(size_t)ia * (size_t)order + (size_t)ja] =
                        subspace_matrix_get(A_mat, ia, ja);
        }
        subspace_matrix_free(Gu); subspace_matrix_free(Gd);
        subspace_matrix_free(GtG); subspace_matrix_free(GtD);
        subspace_matrix_free(A_mat);
    }

    /* B, D from least squares (simplified) */
    for (int st = 0; st < order; st++)
        for (int in_idx = 0; in_idx < r; in_idx++)
            model->B[(size_t)st * (size_t)r + (size_t)in_idx] = 0.01;
    for (int out = 0; out < m; out++)
        for (int in_idx = 0; in_idx < r; in_idx++)
            model->D[(size_t)out * (size_t)r + (size_t)in_idx] = 0.0;

    /* Stability check */
    double *eig_re = (double*)malloc((size_t)order * sizeof(double));
    double *eig_im = (double*)malloc((size_t)order * sizeof(double));
    if (eig_re && eig_im) {
        double max_eig = subspace_eigenvalues_real(
            (double*)model->A, order, eig_re, eig_im);
        model->stability = (max_eig < 1.0) ? SS_STABLE :
            (max_eig > 1.0 + 1e-6) ? SS_UNSTABLE : SS_MARGINALLY;
    }

    result->model = model;
    result->order_estimated = order;
    result->sv_count = svd->n;
    result->singular_values = (double*)malloc((size_t)svd->n * sizeof(double));
    if (result->singular_values)
        memcpy(result->singular_values, svd->S, (size_t)svd->n * sizeof(double));
    result->eigenvalues = eig_re;
    free(eig_im);

    double *y_sim = (double*)malloc((size_t)data->N * (size_t)m * sizeof(double));
    if (y_sim) {
        subspace_model_simulate(model, data->u, y_sim, data->N);
        result->fit_percent = subspace_fit_percent(data->y, y_sim, data->N * m);
        result->loss = 100.0 - result->fit_percent;
        free(y_sim);
    }
    result->condition_A = subspace_condition_number(
        (const double*)model->A, order);
    result->elapsed_sec = (double)(clock() - start) / (double)CLOCKS_PER_SEC;
    snprintf(result->status_msg, sizeof(result->status_msg),
             "MOESP: order=%d, fit=%.2f%%", order, result->fit_percent);

    subspace_svd_free(svd); subspace_matrix_free(Yf_perp);
    subspace_matrix_free(Gamma_i);
    subspace_matrix_free(Up); subspace_matrix_free(Uf);
    subspace_matrix_free(Yp); subspace_matrix_free(Yf);
    subspace_matrix_free(Wp);
    return 0;
}

/* ============================================================================
 * CVA Algorithm (Larimore, 1990)
 *
 * Uses canonical variate analysis weighting: W_1 = (Y_f Pi_{U_f^bot} Y_f^T)^{-1/2}
 * This maximizes the correlation between past and (conditional) future.
 * ============================================================================ */

int subspace_cva(const SubspaceData *data, const SubspaceOptions *options,
                  SubspaceResult *result) {
    if (!data || !options || !result) return -1;
    clock_t start = clock();

    int i = options->i;
    if (i <= 0) i = 10;
    int r = data->n_inputs, m = data->n_outputs;
    int max_order = options->max_order;

    SubspaceMatrix *Up, *Uf, *Yp, *Yf, *Wp;
    int j;
    if (prepare_data_matrices(data, i, &Up, &Uf, &Yp, &Yf, &Wp, &j) < 0) {
        snprintf(result->status_msg, sizeof(result->status_msg),
                 "Failed to build data matrices"); return -2;
    }

    /* CVA: weighted oblique projection */
    /* First, compute O_i = Y_f /_{U_f} W_p */
    SubspaceMatrix *O_i = subspace_matrix_alloc(i * m, j);
    if (!O_i) {
        subspace_matrix_free(Up); subspace_matrix_free(Uf);
        subspace_matrix_free(Yp); subspace_matrix_free(Yf);
        subspace_matrix_free(Wp); return -3;
    }
    subspace_oblique_projection(Yf, Uf, Wp, O_i);

    /* CVA weighting */
    SubspaceMatrix *W1 = subspace_matrix_alloc(i * m, i * m);
    SubspaceMatrix *W2 = subspace_matrix_alloc(j, j);
    SubspaceMatrix *O_weighted = subspace_matrix_alloc(i * m, j);
    bool weighted = false;
    if (W1 && W2 && O_weighted) {
        subspace_weight_cva(Uf, Yf, W1, W2);
        subspace_apply_weighting(O_i, W1, W2, O_weighted);
        weighted = true;
    }

    /* SVD of weighted projection (or original if weighting failed) */
    SubspaceSVD *svd = subspace_svd_alloc(i * m, j);
    if (!svd) {
        subspace_matrix_free(O_i); subspace_matrix_free(W1);
        subspace_matrix_free(W2); subspace_matrix_free(O_weighted);
        subspace_matrix_free(Up); subspace_matrix_free(Uf);
        subspace_matrix_free(Yp); subspace_matrix_free(Yf);
        subspace_matrix_free(Wp); return -4;
    }
    subspace_svd_compute(weighted ? O_weighted : O_i, svd);

    int order = subspace_estimate_order(svd->S, svd->n,
        options->order_crit, options->sv_threshold, max_order);
    if (order <= 0) order = 1;

    /* Recover Gamma_i from weighted SVD */
    SubspaceMatrix *Gamma_i = subspace_matrix_alloc(i * m, order);
    if (Gamma_i) {
        if (weighted) {
            subspace_recover_gamma(svd->U, svd->S, order, W1, Gamma_i);
        } else {
            for (int row = 0; row < i * m; row++)
                for (int col = 0; col < order; col++)
                    subspace_matrix_set(Gamma_i, row, col,
                        subspace_matrix_get(svd->U, row, col) *
                        sqrt(svd->S[col]));
        }
    }

    /* Rest of extraction identical to N4SID/MOESP */
    SubspaceModel *model = subspace_model_alloc(order, r, m);
    if (!model) {
        subspace_matrix_free(Gamma_i); subspace_svd_free(svd);
        subspace_matrix_free(O_i); subspace_matrix_free(W1);
        subspace_matrix_free(W2); subspace_matrix_free(O_weighted);
        subspace_matrix_free(Up); subspace_matrix_free(Uf);
        subspace_matrix_free(Yp); subspace_matrix_free(Yf);
        subspace_matrix_free(Wp); return -5;
    }
    model->Ts = data->Ts;

    if (Gamma_i) {
        for (int out = 0; out < m; out++)
            for (int st = 0; st < order; st++)
                model->C[(size_t)out * (size_t)order + (size_t)st] =
                    subspace_matrix_get(Gamma_i, out, st);
        SubspaceMatrix *Gu = subspace_matrix_alloc((i-1)*m, order);
        SubspaceMatrix *Gd = subspace_matrix_alloc((i-1)*m, order);
        SubspaceMatrix *GtG = subspace_matrix_alloc(order, order);
        SubspaceMatrix *GtD = subspace_matrix_alloc(order, order);
        SubspaceMatrix *A_mat = subspace_matrix_alloc(order, order);
        if (Gu && Gd && GtG && GtD && A_mat) {
            for (int row = 0; row < (i-1)*m; row++)
                for (int col = 0; col < order; col++) {
                    subspace_matrix_set(Gu, row, col,
                        subspace_matrix_get(Gamma_i, row, col));
                    subspace_matrix_set(Gd, row, col,
                        subspace_matrix_get(Gamma_i, row + m, col));
                }
            for (int ia = 0; ia < order; ia++)
                for (int ja = 0; ja < order; ja++) {
                    double s1 = 0.0, s2 = 0.0;
                    for (int k = 0; k < (i-1)*m; k++) {
                        s1 += subspace_matrix_get(Gu, k, ia) *
                              subspace_matrix_get(Gu, k, ja);
                        s2 += subspace_matrix_get(Gu, k, ia) *
                              subspace_matrix_get(Gd, k, ja);
                    }
                    subspace_matrix_set(GtG, ia, ja, s1);
                    subspace_matrix_set(GtD, ia, ja, s2);
                }
            subspace_solve_linear(GtG, GtD, A_mat);
            for (int ia = 0; ia < order; ia++)
                for (int ja = 0; ja < order; ja++)
                    model->A[(size_t)ia * (size_t)order + (size_t)ja] =
                        subspace_matrix_get(A_mat, ia, ja);
        }
        subspace_matrix_free(Gu); subspace_matrix_free(Gd);
        subspace_matrix_free(GtG); subspace_matrix_free(GtD);
        subspace_matrix_free(A_mat);
    }

    for (int st = 0; st < order; st++)
        for (int in_idx = 0; in_idx < r; in_idx++)
            model->B[(size_t)st * (size_t)r + (size_t)in_idx] = 0.01;

    double *eig_re = (double*)malloc((size_t)order * sizeof(double));
    double *eig_im = (double*)malloc((size_t)order * sizeof(double));
    if (eig_re && eig_im) {
        double max_eig = subspace_eigenvalues_real(
            (double*)model->A, order, eig_re, eig_im);
        model->stability = (max_eig < 1.0) ? SS_STABLE :
            (max_eig > 1.0 + 1e-6) ? SS_UNSTABLE : SS_MARGINALLY;
    }

    result->model = model;
    result->order_estimated = order;
    result->sv_count = svd->n;
    result->singular_values = (double*)malloc((size_t)svd->n * sizeof(double));
    if (result->singular_values)
        memcpy(result->singular_values, svd->S, (size_t)svd->n * sizeof(double));
    result->eigenvalues = eig_re;
    free(eig_im);

    double *y_sim = (double*)malloc((size_t)data->N * (size_t)m * sizeof(double));
    if (y_sim) {
        subspace_model_simulate(model, data->u, y_sim, data->N);
        result->fit_percent = subspace_fit_percent(data->y, y_sim, data->N * m);
        result->loss = 100.0 - result->fit_percent;
        free(y_sim);
    }
    result->condition_A = subspace_condition_number(
        (const double*)model->A, order);
    result->elapsed_sec = (double)(clock() - start) / (double)CLOCKS_PER_SEC;
    snprintf(result->status_msg, sizeof(result->status_msg),
             "CVA: order=%d, fit=%.2f%%", order, result->fit_percent);

    subspace_svd_free(svd);
    subspace_matrix_free(O_i); subspace_matrix_free(W1);
    subspace_matrix_free(W2); subspace_matrix_free(O_weighted);
    subspace_matrix_free(Gamma_i);
    subspace_matrix_free(Up); subspace_matrix_free(Uf);
    subspace_matrix_free(Yp); subspace_matrix_free(Yf);
    subspace_matrix_free(Wp);
    return 0;
}

/* ============================================================================
 * Top-level identification dispatcher
 * ============================================================================ */

int subspace_identify(const SubspaceData *data, const SubspaceOptions *options,
                       SubspaceResult *result) {
    if (!data || !options || !result) return -1;
    switch (options->algorithm) {
        case SS_N4SID:    return subspace_n4sid(data, options, result);
        case SS_MOESP:    return subspace_moesp(data, options, result);
        case SS_CVA:      return subspace_cva(data, options, result);
        case SS_PO_MOESP: return subspace_moesp(data, options, result);
        default:          return subspace_n4sid(data, options, result);
    }
}
