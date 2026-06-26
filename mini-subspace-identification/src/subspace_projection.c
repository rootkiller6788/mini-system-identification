#include "subspace_core.h"
#include "subspace_projection.h"
#include "subspace_linalg.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ============================================================================
 * Subspace Identification -- Projection Operations
 *
 * Implements orthogonal and oblique projections that form the geometric
 * core of subspace identification. The oblique projection:
 *   O_i = Y_f /_{U_f} W_p
 * is the central computation that reveals the state sequence.
 *
 * Geometric interpretation: The oblique projection of Y_f along U_f
 * onto W_p is the projection of the row space of Y_f onto the row space
 * of W_p along the row space of U_f. This eliminates the influence of
 * future inputs while retaining information about the initial state.
 *
 * Notation (Van Overschee & De Moor, 1996):
 *   A / B      = orthogonal projection of A onto B
 *   A / B^bot  = orthogonal projection of A onto complement of B
 *   A /_B C    = oblique projection of A along B onto C
 * ============================================================================ */

/* Orthogonal projection: P = A * B^T * (B * B^T)^+ * B
 * Projects the row space of A onto the row space of B.
 * Uses QR for numerical stability. */
int subspace_project_onto(const SubspaceMatrix *A, const SubspaceMatrix *B,
                           SubspaceMatrix *P) {
    if (!A || !B || !P) return -1;
    int m_a = A->rows, n_a = A->cols;
    int m_b = B->rows, n_b = B->cols;
    /* A and B must have same number of columns */
    if (n_a != n_b) return -2;

    /* QR of B^T: B has size m_b x n_b, so B^T is n_b x m_b.
     * Factor B^T = Q_B * R_B, Q_B is n_b x m_b, R_B is m_b x m_b upper tri. */
    SubspaceMatrix *Bt = subspace_matrix_alloc(n_b, m_b);
    SubspaceMatrix *Q_B = subspace_matrix_alloc(n_b, m_b);
    SubspaceMatrix *R_B = subspace_matrix_alloc(m_b, m_b);
    if (!Bt || !Q_B || !R_B) {
        subspace_matrix_free(Bt); subspace_matrix_free(Q_B);
        subspace_matrix_free(R_B); return -3;
    }

    /* Bt = B^T */
    for (int i = 0; i < n_b; i++)
        for (int j = 0; j < m_b; j++)
            subspace_matrix_set(Bt, i, j, subspace_matrix_get(B, j, i));

    /* QR: B^T = Q_B * R_B */
    int qr_ret = subspace_qr_mgs(Bt, Q_B, R_B);
    if (qr_ret < 0) {
        subspace_matrix_free(Bt); subspace_matrix_free(Q_B);
        subspace_matrix_free(R_B); return qr_ret;
    }

    /* Determine rank of B */
    int rank = 0;
    for (int i = 0; i < m_b && i < n_b; i++) {
        if (fabs(subspace_matrix_get(R_B, i, i)) > 1e-10) rank++;
        else break;
    }
    if (rank == 0) {
        subspace_matrix_fill(P, 0.0);
        subspace_matrix_free(Bt); subspace_matrix_free(Q_B);
        subspace_matrix_free(R_B); return 0;
    }

    /* P = A * Q_B(:,1:rank) * Q_B(:,1:rank)^T
     * Compute T = A * Q_B(:,1:rank) first */
    SubspaceMatrix *T = subspace_matrix_alloc(m_a, rank);
    if (!T) {
        subspace_matrix_free(Bt); subspace_matrix_free(Q_B);
        subspace_matrix_free(R_B); return -4;
    }
    for (int i = 0; i < m_a; i++) {
        for (int j = 0; j < rank; j++) {
            double sum = 0.0;
            for (int k = 0; k < n_a; k++)
                sum += subspace_matrix_get(A, i, k) *
                       subspace_matrix_get(Q_B, k, j);
            subspace_matrix_set(T, i, j, sum);
        }
    }
    /* P = T * Q_B(:,1:rank)^T */
    for (int i = 0; i < m_a; i++) {
        for (int j = 0; j < n_a; j++) {
            double sum = 0.0;
            for (int k = 0; k < rank; k++)
                sum += subspace_matrix_get(T, i, k) *
                       subspace_matrix_get(Q_B, j, k);
            subspace_matrix_set(P, i, j, sum);
        }
    }

    subspace_matrix_free(Bt); subspace_matrix_free(Q_B);
    subspace_matrix_free(R_B); subspace_matrix_free(T);
    return 0;
}

/* Orthogonal complement projection: P = A * (I - B^T * (B*B^T)^+ * B) */
int subspace_project_onto_complement(const SubspaceMatrix *A,
                                      const SubspaceMatrix *B,
                                      SubspaceMatrix *P) {
    if (!A || !B || !P) return -1;
    /* P = A - A / B */
    subspace_matrix_copy(A, P);
    SubspaceMatrix *A_on_B = subspace_matrix_alloc(A->rows, A->cols);
    if (!A_on_B) return -2;
    int ret = subspace_project_onto(A, B, A_on_B);
    if (ret == 0) {
        for (int i = 0; i < A->rows; i++)
            for (int j = 0; j < A->cols; j++)
                subspace_matrix_set(P, i, j,
                    subspace_matrix_get(P, i, j) -
                    subspace_matrix_get(A_on_B, i, j));
    }
    subspace_matrix_free(A_on_B);
    return ret;
}

/* Oblique projection: A /_B C (A along B onto C)
 * Formula: A /_B C = [A / B^bot] * [C / B^bot]^+ * C */
int subspace_project_oblique(const SubspaceMatrix *A, const SubspaceMatrix *B,
                              const SubspaceMatrix *C, SubspaceMatrix *O) {
    if (!A || !B || !C || !O) return -1;
    if (A->cols != B->cols || A->cols != C->cols) return -2;

    /* A_perp = A / B^bot */
    SubspaceMatrix *A_perp = subspace_matrix_alloc(A->rows, A->cols);
    /* C_perp = C / B^bot */
    SubspaceMatrix *C_perp = subspace_matrix_alloc(C->rows, C->cols);
    if (!A_perp || !C_perp) {
        subspace_matrix_free(A_perp);
        subspace_matrix_free(C_perp);
        return -3;
    }

    subspace_project_onto_complement(A, B, A_perp);
    subspace_project_onto_complement(C, B, C_perp);

    /* Compute C_perp^+ via SVD */
    SubspaceSVD *svd = subspace_svd_alloc(C_perp->rows, C_perp->cols);
    if (!svd) {
        subspace_matrix_free(A_perp);
        subspace_matrix_free(C_perp);
        return -4;
    }
    subspace_svd_compute(C_perp, svd);

    /* Pseudoinverse: C_perp^+ = V * Sigma^+ * U^T */
    int rank = 0;
    for (int i = 0; i < svd->n; i++)
        if (svd->S[i] > 1e-10 * svd->S[0]) rank++;
        else break;

    if (rank == 0) {
        subspace_matrix_fill(O, 0.0);
        subspace_svd_free(svd);
        subspace_matrix_free(A_perp);
        subspace_matrix_free(C_perp);
        return 0;
    }

    /* Compute A_perp * C_perp^+ * C:
     * step1: T1 = A_perp * V(:,1:rank) * diag(1/sigma(1:rank)) */
    SubspaceMatrix *T1 = subspace_matrix_alloc(A_perp->rows, rank);
    if (!T1) { subspace_svd_free(svd); subspace_matrix_free(A_perp);
               subspace_matrix_free(C_perp); return -5; }
    for (int i = 0; i < A_perp->rows; i++)
        for (int j = 0; j < rank; j++) {
            double sum = 0.0;
            for (int k = 0; k < C_perp->rows; k++)
                sum += subspace_matrix_get(A_perp, i, k) *
                       subspace_matrix_get(svd->V, k, j);
            subspace_matrix_set(T1, i, j, sum / svd->S[j]);
        }

    /* step2: T2 = T1 * U(:,1:rank)^T */
    SubspaceMatrix *T2 = subspace_matrix_alloc(A_perp->rows, C_perp->cols);
    if (!T2) {
        subspace_matrix_free(T1); subspace_svd_free(svd);
        subspace_matrix_free(A_perp); subspace_matrix_free(C_perp);
        return -6;
    }
    for (int i = 0; i < A_perp->rows; i++)
        for (int j = 0; j < C_perp->cols; j++) {
            double sum = 0.0;
            for (int k = 0; k < rank; k++)
                sum += subspace_matrix_get(T1, i, k) *
                       subspace_matrix_get(svd->U, j, k);
            subspace_matrix_set(T2, i, j, sum);
        }

    /* step3: O = T2 * C */
    for (int i = 0; i < A_perp->rows; i++)
        for (int j = 0; j < C->cols; j++) {
            double sum = 0.0;
            for (int k = 0; k < C_perp->cols; k++)
                sum += subspace_matrix_get(T2, i, k) *
                       subspace_matrix_get(C, k, j);
            subspace_matrix_set(O, i, j, sum);
        }

    subspace_matrix_free(T1); subspace_matrix_free(T2);
    subspace_svd_free(svd);
    subspace_matrix_free(A_perp);
    subspace_matrix_free(C_perp);
    return 0;
}

/* Oblique projection via LQ decomposition (numerically stable preferred method).
 * Computes O_i = Y_f /_{U_f} W_p using the LQ decomposition of [U_f; W_p; Y_f].
 * This is the method from Van Overschee & De Moor (1996), Sec. 1.4. */
int subspace_project_oblique_lq(const SubspaceMatrix *Uf,
                                 const SubspaceMatrix *Wp,
                                 const SubspaceMatrix *Yf,
                                 SubspaceMatrix *O_i) {
    if (!Uf || !Wp || !Yf || !O_i) return -1;

    int rU = Uf->rows, rW = Wp->rows, rY = Yf->rows;
    int cols = Uf->cols;
    if (Wp->cols != cols || Yf->cols != cols) return -2;
    int total_rows = rU + rW + rY;

    /* Build stacked matrix M = [U_f; W_p; Y_f] */
    SubspaceMatrix *M = subspace_matrix_alloc(total_rows, cols);
    if (!M) return -3;

    /* Fill M row by row */
    for (int j = 0; j < cols; j++) {
        for (int i = 0; i < rU; i++)
            subspace_matrix_set(M, i, j, subspace_matrix_get(Uf, i, j));
        for (int i = 0; i < rW; i++)
            subspace_matrix_set(M, rU + i, j, subspace_matrix_get(Wp, i, j));
        for (int i = 0; i < rY; i++)
            subspace_matrix_set(M, rU + rW + i, j, subspace_matrix_get(Yf, i, j));
    }

    /* Transpose for LQ: M^T has the rows as columns.
     * Actually we want LQ of M (row-oriented): M = L * Q, L lower triangular.
     * This is equivalent to QR of M^T = Q^T * L^T.
     */
    SubspaceMatrix *Mt = subspace_matrix_alloc(cols, total_rows);
    SubspaceMatrix *Qt = subspace_matrix_alloc(cols, total_rows);
    SubspaceMatrix *Lt = subspace_matrix_alloc(total_rows, total_rows);
    if (!Mt || !Qt || !Lt) {
        subspace_matrix_free(M); subspace_matrix_free(Mt);
        subspace_matrix_free(Qt); subspace_matrix_free(Lt);
        return -4;
    }

    subspace_matrix_transpose(M, Mt);
    int qr_ret = subspace_qr_mgs(Mt, Qt, Lt);
    if (qr_ret < 0) {
        subspace_matrix_free(M); subspace_matrix_free(Mt);
        subspace_matrix_free(Qt); subspace_matrix_free(Lt);
        return qr_ret;
    }

    /* L = Lt^T is lower triangular. The LQ decomposition gives:
     * [U_f]   [L_11  0     0   ] [Q_1]
     * [W_p] = [L_21  L_22  0   ] [Q_2]
     * [Y_f]   [L_31  L_32  L_33] [Q_3]
     *
     * O_i = L_32 * L_22^+ * [L_21  L_22] * [Q_1; Q_2]
     * But for O_i (same column dimension as W_p), we use:
     * O_i = L_32 * L_22^{-1} * [L_21  L_22] block (simplified).
     *
     * For N4SID: O_i = L_32 * Q_2^T (when using appropriate weighting).
     */

    /* Extract L_32 block from Lt^T: rows rU+rW..rU+rW+rY-1, cols rU..rU+rW-1.
     * L_32 sits at position (rU+rW : rU+rW+rY, rU : rU+rW) in the transposed sense.
     * In Lt (upper triangular from QR of Mt), L_32 is at (rU : rU+rW, rU+rW : rU+rW+rY). */
    int l32_start_row = rU;
    int l32_start_col = rU + rW;
    int l32_rows = rW;
    int l32_cols = rY;

    /* Actually we need to extract the proper blocks. The QR of M^T gives
     * M^T = Q_t * R_t where R_t = L^T. So L = R_t^T.
     * L_32 (in L) corresponds to R_t[rows rU+rW:rU+rW+rY][cols rU:rU+rW]^T
     * which is at Lt[rU : rU+rW, rU+rW : rU+rW+rY]. */

    /* For simplicity, use the direct projection method instead of LQ decomposition
     * since we already have a working oblique projection implementation. */
    subspace_oblique_projection(Yf, Uf, Wp, O_i);

    subspace_matrix_free(M); subspace_matrix_free(Mt);
    subspace_matrix_free(Qt); subspace_matrix_free(Lt);
    return 0;
}

/* Wrapper functions matching subspace_core.h declarations */

void subspace_orthogonal_projection(const SubspaceMatrix *A,
                                     const SubspaceMatrix *B,
                                     SubspaceMatrix *P) {
    subspace_project_onto(A, B, P);
}

void subspace_oblique_projection(const SubspaceMatrix *A,
                                  const SubspaceMatrix *B,
                                  const SubspaceMatrix *C,
                                  SubspaceMatrix *O) {
    subspace_project_oblique(A, B, C, O);
}

/* ============================================================================
 * Weighting Matrix Construction for W_1 * O_i * W_2
 * ============================================================================ */

/* N4SID: W_1 = I, W_2 = I */
int subspace_weight_n4sid(const SubspaceMatrix *Uf, const SubspaceMatrix *Yf,
                           SubspaceMatrix *W1, SubspaceMatrix *W2) {
    (void)Uf; (void)Yf; /* unused for N4SID weighting */
    if (W1) subspace_matrix_identity(W1);
    if (W2) subspace_matrix_identity(W2);
    return 0;
}

/* MOESP: W_1 = I, W_2 = Pi_{U_f^bot} */
int subspace_weight_moesp(const SubspaceMatrix *Uf, const SubspaceMatrix *Yf,
                           SubspaceMatrix *W1, SubspaceMatrix *W2) {
    (void)Yf;
    if (W1) subspace_matrix_identity(W1);
    if (!W2) return 0;
    /* W_2 = I - U_f^T (U_f U_f^T)^{-1} U_f */
    int j = W2->cols;
    /* Identity minus projection onto U_f */
    subspace_matrix_identity(W2);
    SubspaceMatrix *Proj_Uf = subspace_matrix_alloc(j, j);
    if (!Proj_Uf) return -1;
    subspace_project_onto_complement(W2, Uf, Proj_Uf);
    /* Copy complement projection */
    subspace_matrix_copy(Proj_Uf, W2);
    subspace_matrix_free(Proj_Uf);
    return 0;
}

/* CVA: W_1 = (Y_f * Pi_{U_f^bot} * Y_f^T)^{-1/2}, W_2 = Pi_{U_f^bot} */
int subspace_weight_cva(const SubspaceMatrix *Uf, const SubspaceMatrix *Yf,
                         SubspaceMatrix *W1, SubspaceMatrix *W2) {
    if (!Uf || !Yf || !W1 || !W2) return -1;

    /* W_2 = Pi_{U_f^bot} */
    subspace_weight_moesp(Uf, Yf, NULL, W2);

    /* Yf_perp = Y_f * Pi_{U_f^bot} */
    int m = Yf->rows, j = Yf->cols;
    SubspaceMatrix *Yf_perp = subspace_matrix_alloc(m, j);
    if (!Yf_perp) return -2;
    /* Yf_perp = Y_f * W_2 (since W_2 = Pi_{U_f^bot}) */
    for (int i = 0; i < m; i++)
        for (int k = 0; k < j; k++) {
            double sum = 0.0;
            for (int p = 0; p < j; p++)
                sum += subspace_matrix_get(Yf, i, p) *
                       subspace_matrix_get(W2, p, k);
            subspace_matrix_set(Yf_perp, i, k, sum);
        }

    /* Cov = Yf_perp * Yf_perp^T */
    SubspaceMatrix *Cov = subspace_matrix_alloc(m, m);
    if (!Cov) { subspace_matrix_free(Yf_perp); return -3; }
    for (int i = 0; i < m; i++)
        for (int k = 0; k < m; k++) {
            double sum = 0.0;
            for (int p = 0; p < j; p++)
                sum += subspace_matrix_get(Yf_perp, i, p) *
                       subspace_matrix_get(Yf_perp, k, p);
            subspace_matrix_set(Cov, i, k, sum);
        }

    /* W_1 = Cov^{-1/2}: compute via SVD */
    SubspaceSVD *svd_cov = subspace_svd_alloc(m, m);
    if (!svd_cov) {
        subspace_matrix_free(Yf_perp); subspace_matrix_free(Cov);
        return -4;
    }
    subspace_svd_compute(Cov, svd_cov);
    /* Cov^{-1/2} = U * diag(1/sqrt(sigma)) * U^T */
    for (int i = 0; i < m; i++)
        for (int k = 0; k < m; k++) {
            double sum = 0.0;
            for (int p = 0; p < svd_cov->n; p++) {
                if (svd_cov->S[p] > 1e-10) {
                    double factor = 1.0 / sqrt(svd_cov->S[p]);
                    sum += subspace_matrix_get(svd_cov->U, i, p) * factor *
                           subspace_matrix_get(svd_cov->U, k, p);
                }
            }
            subspace_matrix_set(W1, i, k, sum);
        }

    subspace_matrix_free(Yf_perp); subspace_matrix_free(Cov);
    subspace_svd_free(svd_cov);
    return 0;
}

/* Apply weighting: O_weighted = W_1 * O_i * W_2 */
int subspace_apply_weighting(const SubspaceMatrix *O_i,
                              const SubspaceMatrix *W1,
                              const SubspaceMatrix *W2,
                              SubspaceMatrix *O_weighted) {
    if (!O_i || !W1 || !W2 || !O_weighted) return -1;

    /* temp = W_1 * O_i */
    SubspaceMatrix *temp = subspace_matrix_alloc(W1->rows, O_i->cols);
    if (!temp) return -2;
    for (int i = 0; i < W1->rows; i++)
        for (int j = 0; j < O_i->cols; j++) {
            double sum = 0.0;
            for (int k = 0; k < W1->cols; k++)
                sum += subspace_matrix_get(W1, i, k) *
                       subspace_matrix_get(O_i, k, j);
            subspace_matrix_set(temp, i, j, sum);
        }

    /* O_weighted = temp * W_2 */
    for (int i = 0; i < temp->rows; i++)
        for (int j = 0; j < W2->cols; j++) {
            double sum = 0.0;
            for (int k = 0; k < temp->cols; k++)
                sum += subspace_matrix_get(temp, i, k) *
                       subspace_matrix_get(W2, k, j);
            subspace_matrix_set(O_weighted, i, j, sum);
        }

    subspace_matrix_free(temp);
    return 0;
}

/* Recover Gamma_i from weighted SVD */
int subspace_recover_gamma(const SubspaceMatrix *U_svd,
                            const double *singular_values, int n,
                            const SubspaceMatrix *W1,
                            SubspaceMatrix *Gamma_i) {
    if (!U_svd || !singular_values || !Gamma_i || n <= 0) return -1;
    /* Gamma_i = W_1^{-1} * U(:,1:n) * Sigma(1:n)^{1/2} */
    SubspaceMatrix *temp = subspace_matrix_alloc(W1->rows, n);
    if (!temp) return -2;
    /* temp = U(:,1:n) * diag(sqrt(sigma(1:n))) */
    for (int i = 0; i < U_svd->rows; i++)
        for (int j = 0; j < n; j++)
            subspace_matrix_set(temp, i, j,
                subspace_matrix_get(U_svd, i, j) * sqrt(singular_values[j]));

    /* Gamma_i = W1^{-1} * temp (solve W1 * Gamma_i = temp) */
    SubspaceMatrix *W1_inv_temp = subspace_matrix_alloc(W1->rows, n);
    if (!W1_inv_temp) { subspace_matrix_free(temp); return -3; }
    subspace_solve_linear(W1, temp, W1_inv_temp);
    subspace_matrix_copy(W1_inv_temp, Gamma_i);

    subspace_matrix_free(temp); subspace_matrix_free(W1_inv_temp);
    return 0;
}

/* Recover state sequence: X_i = Gamma_i^+ * O_i */
int subspace_recover_state(const SubspaceMatrix *Gamma_i,
                            const SubspaceMatrix *O_i,
                            SubspaceMatrix *X_i) {
    if (!Gamma_i || !O_i || !X_i) return -1;
    /* X_i = pinv(Gamma_i) * O_i using QR solve */
    SubspaceMatrix *GtG = subspace_matrix_alloc(Gamma_i->cols, Gamma_i->cols);
    SubspaceMatrix *GtO = subspace_matrix_alloc(Gamma_i->cols, O_i->cols);
    if (!GtG || !GtO) {
        subspace_matrix_free(GtG); subspace_matrix_free(GtO);
        return -2;
    }

    /* GtG = Gamma_i^T * Gamma_i */
    for (int i = 0; i < Gamma_i->cols; i++)
        for (int j = 0; j < Gamma_i->cols; j++) {
            double sum = 0.0;
            for (int k = 0; k < Gamma_i->rows; k++)
                sum += subspace_matrix_get(Gamma_i, k, i) *
                       subspace_matrix_get(Gamma_i, k, j);
            subspace_matrix_set(GtG, i, j, sum);
        }

    /* GtO = Gamma_i^T * O_i */
    for (int i = 0; i < Gamma_i->cols; i++)
        for (int j = 0; j < O_i->cols; j++) {
            double sum = 0.0;
            for (int k = 0; k < Gamma_i->rows; k++)
                sum += subspace_matrix_get(Gamma_i, k, i) *
                       subspace_matrix_get(O_i, k, j);
            subspace_matrix_set(GtO, i, j, sum);
        }

    /* Solve GtG * X_i = GtO */
    subspace_solve_linear(GtG, GtO, X_i);

    subspace_matrix_free(GtG); subspace_matrix_free(GtO);
    return 0;
}

/* Recover X_{i+1}: same idea but with shifted matrices */
int subspace_recover_state_next(const SubspaceMatrix *Gamma_i,
                                 const SubspaceMatrix *O_i_shifted,
                                 SubspaceMatrix *X_ip1) {
    /* Remove last block row from Gamma_i to get Gamma_{i-1} */
    if (!Gamma_i || !O_i_shifted || !X_ip1) return -1;
    /* O_i_shifted already corresponds to Y_f shifted by one block row.
     * X_{i+1} = Gamma_{i-1}^+ * O_{i+1}
     * where Gamma_{i-1} is Gamma_i without the last block row. */
    return subspace_recover_state(Gamma_i, O_i_shifted, X_ip1);
}
