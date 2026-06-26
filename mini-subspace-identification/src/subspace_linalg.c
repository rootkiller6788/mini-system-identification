#include "subspace_core.h"
#include "subspace_linalg.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ============================================================================
 * Subspace Identification -- Linear Algebra Implementation
 *
 * Pure C implementations of core numerical linear algebra needed for
 * subspace identification: QR (MGS + Householder), one-sided Jacobi SVD,
 * Cholesky, triangular solvers, and the QR eigenvalue algorithm.
 *
 * All routines are self-contained with no external BLAS/LAPACK dependency.
 * ============================================================================ */

/* --- BLAS-like Level 1 operations --- */
double subspace_dot(const double *x, const double *y, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += x[i] * y[i];
    return sum;
}

double subspace_nrm2(const double *x, int n) {
    double scale = 0.0, ssq = 1.0;
    for (int i = 0; i < n; i++) {
        if (x[i] != 0.0) {
            double absxi = fabs(x[i]);
            if (scale < absxi) {
                ssq = 1.0 + ssq * (scale / absxi) * (scale / absxi);
                scale = absxi;
            } else {
                ssq += (absxi / scale) * (absxi / scale);
            }
        }
    }
    return scale * sqrt(ssq);
}

void subspace_axpy(double alpha, const double *x, double *y, int n) {
    for (int i = 0; i < n; i++) y[i] += alpha * x[i];
}

void subspace_scal(double alpha, double *x, int n) {
    for (int i = 0; i < n; i++) x[i] *= alpha;
}

double subspace_asum(const double *x, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += fabs(x[i]);
    return sum;
}

/* --- BLAS-like Level 2: GEMV, GER --- */
void subspace_gemv(bool trans, double alpha, const SubspaceMatrix *A,
                    const double *x, double beta, double *y) {
    if (!A || !x || !y) return;
    int m = A->rows, n = A->cols;
    if (!trans) {
        for (int i = 0; i < m; i++) {
            double sum = 0.0;
            for (int j = 0; j < n; j++)
                sum += subspace_matrix_get(A, i, j) * x[j];
            y[i] = alpha * sum + beta * y[i];
        }
    } else {
        for (int i = 0; i < n; i++) {
            double sum = 0.0;
            for (int j = 0; j < m; j++)
                sum += subspace_matrix_get(A, j, i) * x[j];
            y[i] = alpha * sum + beta * y[i];
        }
    }
}

void subspace_ger(double alpha, const double *x, const double *y,
                   SubspaceMatrix *A) {
    if (!A || !x || !y) return;
    for (int i = 0; i < A->rows; i++)
        for (int j = 0; j < A->cols; j++)
            subspace_matrix_set(A, i, j,
                subspace_matrix_get(A, i, j) + alpha * x[i] * y[j]);
}

/* --- BLAS-like Level 3: GEMM --- */
void subspace_gemm(bool transA, bool transB, double alpha,
                    const SubspaceMatrix *A, const SubspaceMatrix *B,
                    double beta, SubspaceMatrix *C) {
    if (!A || !B || !C) return;
    int m = transA ? A->cols : A->rows;
    int n = transB ? B->rows : B->cols;
    int k = transA ? A->rows : A->cols;
    if (k != (transB ? B->cols : B->rows)) return;
    if (C->rows != m || C->cols != n) return;

    if (beta != 1.0) {
        for (int i = 0; i < m * n; i++) C->data[i] *= beta;
    }
    for (int j = 0; j < n; j++) {
        for (int p = 0; p < k; p++) {
            double bv = transB ? subspace_matrix_get(B, j, p)
                               : subspace_matrix_get(B, p, j);
            if (fabs(bv) < 1e-30) continue;
            for (int i = 0; i < m; i++) {
                double av = transA ? subspace_matrix_get(A, p, i)
                                   : subspace_matrix_get(A, i, p);
                subspace_matrix_set(C, i, j,
                    subspace_matrix_get(C, i, j) + alpha * av * bv);
            }
        }
    }
}

/* ============================================================================
 * QR Factorization: Modified Gram-Schmidt with Reorthogonalization
 *
 * Daniel, Gragg, Kaufman & Stewart (1976): Reorthogonalization improves
 * numerical stability for near-rank-deficient matrices, which is critical
 * in subspace identification where the projection matrix may have a
 * sharp drop in singular values.
 * ============================================================================ */

int subspace_qr_mgs(const SubspaceMatrix *A, SubspaceMatrix *Q, SubspaceMatrix *R) {
    if (!A || !Q || !R) return -1;
    int m = A->rows, n = A->cols;
    if (Q->rows != m || Q->cols != n || R->rows != n || R->cols != n) return -2;

    subspace_matrix_copy(A, Q);
    subspace_matrix_fill(R, 0.0);

    double *v = (double*)malloc((size_t)m * sizeof(double));
    if (!v) return -3;

    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) v[i] = subspace_matrix_get(Q, i, j);
        /* First pass */
        for (int k = 0; k < j; k++) {
            double dot = 0.0;
            for (int i = 0; i < m; i++)
                dot += subspace_matrix_get(Q, i, k) * v[i];
            subspace_matrix_set(R, k, j, dot);
            for (int i = 0; i < m; i++)
                v[i] -= dot * subspace_matrix_get(Q, i, k);
        }
        /* Reorthogonalization pass */
        for (int k = 0; k < j; k++) {
            double dot = 0.0;
            for (int i = 0; i < m; i++)
                dot += subspace_matrix_get(Q, i, k) * v[i];
            for (int i = 0; i < m; i++)
                v[i] -= dot * subspace_matrix_get(Q, i, k);
            subspace_matrix_set(R, k, j, subspace_matrix_get(R, k, j) + dot);
        }
        /* Norm */
        double nrm = 0.0;
        for (int i = 0; i < m; i++) nrm += v[i] * v[i];
        nrm = sqrt(nrm);
        subspace_matrix_set(R, j, j, nrm);
        if (nrm > 1e-15) {
            for (int i = 0; i < m; i++)
                subspace_matrix_set(Q, i, j, v[i] / nrm);
        } else {
            for (int i = 0; i < m; i++) subspace_matrix_set(Q, i, j, 0.0);
        }
    }
    free(v);
    return 0;
}

/* ============================================================================
 * SVD: One-Sided Jacobi Algorithm (Demmel & Veselic, 1992)
 *
 * Computes A = U * Sigma * V^T by applying plane rotations to make the
 * columns of A orthogonal. This method computes singular values to high
 * relative accuracy, essential for distinguishing signal from noise in
 * subspace identification.
 * ============================================================================ */

int subspace_svd_compute(const SubspaceMatrix *A, SubspaceSVD *result) {
    if (!A || !result) return -1;
    int m = A->rows, n = A->cols, p = (m < n) ? m : n;

    SubspaceMatrix *W = subspace_matrix_alloc(m, n);
    SubspaceMatrix *Vt = subspace_matrix_alloc(n, n);
    if (!W || !Vt) { subspace_matrix_free(W); subspace_matrix_free(Vt); return -2; }
    subspace_matrix_copy(A, W);
    subspace_matrix_identity(Vt);

    double *cnorms = (double*)malloc((size_t)n * sizeof(double));
    if (!cnorms) { subspace_matrix_free(W); subspace_matrix_free(Vt); return -3; }

    int max_iter = 30 * n, iter;
    for (iter = 0; iter < max_iter; iter++) {
        int rot = 0;
        for (int j = 0; j < n; j++) {
            double nr = 0.0;
            for (int i = 0; i < m; i++) {
                double vv = subspace_matrix_get(W, i, j); nr += vv * vv;
            }
            cnorms[j] = sqrt(nr);
        }
        for (int j = 0; j < n - 1; j++) {
            for (int kk = j + 1; kk < n; kk++) {
                double dot_jk = 0.0;
                for (int i = 0; i < m; i++)
                    dot_jk += subspace_matrix_get(W, i, j) *
                              subspace_matrix_get(W, i, kk);
                double tol = 1e-13 * cnorms[j] * cnorms[kk];
                if (fabs(dot_jk) <= tol) continue;

                double a = cnorms[j] * cnorms[j];
                double c = cnorms[kk] * cnorms[kk];
                double b = dot_jk;
                double zeta = (c - a) / (2.0 * b);
                double t = (zeta >= 0)
                    ? 1.0 / (zeta + sqrt(1.0 + zeta * zeta))
                    : 1.0 / (zeta - sqrt(1.0 + zeta * zeta));
                double cs = 1.0 / sqrt(1.0 + t * t);
                double sn = cs * t;

                for (int i = 0; i < m; i++) {
                    double wij = subspace_matrix_get(W, i, j);
                    double wik = subspace_matrix_get(W, i, kk);
                    subspace_matrix_set(W, i, j, cs * wij - sn * wik);
                    subspace_matrix_set(W, i, kk, sn * wij + cs * wik);
                }
                for (int i = 0; i < n; i++) {
                    double vij = subspace_matrix_get(Vt, i, j);
                    double vik = subspace_matrix_get(Vt, i, kk);
                    subspace_matrix_set(Vt, i, j, cs * vij - sn * vik);
                    subspace_matrix_set(Vt, i, kk, sn * vij + cs * vik);
                }
                rot++;
            }
        }
        if (rot == 0) break;
    }

    for (int j = 0; j < p; j++) {
        double nr = 0.0;
        for (int i = 0; i < m; i++) {
            double vv = subspace_matrix_get(W, i, j); nr += vv * vv;
        }
        nr = sqrt(nr); result->S[j] = nr;
        if (nr > 1e-15 && result->U)
            for (int i = 0; i < m; i++)
                subspace_matrix_set(result->U, i, j,
                    subspace_matrix_get(W, i, j) / nr);
    }
    if (result->V)
        for (int i = 0; i < n; i++)
            for (int j = 0; j < p; j++)
                subspace_matrix_set(result->V, i, j,
                    subspace_matrix_get(Vt, i, j));

    /* Sort descending */
    for (int i = 0; i < p - 1; i++) {
        int mi = i;
        for (int j = i + 1; j < p; j++)
            if (result->S[j] > result->S[mi]) mi = j;
        if (mi != i) {
            double tmp = result->S[i]; result->S[i] = result->S[mi]; result->S[mi] = tmp;
            if (result->U) for (int k = 0; k < m; k++) {
                double t2 = subspace_matrix_get(result->U, k, i);
                subspace_matrix_set(result->U, k, i,
                    subspace_matrix_get(result->U, k, mi));
                subspace_matrix_set(result->U, k, mi, t2);
            }
            if (result->V) for (int k = 0; k < n; k++) {
                double t2 = subspace_matrix_get(result->V, k, i);
                subspace_matrix_set(result->V, k, i,
                    subspace_matrix_get(result->V, k, mi));
                subspace_matrix_set(result->V, k, mi, t2);
            }
        }
    }

    free(cnorms);
    subspace_matrix_free(W); subspace_matrix_free(Vt);
    return iter;
}

/* ============================================================================
 * Cholesky: A = L * L^T (column-oriented, Golub & Van Loan Sec 4.2)
 * ============================================================================ */

int subspace_cholesky(const SubspaceMatrix *A, SubspaceMatrix *L) {
    if (!A || !L) return -1;
    int n = A->rows;
    if (A->cols != n || L->rows != n || L->cols != n) return -2;
    subspace_matrix_fill(L, 0.0);
    for (int j = 0; j < n; j++) {
        double sum = subspace_matrix_get(A, j, j);
        for (int k = 0; k < j; k++) {
            double ljk = subspace_matrix_get(L, j, k);
            sum -= ljk * ljk;
        }
        if (sum <= 1e-15) return -1;
        double ljj = sqrt(sum);
        subspace_matrix_set(L, j, j, ljj);
        for (int i = j + 1; i < n; i++) {
            double sum_off = subspace_matrix_get(A, i, j);
            for (int k = 0; k < j; k++)
                sum_off -= subspace_matrix_get(L, i, k) *
                           subspace_matrix_get(L, j, k);
            subspace_matrix_set(L, i, j, sum_off / ljj);
        }
    }
    return 0;
}

/* ============================================================================
 * Householder Reflectors for QR
 * ============================================================================ */

double subspace_householder(double *x, int n) {
    double nrm = 0.0;
    for (int i = 0; i < n; i++) nrm += x[i] * x[i];
    nrm = sqrt(nrm);
    if (nrm < 1e-15) return 0.0;
    double alpha = x[0], sigma = (alpha > 0) ? -nrm : nrm;
    x[0] = alpha - sigma;
    double tau = -sigma * x[0] / (nrm * nrm);
    double x0_inv = 1.0 / x[0];
    for (int i = 0; i < n; i++) x[i] *= x0_inv;
    x[0] = 1.0;
    return tau;
}

int subspace_qr_householder(const SubspaceMatrix *A, SubspaceMatrix *Q,
                             SubspaceMatrix *R, double *tau) {
    if (!A || !Q || !R) return -1;
    int m = A->rows, n = A->cols;
    if (n > m) return -2;
    subspace_matrix_copy(A, R);
    subspace_matrix_identity(Q);
    double *v = (double*)malloc((size_t)m * sizeof(double));
    if (!v) return -3;
    for (int j = 0; j < n; j++) {
        int len = m - j;
        for (int i = 0; i < len; i++)
            v[i] = subspace_matrix_get(R, j + i, j);
        double tau_j = subspace_householder(v, len);
        if (tau) tau[j] = tau_j;
        for (int col = j; col < n; col++) {
            double dot = 0.0;
            for (int i = 0; i < len; i++)
                dot += v[i] * subspace_matrix_get(R, j + i, col);
            double factor = tau_j * dot;
            for (int i = 0; i < len; i++)
                subspace_matrix_set(R, j + i, col,
                    subspace_matrix_get(R, j + i, col) - factor * v[i]);
        }
        for (int row = 0; row < m; row++) {
            double dot = 0.0;
            for (int i = 0; i < len; i++)
                dot += v[i] * subspace_matrix_get(Q, row, j + i);
            double factor = tau_j * dot;
            for (int i = 0; i < len; i++)
                subspace_matrix_set(Q, row, j + i,
                    subspace_matrix_get(Q, row, j + i) - factor * v[i]);
        }
    }
    free(v);
    return 0;
}

void subspace_qr_decompose(const SubspaceMatrix *A, SubspaceMatrix *Q,
                            SubspaceMatrix *R) {
    subspace_qr_mgs(A, Q, R);
}

/* ============================================================================
 * Triangular Solver and Linear System Solver
 * ============================================================================ */

void subspace_solve_triangular(const SubspaceMatrix *R, const SubspaceMatrix *B,
                                SubspaceMatrix *X, bool upper, bool trans) {
    if (!R || !B || !X) return;
    int n = R->rows, nrhs = B->cols;
    if (R->cols != n || B->rows != n || X->rows != n || X->cols != nrhs) return;
    subspace_matrix_copy(B, X);

    if (upper && !trans) {
        for (int j = 0; j < nrhs; j++)
            for (int i = n - 1; i >= 0; i--) {
                double sum = subspace_matrix_get(X, i, j);
                for (int k = i + 1; k < n; k++)
                    sum -= subspace_matrix_get(R, i, k) *
                           subspace_matrix_get(X, k, j);
                double rii = subspace_matrix_get(R, i, i);
                subspace_matrix_set(X, i, j, (fabs(rii) > 1e-15) ? sum / rii : 0.0);
            }
    } else if (!upper && !trans) {
        for (int j = 0; j < nrhs; j++)
            for (int i = 0; i < n; i++) {
                double sum = subspace_matrix_get(X, i, j);
                for (int k = 0; k < i; k++)
                    sum -= subspace_matrix_get(R, i, k) *
                           subspace_matrix_get(X, k, j);
                double rii = subspace_matrix_get(R, i, i);
                subspace_matrix_set(X, i, j, (fabs(rii) > 1e-15) ? sum / rii : 0.0);
            }
    } else if (upper && trans) {
        for (int j = 0; j < nrhs; j++)
            for (int i = 0; i < n; i++) {
                double sum = subspace_matrix_get(X, i, j);
                for (int k = 0; k < i; k++)
                    sum -= subspace_matrix_get(R, k, i) *
                           subspace_matrix_get(X, k, j);
                double rii = subspace_matrix_get(R, i, i);
                subspace_matrix_set(X, i, j, (fabs(rii) > 1e-15) ? sum / rii : 0.0);
            }
    } else {
        for (int j = 0; j < nrhs; j++)
            for (int i = n - 1; i >= 0; i--) {
                double sum = subspace_matrix_get(X, i, j);
                for (int k = i + 1; k < n; k++)
                    sum -= subspace_matrix_get(R, k, i) *
                           subspace_matrix_get(X, k, j);
                double rii = subspace_matrix_get(R, i, i);
                subspace_matrix_set(X, i, j, (fabs(rii) > 1e-15) ? sum / rii : 0.0);
            }
    }
}

int subspace_solve_linear(const SubspaceMatrix *A, const SubspaceMatrix *B,
                           SubspaceMatrix *X) {
    if (!A || !B || !X) return -1;
    int m = A->rows, n = A->cols;
    if (m != n) return -2;

    SubspaceMatrix *Q = subspace_matrix_alloc(m, n);
    SubspaceMatrix *R = subspace_matrix_alloc(n, n);
    SubspaceMatrix *QtB = subspace_matrix_alloc(n, B->cols);
    if (!Q || !R || !QtB) {
        subspace_matrix_free(Q); subspace_matrix_free(R);
        subspace_matrix_free(QtB); return -3;
    }
    subspace_qr_mgs(A, Q, R);
    for (int j = 0; j < B->cols; j++)
        for (int i = 0; i < n; i++) {
            double sum = 0.0;
            for (int k = 0; k < m; k++)
                sum += subspace_matrix_get(Q, k, i) *
                       subspace_matrix_get(B, k, j);
            subspace_matrix_set(QtB, i, j, sum);
        }
    subspace_solve_triangular(R, QtB, X, true, false);
    subspace_matrix_free(Q); subspace_matrix_free(R);
    subspace_matrix_free(QtB);
    return 0;
}

/* ============================================================================
 * Hessenberg Reduction + QR Eigenvalue Algorithm
 * ============================================================================ */

static void hessenberg_reduce(double *H, int n) {
    for (int k = 0; k < n - 2; k++) {
        double nrm = 0.0;
        for (int i = k + 1; i < n; i++)
            nrm += H[(size_t)i * n + k] * H[(size_t)i * n + k];
        nrm = sqrt(nrm);
        if (fabs(nrm) < 1e-15) continue;
        double alpha = H[(size_t)(k+1) * n + k];
        double sigma = (alpha > 0) ? -nrm : nrm;
        double *v = (double*)calloc((size_t)n, sizeof(double));
        if (!v) continue;
        v[k+1] = alpha - sigma;
        for (int i = k + 2; i < n; i++) v[i] = H[(size_t)i * n + k];
        double vnsq = 0.0;
        for (int i = k + 1; i < n; i++) vnsq += v[i] * v[i];
        double tau = 2.0 / vnsq;
        for (int j = k; j < n; j++) {
            double dot = 0.0;
            for (int i = k + 1; i < n; i++)
                dot += v[i] * H[(size_t)i * n + j];
            double factor = tau * dot;
            for (int i = k + 1; i < n; i++)
                H[(size_t)i * n + j] -= factor * v[i];
        }
        for (int i = 0; i < n; i++) {
            double dot = 0.0;
            for (int j = k + 1; j < n; j++)
                dot += H[(size_t)i * n + j] * v[j];
            double factor = tau * dot;
            for (int j = k + 1; j < n; j++)
                H[(size_t)i * n + j] -= factor * v[j];
        }
        free(v);
    }
}

double subspace_eigenvalues_real(double *A_data, int n, double *eig_real,
                                  double *eig_imag) {
    if (!A_data || !eig_real || !eig_imag || n <= 0) return 0.0;
    double max_eig = 0.0;

    double *H = (double*)malloc((size_t)n * (size_t)n * sizeof(double));
    if (!H) return 0.0;
    memcpy(H, A_data, (size_t)n * (size_t)n * sizeof(double));

    hessenberg_reduce(H, n);

    int max_iter = 100 * n, iter;
    for (iter = 0; iter < max_iter; iter++) {
        bool converged = true;
        for (int i = 1; i < n; i++)
            if (fabs(H[(size_t)i * n + (i-1)]) > 1e-12 * (
                fabs(H[(size_t)(i-1) * n + (i-1)]) + fabs(H[(size_t)i * n + i]))) {
                converged = false; break;
            }
        if (converged) break;

        double a = H[(size_t)(n-2) * n + (n-2)];
        double b = H[(size_t)(n-2) * n + (n-1)];
        double c = H[(size_t)(n-1) * n + (n-2)];
        double d = H[(size_t)(n-1) * n + (n-1)];
        double tr = a + d, det = a * d - b * c;
        double disc = tr * tr - 4.0 * det;
        double mu = (disc >= 0) ?
            ((fabs((tr + sqrt(disc))/2 - d) < fabs((tr - sqrt(disc))/2 - d))
             ? (tr + sqrt(disc))/2 : (tr - sqrt(disc))/2) : tr / 2.0;

        for (int k = 0; k < n - 1; k++) {
            double x = H[(size_t)k * n + k] - mu;
            double y = H[(size_t)(k+1) * n + k];
            double r = sqrt(x * x + y * y);
            if (r < 1e-15) continue;
            double cs = x / r, sn = -y / r;
            for (int j = k; j < n; j++) {
                double t1 = H[(size_t)k * n + j];
                double t2 = H[(size_t)(k+1) * n + j];
                H[(size_t)k * n + j]     = cs * t1 - sn * t2;
                H[(size_t)(k+1) * n + j] = sn * t1 + cs * t2;
            }
            for (int i = 0; i < n && i <= k + 1; i++) {
                double t1 = H[(size_t)i * n + k];
                double t2 = H[(size_t)i * n + (k+1)];
                H[(size_t)i * n + k]     = cs * t1 - sn * t2;
                H[(size_t)i * n + (k+1)] = sn * t1 + cs * t2;
            }
        }
    }

    int idx = 0;
    while (idx < n) {
        if (idx == n - 1) {
            eig_real[idx] = H[(size_t)idx * n + idx];
            eig_imag[idx] = 0.0; idx++;
        } else {
            double sub = fabs(H[(size_t)(idx+1) * n + idx]);
            if (sub < 1e-12 * (fabs(H[(size_t)idx * n + idx]) +
                               fabs(H[(size_t)(idx+1) * n + (idx+1)]))) {
                eig_real[idx] = H[(size_t)idx * n + idx];
                eig_imag[idx] = 0.0; idx++;
            } else {
                double a2 = H[(size_t)idx * n + idx];
                double b2 = H[(size_t)idx * n + (idx+1)];
                double c2 = H[(size_t)(idx+1) * n + idx];
                double d2 = H[(size_t)(idx+1) * n + (idx+1)];
                double tr2 = a2 + d2, det2 = a2 * d2 - b2 * c2;
                double disc2 = tr2 * tr2 - 4.0 * det2;
                if (disc2 >= 0) {
                    double sd = sqrt(disc2);
                    eig_real[idx] = (tr2 + sd) / 2.0; eig_imag[idx] = 0.0;
                    eig_real[idx+1] = (tr2 - sd) / 2.0; eig_imag[idx+1] = 0.0;
                } else {
                    eig_real[idx] = tr2 / 2.0; eig_imag[idx] = sqrt(-disc2) / 2.0;
                    eig_real[idx+1] = tr2 / 2.0; eig_imag[idx+1] = -sqrt(-disc2) / 2.0;
                }
                idx += 2;
            }
        }
    }

    for (int k = 0; k < n; k++) {
        double mag = sqrt(eig_real[k]*eig_real[k] + eig_imag[k]*eig_imag[k]);
        if (mag > max_eig) max_eig = mag;
    }
    free(H);
    return max_eig;
}

/* ============================================================================
 * Utility functions
 * ============================================================================ */

double subspace_machine_epsilon(void) {
    double eps = 1.0;
    while (1.0 + eps > 1.0) eps /= 2.0;
    return eps * 2.0;
}

void subspace_sort_descending(double *values, int n, int *perm) {
    if (perm) for (int i = 0; i < n; i++) perm[i] = i;
    for (int i = 0; i < n - 1; i++) {
        int best = i;
        for (int j = i + 1; j < n; j++)
            if (values[j] > values[best]) best = j;
        if (best != i) {
            double tmp = values[i]; values[i] = values[best]; values[best] = tmp;
            if (perm) { int t = perm[i]; perm[i] = perm[best]; perm[best] = t; }
        }
    }
}

double subspace_hypot(double a, double b) {
    double aa = fabs(a), ab = fabs(b);
    if (aa > ab) { double r = ab / aa; return aa * sqrt(1.0 + r*r); }
    else if (ab > 0.0) { double r = aa / ab; return ab * sqrt(1.0 + r*r); }
    return 0.0;
}

void subspace_givens(double a, double b, double *c, double *s) {
    if (fabs(b) < 1e-15) { *c = 1.0; *s = 0.0; return; }
    if (fabs(a) < 1e-15) { *c = 0.0; *s = 1.0; return; }
    double r = subspace_hypot(a, b);
    *c = a / r; *s = -b / r;
}

void subspace_matrix_printf(const SubspaceMatrix *mat, const char *fmt) {
    if (!mat) return;
    for (int i = 0; i < mat->rows; i++) {
        for (int j = 0; j < mat->cols; j++)
            printf(fmt ? fmt : "% .6e ", subspace_matrix_get(mat, i, j));
        printf("\n");
    }
}
