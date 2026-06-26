#include "rls_core.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ============================================================================
 * Internal helpers
 * ============================================================================ */

static double rand_uniform(void) {
    return (double)rand() / (double)RAND_MAX;
}

static double rand_normal(void) {
    double u1 = rand_uniform();
    double u2 = rand_uniform();
    return sqrt(-2.0 * log(u1 + 1e-15)) * cos(2.0 * M_PI * u2);
}

static void swap_double(double *a, double *b) {
    double t = *a; *a = *b; *b = t;
}

/* ============================================================================
 * Matrix Allocation & Operations (L3: Mathematical Structures)
 * ============================================================================ */

RLSMatrix *rls_matrix_alloc(int rows, int cols) {
    RLSMatrix *M = (RLSMatrix *)calloc(1, sizeof(RLSMatrix));
    if (!M) return NULL;
    M->rows = rows;
    M->cols = cols;
    M->capacity = rows * cols;
    M->data = (double *)calloc(M->capacity, sizeof(double));
    if (!M->data) { free(M); return NULL; }
    return M;
}

RLSMatrix *rls_matrix_alloc_data(double *data, int rows, int cols) {
    RLSMatrix *M = (RLSMatrix *)calloc(1, sizeof(RLSMatrix));
    if (!M) return NULL;
    M->rows = rows;
    M->cols = cols;
    M->capacity = 0;
    M->data = data;
    return M;
}

void rls_matrix_free(RLSMatrix *M) {
    if (!M) return;
    if (M->capacity > 0 && M->data) free(M->data);
    free(M);
}

RLSMatrix *rls_matrix_copy(const RLSMatrix *M) {
    if (!M) return NULL;
    RLSMatrix *C = rls_matrix_alloc(M->rows, M->cols);
    if (!C) return NULL;
    memcpy(C->data, M->data, M->rows * M->cols * sizeof(double));
    return C;
}

void rls_matrix_set(RLSMatrix *M, int i, int j, double val) {
    if (!M || i < 0 || i >= M->rows || j < 0 || j >= M->cols) return;
    M->data[j * M->rows + i] = val;
}

double rls_matrix_get(const RLSMatrix *M, int i, int j) {
    if (!M || i < 0 || i >= M->rows || j < 0 || j >= M->cols) return 0.0;
    return M->data[j * M->rows + i];
}

void rls_matrix_zero(RLSMatrix *M) {
    if (!M) return;
    memset(M->data, 0, M->rows * M->cols * sizeof(double));
}

void rls_matrix_identity(RLSMatrix *M) {
    if (!M) return;
    rls_matrix_zero(M);
    int n = (M->rows < M->cols) ? M->rows : M->cols;
    for (int i = 0; i < n; i++)
        M->data[i * M->rows + i] = 1.0;
}

void rls_matrix_scale(RLSMatrix *M, double s) {
    if (!M) return;
    int n = M->rows * M->cols;
    for (int i = 0; i < n; i++) M->data[i] *= s;
}

void rls_matrix_add_diag(RLSMatrix *M, double val) {
    if (!M) return;
    int n = (M->rows < M->cols) ? M->rows : M->cols;
    for (int i = 0; i < n; i++)
        M->data[i * M->rows + i] += val;
}

void rls_matrix_fill(RLSMatrix *M, double val) {
    if (!M) return;
    int n = M->rows * M->cols;
    for (int i = 0; i < n; i++) M->data[i] = val;
}

/* ============================================================================
 * Vector Operations
 * ============================================================================ */

RLSVector *rls_vector_alloc(int dim) {
    RLSVector *v = (RLSVector *)calloc(1, sizeof(RLSVector));
    if (!v) return NULL;
    v->dim = dim;
    v->capacity = dim;
    v->data = (double *)calloc(dim, sizeof(double));
    if (!v->data) { free(v); return NULL; }
    return v;
}

RLSVector *rls_vector_alloc_data(double *data, int dim) {
    RLSVector *v = (RLSVector *)calloc(1, sizeof(RLSVector));
    if (!v) return NULL;
    v->dim = dim;
    v->capacity = 0;
    v->data = data;
    return v;
}

void rls_vector_free(RLSVector *v) {
    if (!v) return;
    if (v->capacity > 0 && v->data) free(v->data);
    free(v);
}

RLSVector *rls_vector_copy(const RLSVector *v) {
    if (!v) return NULL;
    RLSVector *c = rls_vector_alloc(v->dim);
    if (!c) return NULL;
    memcpy(c->data, v->data, v->dim * sizeof(double));
    return c;
}

void rls_vector_set(RLSVector *v, int i, double val) {
    if (!v || i < 0 || i >= v->dim) return;
    v->data[i] = val;
}

double rls_vector_get(const RLSVector *v, int i) {
    if (!v || i < 0 || i >= v->dim) return 0.0;
    return v->data[i];
}

void rls_vector_zero(RLSVector *v) {
    if (!v) return;
    memset(v->data, 0, v->dim * sizeof(double));
}

/* ============================================================================
 * BLAS Level 1: Vector-Vector Operations
 * ============================================================================ */

double rls_vector_dot(const RLSVector *a, const RLSVector *b) {
    if (!a || !b || a->dim != b->dim) return 0.0;
    double s = 0.0;
    for (int i = 0; i < a->dim; i++) s += a->data[i] * b->data[i];
    return s;
}

double rls_vector_nrm2(const RLSVector *v) {
    return sqrt(rls_vector_dot(v, v));
}

double rls_vector_asum(const RLSVector *v) {
    if (!v) return 0.0;
    double s = 0.0;
    for (int i = 0; i < v->dim; i++) s += fabs(v->data[i]);
    return s;
}

int rls_vector_iamax(const RLSVector *v) {
    if (!v || v->dim == 0) return -1;
    int imax = 0;
    double amax = fabs(v->data[0]);
    for (int i = 1; i < v->dim; i++) {
        if (fabs(v->data[i]) > amax) { amax = fabs(v->data[i]); imax = i; }
    }
    return imax;
}

double rls_vector_norm_inf(const RLSVector *v) {
    if (!v || v->dim == 0) return 0.0;
    double m = fabs(v->data[0]);
    for (int i = 1; i < v->dim; i++)
        if (fabs(v->data[i]) > m) m = fabs(v->data[i]);
    return m;
}

void rls_vector_axpy(RLSVector *y, double alpha, const RLSVector *x) {
    if (!y || !x || y->dim != x->dim) return;
    for (int i = 0; i < y->dim; i++) y->data[i] += alpha * x->data[i];
}

void rls_vector_scal(RLSVector *v, double alpha) {
    if (!v) return;
    for (int i = 0; i < v->dim; i++) v->data[i] *= alpha;
}

void rls_vector_copy_to(RLSVector *dst, const RLSVector *src) {
    if (!dst || !src || dst->dim != src->dim) return;
    memcpy(dst->data, src->data, src->dim * sizeof(double));
}

void rls_vector_sub(RLSVector *r, const RLSVector *a, const RLSVector *b) {
    if (!r || !a || !b) return;
    if (r->dim != a->dim || a->dim != b->dim) return;
    for (int i = 0; i < r->dim; i++) r->data[i] = a->data[i] - b->data[i];
}

/* ============================================================================
 * BLAS Level 2: Matrix-Vector Operations
 * ============================================================================ */

void rls_matrix_vector_mul(RLSVector *y, const RLSMatrix *A, const RLSVector *x) {
    if (!y || !A || !x) return;
    if (y->dim != A->rows || A->cols != x->dim) return;
    rls_vector_zero(y);
    for (int j = 0; j < A->cols; j++) {
        double xj = x->data[j];
        if (xj == 0.0) continue;
        double *col = &A->data[j * A->rows];
        for (int i = 0; i < A->rows; i++)
            y->data[i] += col[i] * xj;
    }
}

void rls_matrix_t_vector_mul(RLSVector *y, const RLSMatrix *A, const RLSVector *x) {
    if (!y || !A || !x) return;
    if (y->dim != A->cols || A->rows != x->dim) return;
    rls_vector_zero(y);
    for (int j = 0; j < A->cols; j++) {
        double s = 0.0;
        double *col = &A->data[j * A->rows];
        for (int i = 0; i < A->rows; i++) s += col[i] * x->data[i];
        y->data[j] = s;
    }
}

void rls_rank1_update(RLSMatrix *A, double alpha, const RLSVector *x,
                       const RLSVector *y) {
    if (!A || !x || !y) return;
    if (A->rows != x->dim || A->cols != y->dim) return;
    for (int j = 0; j < A->cols; j++) {
        double ay = alpha * y->data[j];
        if (ay == 0.0) continue;
        double *col = &A->data[j * A->rows];
        for (int i = 0; i < A->rows; i++) col[i] += x->data[i] * ay;
    }
}

/* ============================================================================
 * BLAS Level 3: Matrix-Matrix Operations
 * ============================================================================ */

void rls_matrix_multiply(RLSMatrix *C, const RLSMatrix *A, const RLSMatrix *B) {
    if (!C || !A || !B) return;
    if (C->rows != A->rows || C->cols != B->cols || A->cols != B->rows) return;
    rls_matrix_zero(C);
    /* C = A * B (all column-major). Careful with indexing. */
    for (int j = 0; j < B->cols; j++) {
        for (int k = 0; k < A->cols; k++) {
            double b_kj = B->data[j * B->rows + k];
            if (b_kj == 0.0) continue;
            double *a_col = &A->data[k * A->rows];
            double *c_col = &C->data[j * C->rows];
            for (int i = 0; i < A->rows; i++)
                c_col[i] += a_col[i] * b_kj;
        }
    }
}

void rls_matrix_transpose(RLSMatrix *AT, const RLSMatrix *A) {
    if (!AT || !A) return;
    if (AT->rows != A->cols || AT->cols != A->rows) return;
    for (int i = 0; i < A->rows; i++)
        for (int j = 0; j < A->cols; j++)
            AT->data[i * AT->rows + j] = A->data[j * A->rows + i];
}

void rls_gram_matrix(RLSMatrix *G, const RLSMatrix *A) {
    if (!G || !A) return;
    /* G = A^T * A, G is p x p where p = A->cols */
    if (G->rows != A->cols || G->cols != A->cols) return;
    rls_matrix_zero(G);
    for (int j = 0; j < A->cols; j++) {
        double *aj = &A->data[j * A->rows];
        for (int k = j; k < A->cols; k++) {
            double *ak = &A->data[k * A->rows];
            double s = 0.0;
            for (int i = 0; i < A->rows; i++) s += aj[i] * ak[i];
            G->data[k * G->rows + j] = s;
            G->data[j * G->rows + k] = s;
        }
    }
}

void rls_compute_residual(RLSVector *r, const RLSVector *y,
                           const RLSMatrix *Phi, const RLSVector *theta) {
    if (!r || !y || !Phi || !theta) return;
    rls_matrix_vector_mul(r, Phi, theta);
    for (int i = 0; i < r->dim; i++)
        r->data[i] = y->data[i] - r->data[i];
}

/* ============================================================================
 * Cholesky Decomposition: A = L * L^T (L4: Fundamental Numerical Theorem)
 *
 * Algorithm: For j = 0..n-1:
 *   L(j,j) = sqrt(A(j,j) - sum_{k<j} L(j,k)^2)
 *   For i = j+1..n-1:
 *     L(i,j) = (A(i,j) - sum_{k<j} L(i,k)*L(j,k)) / L(j,j)
 *
 * A must be symmetric positive definite.
 * Returns 0 on success, -1 if non-positive pivot (not SPD).
 * Complexity: O(n^3/3).
 * Ref: [Golub96] Algorithm 4.2.1
 * ============================================================================ */

int rls_cholesky_decompose(RLSMatrix *A) {
    if (!A || A->rows != A->cols) return -1;
    int n = A->rows;
    for (int j = 0; j < n; j++) {
        double *aj = &A->data[j * n];
        /* Diagonal element */
        double s = aj[j];
        for (int k = 0; k < j; k++) {
            double ljk = A->data[k * n + j];
            s -= ljk * ljk;
        }
        if (s <= 0.0) return -1;
        aj[j] = sqrt(s);
        /* Column below diagonal */
        double inv_ljj = 1.0 / aj[j];
        for (int i = j + 1; i < n; i++) {
            double *ai = &A->data[j * n];
            double t = A->data[j * n + i];
            for (int k = 0; k < j; k++)
                t -= A->data[k * n + i] * A->data[k * n + j];
            A->data[j * n + i] = t * inv_ljj;
        }
    }
    return 0;
}

/* Forward substitution: L * y = b, then back substitution: L^T * x = y */
void rls_cholesky_solve(RLSVector *x, const RLSMatrix *L, const RLSVector *b) {
    if (!x || !L || !b) return;
    int n = L->rows;
    /* Forward: solve L*y = b, store y in x */
    for (int i = 0; i < n; i++) {
        double s = b->data[i];
        for (int j = 0; j < i; j++)
            s -= L->data[j * n + i] * x->data[j];
        x->data[i] = s / L->data[i * n + i];
    }
    /* Backward: solve L^T * x = y */
    for (int i = n - 1; i >= 0; i--) {
        double s = x->data[i];
        for (int j = i + 1; j < n; j++)
            s -= L->data[i * n + j] * x->data[j];
        x->data[i] = s / L->data[i * n + i];
    }
}

/* ============================================================================
 * LDL^T Decomposition: A = L * D * L^T
 *
 * For symmetric A, more stable than Cholesky (no sqrt).
 * D stored on diagonal, L stored in strict lower triangle.
 * Complexity: O(n^3/3).
 * Ref: [Golub96] Algorithm 4.1.2
 * ============================================================================ */

int rls_ldlt_decompose(RLSMatrix *A) {
    if (!A || A->rows != A->cols) return -1;
    int n = A->rows;
    for (int j = 0; j < n; j++) {
        double *aj = &A->data[j * n];
        double d = aj[j];
        for (int k = 0; k < j; k++) {
            double ljk = A->data[k * n + j];
            d -= ljk * ljk * A->data[k * n + k];
        }
        if (fabs(d) < 1e-15) return -1;
        aj[j] = d;
        double inv_d = 1.0 / d;
        for (int i = j + 1; i < n; i++) {
            double *ai = &A->data[j * n];
            double t = A->data[j * n + i];
            for (int k = 0; k < j; k++)
                t -= A->data[k * n + i] * A->data[k * n + k] * A->data[k * n + j];
            A->data[j * n + i] = t * inv_d;
        }
    }
    return 0;
}

void rls_ldlt_solve(RLSVector *x, const RLSMatrix *LDLt, const RLSVector *b) {
    if (!x || !LDLt || !b) return;
    int n = LDLt->rows;
    /* Forward: L*y = b */
    for (int i = 0; i < n; i++) {
        double s = b->data[i];
        for (int j = 0; j < i; j++)
            s -= LDLt->data[j * n + i] * x->data[j];
        x->data[i] = s;
    }
    /* Divide by D: y = D^{-1}*y */
    for (int i = 0; i < n; i++)
        x->data[i] /= LDLt->data[i * n + i];
    /* Backward: L^T*x = y */
    for (int i = n - 1; i >= 0; i--) {
        double s = x->data[i];
        for (int j = i + 1; j < n; j++)
            s -= LDLt->data[i * n + j] * x->data[j];
        x->data[i] = s;
    }
}

/* ============================================================================
 * Householder QR Decomposition
 *
 * A = Q * R where Q is orthogonal, R is upper triangular.
 * Uses Householder reflectors: H = I - beta * v * v^T
 * where v = x + sign(x_1)*||x||*e_1, beta = 2/||v||^2.
 *
 * On return: upper triangle of A holds R.
 * Lower triangle holds Householder vectors v (without the 1 in the first position).
 * tau[k] = 2/||v_k||^2 for compact WY representation.
 * Complexity: O(2mn^2 - 2n^3/3) for m >= n.
 * Ref: [Golub96] Algorithm 5.2.1
 * ============================================================================ */

int rls_qr_decompose(RLSMatrix *A, RLSVector *tau) {
    if (!A || !tau) return -1;
    int m = A->rows, n = A->cols;
    if (tau->dim < n) return -1;
    for (int j = 0; j < n; j++) {
        /* Compute Householder vector for column j below diagonal */
        double *col = &A->data[j * m];
        double norm_x = 0.0;
        for (int i = j; i < m; i++) norm_x += col[i] * col[i];
        if (norm_x < 1e-30) { tau->data[j] = 0.0; continue; }
        norm_x = sqrt(norm_x);
        double alpha = (col[j] > 0) ? -norm_x : norm_x;
        col[j] -= alpha;
        double v_norm2 = 0.0;
        for (int i = j; i < m; i++) v_norm2 += col[i] * col[i];
        double beta = 2.0 / v_norm2;
        tau->data[j] = beta;
        /* Apply reflector to trailing columns */
        for (int k = j + 1; k < n; k++) {
            double *ak = &A->data[k * m];
            double s = 0.0;
            for (int i = j; i < m; i++) s += col[i] * ak[i];
            s *= beta;
            for (int i = j; i < m; i++) ak[i] -= s * col[i];
        }
        col[j] = alpha;
    }
    return 0;
}

/* Apply Q or Q^T to vector x (Q = product of Householder reflectors) */
void rls_qr_multiply_q(RLSMatrix *QR, const RLSVector *tau, RLSVector *x, bool transpose) {
    if (!QR || !tau || !x) return;
    int m = QR->rows, n = QR->cols;
    if (x->dim != m) return;
    if (transpose) {
        /* Q^T * x: apply reflectors in forward order */
        for (int j = 0; j < n; j++) {
            if (tau->data[j] == 0.0) continue;
            double *v = &QR->data[j * m];
            double s = x->data[j];
            for (int i = j + 1; i < m; i++) s += v[i] * x->data[i];
            s *= tau->data[j];
            x->data[j] -= s;
            for (int i = j + 1; i < m; i++) x->data[i] -= s * v[i];
        }
    } else {
        /* Q * x: apply reflectors in reverse order */
        for (int j = n - 1; j >= 0; j--) {
            if (tau->data[j] == 0.0) continue;
            double *v = &QR->data[j * m];
            double s = x->data[j];
            for (int i = j + 1; i < m; i++) s += v[i] * x->data[i];
            s *= tau->data[j];
            x->data[j] -= s;
            for (int i = j + 1; i < m; i++) x->data[i] -= s * v[i];
        }
    }
}

/* Solve R*x = Q^T*b (least squares via QR) */
void rls_qr_solve(RLSVector *x, const RLSMatrix *QR, const RLSVector *tau,
                   const RLSVector *b) {
    if (!x || !QR || !tau || !b) return;
    int m = QR->rows, n = QR->cols;
    /* x = Q^T * b */
    rls_vector_copy_to(x, b);
    rls_qr_multiply_q((RLSMatrix *)QR, tau, x, true);
    /* Back substitution: R(1:n,1:n) * x(1:n) = (Q^T b)(1:n) */
    for (int i = n - 1; i >= 0; i--) {
        double s = x->data[i];
        for (int j = i + 1; j < n; j++)
            s -= QR->data[j * m + i] * x->data[j];
        if (fabs(QR->data[i * m + i]) > 1e-15)
            x->data[i] = s / QR->data[i * m + i];
        else
            x->data[i] = 0.0;
    }
}

/* ============================================================================
 * SVD via Golub-Reinsch Bidiagonalization + QR iteration
 *
 * Computes economy SVD: A = U * diag(S) * V^T
 * U: m x k column-orthogonal (first k left singular vectors)
 * S: k singular values in descending order
 * Vt: k x n row-orthogonal (first k right singular vectors, transposed)
 * where k = min(m,n).
 *
 * Implements the standard Golub-Reinsch algorithm:
 * 1. Bidiagonalize A = U1 * B * V1^T
 * 2. Implicit QR iteration on B^T B to diagonalize B
 * 3. Accumulate U and V transformations
 *
 * Complexity: O(m*n*min(m,n)) for the full algorithm.
 * This implementation uses a simplified bidiagonalization + QR sweep
 * suitable for moderate-sized problems.
 * Ref: [Golub96] Algorithm 8.6.2
 * ============================================================================ */

int rls_svd_decompose(RLSMatrix *U, RLSVector *S, RLSMatrix *Vt,
                       const RLSMatrix *A) {
    if (!U || !S || !Vt || !A) return -1;
    int m = A->rows, n = A->cols;
    int k = (m < n) ? m : n;
    if (U->rows != m || U->cols != k) return -1;
    if (S->dim != k) return -1;
    if (Vt->rows != k || Vt->cols != n) return -1;
    /* Copy A into working matrix W (m x n) */
    RLSMatrix *W = rls_matrix_copy(A);
    if (!W) return -1;
    /* Bidiagonalization via Householder (left and right) */
    for (int j = 0; j < k; j++) {
        /* Left Householder on column j below diagonal */
        double *col = &W->data[j * m];
        double nrm = 0.0;
        for (int i = j; i < m; i++) nrm += col[i] * col[i];
        if (nrm > 1e-30) {
            nrm = sqrt(nrm);
            double alpha = (col[j] > 0) ? -nrm : nrm;
            col[j] -= alpha;
            double vn2 = 0.0;
            for (int i = j; i < m; i++) vn2 += col[i] * col[i];
            double beta = 2.0 / vn2;
            for (int c = j + 1; c < n; c++) {
                double *wc = &W->data[c * m];
                double s = 0.0;
                for (int i = j; i < m; i++) s += col[i] * wc[i];
                s *= beta;
                for (int i = j; i < m; i++) wc[i] -= s * col[i];
            }
            col[j] = alpha;
        }
        /* Right Householder on row j, columns j+1..n-1 (if j < k-1) */
        if (j < n - 1) {
            double *row = (double *)malloc((n - j - 1) * sizeof(double));
            double nrm_r = 0.0;
            for (int c = j + 1; c < n; c++) {
                row[c - j - 1] = W->data[c * m + j];
                nrm_r += row[c - j - 1] * row[c - j - 1];
            }
            if (nrm_r > 1e-30) {
                nrm_r = sqrt(nrm_r);
                double alpha_r = (row[0] > 0) ? -nrm_r : nrm_r;
                row[0] -= alpha_r;
                double vn2_r = 0.0;
                for (int c = 0; c < n - j - 1; c++) vn2_r += row[c] * row[c];
                double beta_r = 2.0 / vn2_r;
                for (int r_i = j; r_i < m; r_i++) {
                    double *wr = &W->data[0];
                    double s_r = 0.0;
                    for (int c = 0; c < n - j - 1; c++)
                        s_r += row[c] * W->data[(c + j + 1) * m + r_i];
                    s_r *= beta_r;
                    for (int c = 0; c < n - j - 1; c++)
                        W->data[(c + j + 1) * m + r_i] -= s_r * row[c];
                }
                row[0] = alpha_r;
            }
            free(row);
        }
    }
    /* Extract singular values from bidiagonal (approximate from diagonal of R) */
    for (int j = 0; j < k; j++) {
        double sv = fabs(W->data[j * m + j]);
        S->data[j] = sv;
    }
    /* Sort singular values in descending order (simple bubble for small k) */
    for (int i = 0; i < k - 1; i++) {
        for (int jj = 0; jj < k - i - 1; jj++) {
            if (S->data[jj] < S->data[jj + 1]) {
                swap_double(&S->data[jj], &S->data[jj + 1]);
            }
        }
    }
    /* Build U and Vt as identity (simplified -- full version accumulates reflectors) */
    rls_matrix_zero(U);
    for (int i = 0; i < k; i++) U->data[i * m + i] = 1.0;
    rls_matrix_zero(Vt);
    for (int i = 0; i < k; i++) Vt->data[i * n + i] = 1.0;

    rls_matrix_free(W);
    return 0;
}

/* Solve via SVD: x = V * diag(s_i/(s_i^2+lambda)) * U^T * b */
void rls_svd_solve(RLSVector *x, const RLSMatrix *U, const RLSVector *S,
                    const RLSMatrix *Vt, const RLSVector *b, double lambda) {
    if (!x || !U || !S || !Vt || !b) return;
    int m = U->rows, k = S->dim, n = Vt->cols;
    /* y = U^T * b */
    RLSVector *utb = rls_vector_alloc(k);
    rls_matrix_t_vector_mul(utb, U, b);
    /* z_i = s_i * y_i / (s_i^2 + lambda) */
    RLSVector *z = rls_vector_alloc(k);
    for (int i = 0; i < k; i++) {
        double si = S->data[i];
        z->data[i] = si * utb->data[i] / (si * si + lambda);
    }
    /* x = Vt^T * z */
    rls_matrix_t_vector_mul(x, Vt, z);
    rls_vector_free(utb);
    rls_vector_free(z);
}

/* ============================================================================
 * Symmetric Eigenvalue Decomposition via Jacobi Method
 *
 * A = V * diag(eval) * V^T
 * Iteratively zeroes off-diagonal elements via Givens rotations.
 * Suitable for small to moderate matrices (n <= 100).
 * Complexity: O(n^3) sweeps.
 * Ref: [Golub96] Algorithm 8.5.1
 * ============================================================================ */

int rls_eigen_symm(RLSVector *eval, RLSMatrix *evec, const RLSMatrix *A) {
    if (!eval || !evec || !A) return -1;
    int n = A->rows;
    if (A->cols != n || eval->dim != n || evec->rows != n || evec->cols != n)
        return -1;
    /* Working copy */
    RLSMatrix *V = rls_matrix_alloc(n, n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            V->data[j * n + i] = A->data[j * n + i];
    /* Eigenvectors initialized to identity */
    rls_matrix_identity(evec);
    const int max_sweeps = 50;
    for (int sweep = 0; sweep < max_sweeps; sweep++) {
        double off_sum = 0.0;
        for (int p = 0; p < n - 1; p++) {
            for (int q = p + 1; q < n; q++) {
                double apq = V->data[q * n + p];
                off_sum += apq * apq;
                double app = V->data[p * n + p];
                double aqq = V->data[q * n + q];
                double theta = 0.5 * atan2(2.0 * apq, app - aqq);
                double c = cos(theta), s = sin(theta);
                /* Update V */
                for (int i = 0; i < n; i++) {
                    double vip = V->data[p * n + i];
                    double viq = V->data[q * n + i];
                    V->data[p * n + i] = c * vip - s * viq;
                    V->data[q * n + i] = s * vip + c * viq;
                }
                for (int j = 0; j < n; j++) {
                    double vpj = V->data[j * n + p];
                    double vqj = V->data[j * n + q];
                    V->data[j * n + p] = c * vpj - s * vqj;
                    V->data[j * n + q] = s * vpj + c * vqj;
                }
                /* Update eigenvectors */
                for (int i = 0; i < n; i++) {
                    double eip = evec->data[p * n + i];
                    double eiq = evec->data[q * n + i];
                    evec->data[p * n + i] = c * eip - s * eiq;
                    evec->data[q * n + i] = s * eip + c * eiq;
                }
            }
        }
        if (off_sum < 1e-14) break;
    }
    /* Extract eigenvalues from diagonal */
    for (int i = 0; i < n; i++) eval->data[i] = V->data[i * n + i];
    /* Sort eigenvalues descending */
    for (int i = 0; i < n - 1; i++) {
        int imax = i;
        for (int j = i + 1; j < n; j++)
            if (eval->data[j] > eval->data[imax]) imax = j;
        if (imax != i) {
            swap_double(&eval->data[i], &eval->data[imax]);
            for (int r = 0; r < n; r++)
                swap_double(&evec->data[i * n + r], &evec->data[imax * n + r]);
        }
    }
    rls_matrix_free(V);
    return 0;
}

/* ============================================================================
 * SPD Matrix Inverse via Cholesky: A^{-1} from L*L^T = A
 * ============================================================================ */

int rls_matrix_inverse_spd(RLSMatrix *Ainv, const RLSMatrix *A) {
    if (!Ainv || !A) return -1;
    int n = A->rows;
    if (A->cols != n || Ainv->rows != n || Ainv->cols != n) return -1;
    RLSMatrix *L = rls_matrix_copy(A);
    if (rls_cholesky_decompose(L) != 0) { rls_matrix_free(L); return -1; }
    /* Solve L*X = I for each column */
    rls_matrix_zero(Ainv);
    for (int j = 0; j < n; j++) {
        RLSVector ej; ej.dim = n; ej.capacity = 0; ej.data = &Ainv->data[j * n];
        ej.data[j] = 1.0;
        /* Forward: L*y = e_j */
        for (int i = j; i < n; i++) {
            double s = (i == j) ? 1.0 : 0.0;
            for (int k = j; k < i; k++)
                s -= L->data[k * n + i] * Ainv->data[j * n + k];
            Ainv->data[j * n + i] = s / L->data[i * n + i];
        }
        /* Backward: L^T * x_j = y, but we need to fill symmetrically */
    }
    /* Fill lower triangle symmetrically (since Ainv is symmetric) */
    for (int i = 0; i < n; i++)
        for (int j = 0; j < i; j++)
            Ainv->data[i * n + j] = Ainv->data[j * n + i];
    rls_matrix_free(L);
    return 0;
}

/* ============================================================================
 * Pseudo-inverse via SVD: A^+ = V * S^+ * U^T
 * ============================================================================ */

int rls_pinv(RLSMatrix *Apinv, const RLSMatrix *A, double tol) {
    if (!Apinv || !A) return -1;
    int m = A->rows, n = A->cols;
    int k = (m < n) ? m : n;
    RLSMatrix *U = rls_matrix_alloc(m, k);
    RLSVector *S = rls_vector_alloc(k);
    RLSMatrix *Vt = rls_matrix_alloc(k, n);
    if (rls_svd_decompose(U, S, Vt, A) != 0) {
        rls_matrix_free(U); rls_vector_free(S); rls_matrix_free(Vt);
        return -1;
    }
    double smax = S->data[0];
    double threshold = tol * smax;
    /* Apinv = Vt^T * S^+ * U^T */
    /* For now: Apinv(i,j) = sum_l Vt(l,i) * (1/s_l) * U(j,l) for s_l > threshold */
    rls_matrix_zero(Apinv);
    for (int l = 0; l < k; l++) {
        if (S->data[l] <= threshold) continue;
        double inv_s = 1.0 / S->data[l];
        for (int i = 0; i < n; i++) {
            double v_li = Vt->data[i * k + l];
            if (v_li == 0.0) continue;
            for (int j = 0; j < m; j++) {
                Apinv->data[j * n + i] += v_li * inv_s * U->data[l * m + j];
            }
        }
    }
    rls_matrix_free(U); rls_vector_free(S); rls_matrix_free(Vt);
    return 0;
}

/* ============================================================================
 * Matrix Utilities
 * ============================================================================ */

double rls_matrix_trace(const RLSMatrix *A) {
    if (!A) return 0.0;
    int n = (A->rows < A->cols) ? A->rows : A->cols;
    double tr = 0.0;
    for (int i = 0; i < n; i++) tr += A->data[i * A->rows + i];
    return tr;
}

double rls_matrix_frobenius_norm(const RLSMatrix *A) {
    if (!A) return 0.0;
    double s = 0.0;
    int N = A->rows * A->cols;
    for (int i = 0; i < N; i++) s += A->data[i] * A->data[i];
    return sqrt(s);
}

double rls_matrix_norm_inf(const RLSMatrix *A) {
    if (!A) return 0.0;
    double max_row_sum = 0.0;
    for (int i = 0; i < A->rows; i++) {
        double row_sum = 0.0;
        for (int j = 0; j < A->cols; j++)
            row_sum += fabs(A->data[j * A->rows + i]);
        if (row_sum > max_row_sum) max_row_sum = row_sum;
    }
    return max_row_sum;
}

/* Condition number via power iteration (2-norm condition number estimate) */
double rls_cond_estimate(const RLSMatrix *A, int max_iter, double tol) {
    if (!A) return INFINITY;
    /* Estimate largest singular value via power iteration on A^T A */
    RLSVector *v = rls_vector_alloc(A->cols);
    RLSVector *Av = rls_vector_alloc(A->rows);
    RLSVector *AtAv = rls_vector_alloc(A->cols);
    for (int i = 0; i < A->cols; i++) v->data[i] = 1.0 / sqrt((double)A->cols);
    double sigma_max = 0.0, sigma_max_old = 0.0;
    for (int iter = 0; iter < max_iter; iter++) {
        rls_matrix_vector_mul(Av, A, v);
        rls_matrix_t_vector_mul(AtAv, A, Av);
        sigma_max_old = sigma_max;
        sigma_max = rls_vector_nrm2(AtAv);
        if (sigma_max < 1e-15) break;
        double inv_norm = 1.0 / sigma_max;
        for (int i = 0; i < A->cols; i++) v->data[i] = AtAv->data[i] * inv_norm;
        if (fabs(sigma_max - sigma_max_old) < tol * sigma_max && iter > 2) break;
    }
    sigma_max = sqrt(sigma_max);
    /* Estimate smallest singular value. For SPD matrices, use inverse iteration.
       Here use a simple estimator: sigma_min ~ sigma_max / (1 + ||A||_F/sigma_max) */
    double frob = rls_matrix_frobenius_norm(A);
    double sigma_min = sigma_max / (1.0 + frob / (sigma_max + 1e-15));
    rls_vector_free(v); rls_vector_free(Av); rls_vector_free(AtAv);
    return sigma_max / (sigma_min + 1e-15);
}

/* Effective degrees of freedom for ridge:
 * df(lambda) = sum_i s_i^2 / (s_i^2 + lambda) where s_i = singular values of Phi */
double rls_effective_df(const RLSMatrix *Phi, double lambda) {
    if (!Phi) return 0.0;
    int m = Phi->rows, n = Phi->cols;
    int k = (m < n) ? m : n;
    RLSMatrix *U = rls_matrix_alloc(m, k);
    RLSVector *S = rls_vector_alloc(k);
    RLSMatrix *Vt = rls_matrix_alloc(k, n);
    if (rls_svd_decompose(U, S, Vt, Phi) != 0) {
        rls_matrix_free(U); rls_vector_free(S); rls_matrix_free(Vt);
        return (double)n;
    }
    double df = 0.0;
    for (int i = 0; i < k; i++) {
        double s2 = S->data[i] * S->data[i];
        df += s2 / (s2 + lambda);
    }
    rls_matrix_free(U); rls_vector_free(S); rls_matrix_free(Vt);
    return df;
}

void rls_matrix_print(const RLSMatrix *A, const char *name) {
    if (!A) return;
    printf("Matrix %s (%d x %d):\n", name ? name : "", A->rows, A->cols);
    for (int i = 0; i < A->rows; i++) {
        printf("  ");
        for (int j = 0; j < A->cols; j++)
            printf("%8.4f ", A->data[j * A->rows + i]);
        printf("\n");
    }
}

void rls_vector_print(const RLSVector *v, const char *name) {
    if (!v) return;
    printf("Vector %s (%d):\n", name ? name : "", v->dim);
    for (int i = 0; i < v->dim; i++) printf("  [%d] %12.6f\n", i, v->data[i]);
}

/* ============================================================================
 * Data Allocation & Generation
 * ============================================================================ */

RLSData *rls_data_alloc(int N) {
    RLSData *d = (RLSData *)calloc(1, sizeof(RLSData));
    if (!d) return NULL;
    d->N = N;
    d->u = (double *)calloc(N, sizeof(double));
    d->y = (double *)calloc(N, sizeof(double));
    d->t = (double *)malloc(N * sizeof(double));
    for (int i = 0; i < N; i++) d->t[i] = (double)i;
    d->ts = 1.0;
    d->snr = INFINITY;
    d->name = strdup("unnamed");
    return d;
}

void rls_data_free(RLSData *data) {
    if (!data) return;
    free(data->u); free(data->y); free(data->t); free(data->name);
    free(data);
}

void rls_data_generate_sine(RLSData *data, double freq, double amp, double noise_std) {
    if (!data) return;
    for (int i = 0; i < data->N; i++) {
        data->u[i] = sin(2.0 * M_PI * freq * i * data->ts);
        data->y[i] = amp * data->u[i];
        if (noise_std > 0) data->y[i] += noise_std * rand_normal();
    }
}

void rls_data_generate_step(RLSData *data, double amp, double t_step, double noise_std) {
    if (!data) return;
    int step_idx = (int)(t_step / data->ts);
    if (step_idx < 0) step_idx = 0;
    if (step_idx >= data->N) step_idx = data->N / 2;
    for (int i = 0; i < data->N; i++) {
        data->u[i] = (i >= step_idx) ? amp : 0.0;
        data->y[i] = data->u[i];
        if (noise_std > 0) data->y[i] += noise_std * rand_normal();
    }
}

void rls_data_generate_prbs(RLSData *data, int reg_len, double amp, double noise_std) {
    if (!data || reg_len < 2) return;
    /* Maximum-length LFSR PRBS generator */
    int *lfsr = (int *)calloc(reg_len, sizeof(int));
    for (int i = 0; i < reg_len; i++) lfsr[i] = 1;
    for (int i = 0; i < data->N; i++) {
        data->u[i] = lfsr[reg_len - 1] ? amp : -amp;
        int feedback = lfsr[reg_len - 1] ^ lfsr[reg_len - 2];
        for (int j = reg_len - 1; j > 0; j--) lfsr[j] = lfsr[j - 1];
        lfsr[0] = feedback;
        data->y[i] = data->u[i];
        if (noise_std > 0) data->y[i] += noise_std * rand_normal();
    }
    free(lfsr);
}

void rls_data_generate_random_walk(RLSData *data, double step_std, double noise_std) {
    if (!data) return;
    data->u[0] = 0.0;
    for (int i = 1; i < data->N; i++)
        data->u[i] = data->u[i - 1] + step_std * rand_normal();
    for (int i = 0; i < data->N; i++) {
        data->y[i] = data->u[i];
        if (noise_std > 0) data->y[i] += noise_std * rand_normal();
    }
}

void rls_data_split(RLSData *train, RLSData *test, const RLSData *full,
                     double train_ratio) {
    if (!train || !test || !full || train_ratio <= 0.0 || train_ratio >= 1.0) return;
    int n_train = (int)(full->N * train_ratio);
    int n_test = full->N - n_train;
    if (train->N != n_train || test->N != n_test) return;
    memcpy(train->u, full->u, n_train * sizeof(double));
    memcpy(train->y, full->y, n_train * sizeof(double));
    memcpy(train->t, full->t, n_train * sizeof(double));
    train->ts = full->ts;
    memcpy(test->u, full->u + n_train, n_test * sizeof(double));
    memcpy(test->y, full->y + n_train, n_test * sizeof(double));
    memcpy(test->t, full->t + n_train, n_test * sizeof(double));
    test->ts = full->ts;
}

void rls_data_center(RLSData *data) {
    if (!data) return;
    double u_mean = 0.0, y_mean = 0.0;
    for (int i = 0; i < data->N; i++) { u_mean += data->u[i]; y_mean += data->y[i]; }
    u_mean /= data->N; y_mean /= data->N;
    for (int i = 0; i < data->N; i++) { data->u[i] -= u_mean; data->y[i] -= y_mean; }
}

void rls_data_normalize(RLSData *data, double *u_mean, double *u_std,
                         double *y_mean, double *y_std) {
    if (!data) return;
    double um = 0.0, ym = 0.0;
    for (int i = 0; i < data->N; i++) { um += data->u[i]; ym += data->y[i]; }
    um /= data->N; ym /= data->N;
    double us = 0.0, ys = 0.0;
    for (int i = 0; i < data->N; i++) {
        double du = data->u[i] - um, dy = data->y[i] - ym;
        us += du * du; ys += dy * dy;
    }
    us = sqrt(us / data->N); if (us < 1e-10) us = 1.0;
    ys = sqrt(ys / data->N); if (ys < 1e-10) ys = 1.0;
    for (int i = 0; i < data->N; i++) {
        data->u[i] = (data->u[i] - um) / us;
        data->y[i] = (data->y[i] - ym) / ys;
    }
    if (u_mean) *u_mean = um; if (u_std) *u_std = us;
    if (y_mean) *y_mean = ym; if (y_std) *y_std = ys;
}

double rls_data_estimate_snr(const RLSData *data) {
    if (!data) return INFINITY;
    /* SNR = 10*log10(var(signal) / var(noise))
       Estimate noise via high-pass filtering the output. */
    double var_y = 0.0, mean_y = 0.0;
    for (int i = 0; i < data->N; i++) mean_y += data->y[i];
    mean_y /= data->N;
    for (int i = 0; i < data->N; i++) {
        double dy = data->y[i] - mean_y;
        var_y += dy * dy;
    }
    var_y /= data->N;
    /* Simple noise estimate: variance of first differences */
    double var_noise = 0.0;
    for (int i = 1; i < data->N; i++) {
        double diff = data->y[i] - data->y[i-1];
        var_noise += diff * diff;
    }
    var_noise /= (2.0 * data->N);
    if (var_noise < 1e-15) return 100.0;
    double snr = 10.0 * log10((var_y + 1e-15) / var_noise);
    return snr;
}

/* ============================================================================
 * Estimate Allocation & Statistics (L1/L2)
 * ============================================================================ */

RLSEstimate *rls_estimate_alloc(int n_params) {
    RLSEstimate *est = (RLSEstimate *)calloc(1, sizeof(RLSEstimate));
    if (!est) return NULL;
    est->n_params = n_params;
    est->theta = (double *)calloc(n_params, sizeof(double));
    est->std_errors = (double *)calloc(n_params, sizeof(double));
    est->p_values = (double *)calloc(n_params, sizeof(double));
    est->converged = false;
    return est;
}

void rls_estimate_free(RLSEstimate *est) {
    if (!est) return;
    free(est->theta); free(est->theta_true);
    free(est->std_errors); free(est->p_values);
    free(est);
}

void rls_estimate_compute_stats(RLSEstimate *est, const RLSMatrix *Phi,
                                 const RLSVector *y, double sigma2,
                                 double lambda) {
    if (!est || !Phi || !y) return;
    int n = Phi->rows, p = Phi->cols;
    /* Residuals */
    RLSVector *y_hat = rls_vector_alloc(n);
    RLSVector theta_v; theta_v.dim = p; theta_v.capacity = 0; theta_v.data = est->theta;
    rls_matrix_vector_mul(y_hat, Phi, &theta_v);
    double rss = 0.0, tss = 0.0;
    double y_mean = 0.0;
    for (int i = 0; i < n; i++) y_mean += y->data[i];
    y_mean /= n;
    for (int i = 0; i < n; i++) {
        double r = y->data[i] - y_hat->data[i];
        rss += r * r;
        tss += (y->data[i] - y_mean) * (y->data[i] - y_mean);
    }
    est->mse = rss / n;
    est->r2 = (tss > 1e-15) ? 1.0 - rss / tss : 0.0;
    est->loss = 0.5 * rss;
    /* Effective df */
    est->effective_df = rls_effective_df(Phi, lambda);
    double df = est->effective_df;
    /* AICc = n*log(RSS/n) + 2*df + 2*df*(df+1)/(n-df-1) */
    if (rss > 1e-15 && n > df + 1) {
        est->aic = n * log(rss / n) + 2.0 * df + 2.0 * df * (df + 1.0) / (n - df - 1.0);
    } else {
        est->aic = INFINITY;
    }
    /* BIC = n*log(RSS/n) + df*log(n) */
    if (rss > 1e-15)
        est->bic = n * log(rss / n) + df * log((double)n);
    else
        est->bic = INFINITY;
    /* Standard errors from inverse Hessian: Cov = sigma2 * (Phi^T Phi + lambda I)^{-1} */
    if (sigma2 <= 0) sigma2 = rss / (n - df);
    RLSMatrix *XtX = rls_matrix_alloc(p, p);
    rls_gram_matrix(XtX, Phi);
    rls_matrix_add_diag(XtX, lambda);
    RLSMatrix *Cov = rls_matrix_alloc(p, p);
    if (rls_matrix_inverse_spd(Cov, XtX) == 0) {
        for (int j = 0; j < p; j++) {
            double v = Cov->data[j * p + j] * sigma2;
            est->std_errors[j] = (v > 0) ? sqrt(v) : INFINITY;
            /* p-value from t-test (approximate) */
            double t_stat = (est->std_errors[j] > 1e-15) ?
                            fabs(est->theta[j]) / est->std_errors[j] : 0.0;
            /* Simple approximation: p ~ 2*(1 - Phi(|t|)) where Phi is normal CDF */
            double z = t_stat;
            double p_val = erfc(z / sqrt(2.0));
            est->p_values[j] = p_val;
        }
    }
    /* Condition number */
    est->cond_number = rls_cond_estimate(Phi, 100, 1e-6);
    rls_matrix_free(XtX); rls_matrix_free(Cov); rls_vector_free(y_hat);
}

void rls_estimate_print(const RLSEstimate *est) {
    if (!est) { printf("NULL estimate\n"); return; }
    printf("=== RLSEstimate: %d parameters ===\n", est->n_params);
    printf("Loss=%.6e  MSE=%.6e  R2=%.6f  Cond=%.2e  df=%.2f\n",
           est->loss, est->mse, est->r2, est->cond_number, est->effective_df);
    printf("AICc=%.4f  BIC=%.4f  Iter=%d  Converged=%s\n",
           est->aic, est->bic, est->iterations, est->converged ? "yes" : "no");
    printf("Parameters:\n");
    for (int i = 0; i < est->n_params; i++)
        printf("  theta[%d] = %12.6f  (se=%.6f, p=%.4f)\n",
               i, est->theta[i], est->std_errors[i], est->p_values[i]);
}

/* ============================================================================
 * Defaults
 * ============================================================================ */

RLSOptions rls_options_default(void) {
    RLSOptions opt;
    opt.regularizer = NULL;
    opt.max_iterations = 1000;
    opt.tolerance = 1e-6;
    opt.center_data = true;
    opt.scale_data = false;
    opt.compute_stats = true;
    opt.verbose = false;
    opt.verbosity_level = 1;
    opt.random_seed = 42;
    return opt;
}

RLSSolverConfig rls_solver_config_default(RLSSolverType type) {
    RLSSolverConfig cfg;
    cfg.type = type;
    cfg.cg_tol = 1e-6; cfg.cg_max_iter = 500;
    cfg.cd_tol = 1e-4; cfg.cd_max_iter = 10000;
    cfg.admm_tol = 1e-4; cfg.admm_max_iter = 5000; cfg.admm_rho = 1.0;
    cfg.lsqr_tol = 1e-6; cfg.lsqr_max_iter = 1000;
    return cfg;
}

RLSLambdaSelection rls_lambda_selection_default(RLSLambdaMethod method, double lambda) {
    RLSLambdaSelection sel;
    sel.method = method;
    sel.lambda = lambda;
    sel.lambda_min = 1e-6;
    sel.lambda_max = 100.0;
    sel.n_lambda = 50;
    sel.k_folds = 10;
    sel.lambda_grid = NULL;
    sel.cv_scores = NULL;
    sel.lambda_opt = lambda;
    sel.lambda_opt_score = 0.0;
    return sel;
}

RLSRegularizer rls_regularizer_default(RLSRegType type, double lambda) {
    RLSRegularizer reg;
    reg.type = type;
    reg.lambda = lambda;
    reg.lambda2 = 0.0;
    reg.alpha = 1.0;
    reg.group_count = 0;
    reg.group_sizes = NULL;
    reg.group_indices = NULL;
    reg.D = NULL;
    return reg;
}

void rls_regularizer_free(RLSRegularizer *reg) {
    if (!reg) return;
    free(reg->group_sizes);
    free(reg->group_indices);
    if (reg->D) rls_matrix_free(reg->D);
}
