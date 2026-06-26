#include "subspace_core.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ============================================================================
 * Subspace Identification -- Core Implementation
 *
 * Implements memory management, basic matrix operations, and utility
 * functions for the subspace identification library.
 * ============================================================================ */

/* ============================================================================
 * Memory Management
 * ============================================================================ */

SubspaceData* subspace_data_alloc(int N, int n_inputs, int n_outputs) {
    SubspaceData *data = (SubspaceData*)calloc(1, sizeof(SubspaceData));
    if (!data) return NULL;
    data->N = N;
    data->n_inputs = n_inputs;
    data->n_outputs = n_outputs;
    data->Ts = 1.0;
    size_t u_size = (size_t)N * (size_t)n_inputs;
    size_t y_size = (size_t)N * (size_t)n_outputs;
    data->u = (double*)calloc(u_size, sizeof(double));
    data->y = (double*)calloc(y_size, sizeof(double));
    if ((u_size > 0 && !data->u) || (y_size > 0 && !data->y)) {
        subspace_data_free(data);
        return NULL;
    }
    return data;
}

void subspace_data_free(SubspaceData *data) {
    if (!data) return;
    free(data->u);
    free(data->y);
    free(data->name);
    free(data);
}

SubspaceMatrix* subspace_matrix_alloc(int rows, int cols) {
    if (rows <= 0 || cols <= 0) return NULL;
    SubspaceMatrix *mat = (SubspaceMatrix*)malloc(sizeof(SubspaceMatrix));
    if (!mat) return NULL;
    mat->rows = rows;
    mat->cols = cols;
    mat->data = (double*)calloc((size_t)rows * (size_t)cols, sizeof(double));
    if (!mat->data) {
        free(mat);
        return NULL;
    }
    return mat;
}

void subspace_matrix_free(SubspaceMatrix *mat) {
    if (!mat) return;
    free(mat->data);
    free(mat);
}

SubspaceHankel* subspace_hankel_alloc(int block_rows, int block_cols, int dim_per_block) {
    SubspaceHankel *H = (SubspaceHankel*)malloc(sizeof(SubspaceHankel));
    if (!H) return NULL;
    H->block_rows = block_rows;
    H->block_cols = block_cols;
    H->dim_per_block = dim_per_block;
    H->total_rows = block_rows * dim_per_block;
    H->total_cols = block_cols;
    H->data = (double*)calloc((size_t)H->total_rows * (size_t)H->total_cols, sizeof(double));
    if (!H->data) { free(H); return NULL; }
    return H;
}

void subspace_hankel_free(SubspaceHankel *H) {
    if (!H) return;
    free(H->data);
    free(H);
}

SubspaceModel* subspace_model_alloc(int n, int r, int m) {
    SubspaceModel *model = (SubspaceModel*)calloc(1, sizeof(SubspaceModel));
    if (!model) return NULL;
    model->n = n;
    model->r = r;
    model->m = m;
    model->A = (double*)calloc((size_t)n * (size_t)n, sizeof(double));
    model->B = (double*)calloc((size_t)n * (size_t)r, sizeof(double));
    model->C = (double*)calloc((size_t)m * (size_t)n, sizeof(double));
    model->D = (double*)calloc((size_t)m * (size_t)r, sizeof(double));
    model->K = (double*)calloc((size_t)n * (size_t)m, sizeof(double));
    model->x0 = (double*)calloc((size_t)n, sizeof(double));
    model->stability = SS_NOT_CHECKED;
    model->Ts = 1.0;
    if ((n > 0 && !model->A) || (n > 0 && r > 0 && !model->B) ||
        (m > 0 && n > 0 && !model->C) || (m > 0 && r > 0 && !model->D)) {
        subspace_model_free(model);
        return NULL;
    }
    return model;
}

void subspace_model_free(SubspaceModel *model) {
    if (!model) return;
    free(model->A); free(model->B); free(model->C);
    free(model->D); free(model->K); free(model->x0);
    free(model);
}

SubspaceResult* subspace_result_alloc(void) {
    SubspaceResult *r = (SubspaceResult*)calloc(1, sizeof(SubspaceResult));
    if (!r) return NULL;
    r->status_msg[0] = '\0';
    return r;
}

void subspace_result_free(SubspaceResult *result) {
    if (!result) return;
    if (result->model) subspace_model_free(result->model);
    free(result->singular_values);
    free(result->eigenvalues);
    free(result);
}

SubspaceSVD* subspace_svd_alloc(int m, int n) {
    SubspaceSVD *svd = (SubspaceSVD*)calloc(1, sizeof(SubspaceSVD));
    if (!svd) return NULL;
    int mn_min = (m < n) ? m : n;
    svd->n = mn_min;
    svd->S = (double*)calloc((size_t)mn_min, sizeof(double));
    svd->U = subspace_matrix_alloc(m, mn_min);
    svd->V = subspace_matrix_alloc(n, mn_min);
    if (!svd->S || !svd->U || !svd->V) {
        subspace_svd_free(svd);
        return NULL;
    }
    return svd;
}

void subspace_svd_free(SubspaceSVD *svd) {
    if (!svd) return;
    free(svd->S);
    subspace_matrix_free(svd->U);
    subspace_matrix_free(svd->V);
    free(svd);
}

SubspaceOptions subspace_options_default(void) {
    SubspaceOptions opts;
    opts.i = 10;
    opts.max_order = 20;
    opts.algorithm = SS_N4SID;
    opts.weighting = SS_WGT_N4SID;
    opts.order_crit = SS_ORDER_SVD_GAP;
    opts.sv_threshold = 0.01;
    opts.estimate_D = true;
    opts.estimate_K = true;
    opts.enforce_stability = false;
    opts.verbose = false;
    opts.past_horizon_mult = 1;
    return opts;
}

/* ============================================================================
 * Matrix Element Access (Column-Major)
 * ============================================================================ */

void subspace_matrix_set(SubspaceMatrix *mat, int i, int j, double val) {
    if (!mat || !mat->data) return;
    if (i < 0 || i >= mat->rows || j < 0 || j >= mat->cols) return;
    mat->data[(size_t)j * (size_t)mat->rows + (size_t)i] = val;
}

double subspace_matrix_get(const SubspaceMatrix *mat, int i, int j) {
    if (!mat || !mat->data) return 0.0;
    if (i < 0 || i >= mat->rows || j < 0 || j >= mat->cols) return 0.0;
    return mat->data[(size_t)j * (size_t)mat->rows + (size_t)i];
}

void subspace_matrix_fill(SubspaceMatrix *mat, double val) {
    if (!mat || !mat->data) return;
    size_t total = (size_t)mat->rows * (size_t)mat->cols;
    for (size_t k = 0; k < total; k++) mat->data[k] = val;
}

void subspace_matrix_copy(const SubspaceMatrix *src, SubspaceMatrix *dst) {
    if (!src || !dst || !src->data || !dst->data) return;
    int min_rows = (src->rows < dst->rows) ? src->rows : dst->rows;
    int min_cols = (src->cols < dst->cols) ? src->cols : dst->cols;
    for (int j = 0; j < min_cols; j++) {
        for (int i = 0; i < min_rows; i++) {
            subspace_matrix_set(dst, i, j, subspace_matrix_get(src, i, j));
        }
    }
}

void subspace_matrix_transpose(const SubspaceMatrix *src, SubspaceMatrix *dst) {
    if (!src || !dst || !src->data || !dst->data) return;
    if (dst->rows < src->cols || dst->cols < src->rows) return;
    for (int i = 0; i < src->rows; i++) {
        for (int j = 0; j < src->cols; j++) {
            subspace_matrix_set(dst, j, i, subspace_matrix_get(src, i, j));
        }
    }
}

/* ============================================================================
 * Matrix Arithmetic
 * ============================================================================ */

void subspace_matrix_multiply(const SubspaceMatrix *A, const SubspaceMatrix *B,
                               SubspaceMatrix *C) {
    if (!A || !B || !C || !A->data || !B->data || !C->data) return;
    int m = A->rows, k = A->cols, n = B->cols;
    if (B->rows != k || C->rows != m || C->cols != n) return;
    subspace_matrix_fill(C, 0.0);
    for (int j = 0; j < n; j++) {
        for (int p = 0; p < k; p++) {
            double b_pj = subspace_matrix_get(B, p, j);
            if (fabs(b_pj) < 1e-30) continue;
            for (int i = 0; i < m; i++) {
                double a_ip = subspace_matrix_get(A, i, p);
                double old = subspace_matrix_get(C, i, j);
                subspace_matrix_set(C, i, j, old + a_ip * b_pj);
            }
        }
    }
}

void subspace_matrix_add(const SubspaceMatrix *A, const SubspaceMatrix *B,
                          SubspaceMatrix *C) {
    if (!A || !B || !C || !A->data || !B->data || !C->data) return;
    if (A->rows != B->rows || A->cols != B->cols ||
        A->rows != C->rows || A->cols != C->cols) return;
    size_t total = (size_t)A->rows * (size_t)A->cols;
    for (size_t k = 0; k < total; k++) {
        C->data[k] = A->data[k] + B->data[k];
    }
}

void subspace_matrix_subtract(const SubspaceMatrix *A, const SubspaceMatrix *B,
                               SubspaceMatrix *C) {
    if (!A || !B || !C || !A->data || !B->data || !C->data) return;
    if (A->rows != B->rows || A->cols != B->cols ||
        A->rows != C->rows || A->cols != C->cols) return;
    size_t total = (size_t)A->rows * (size_t)A->cols;
    for (size_t k = 0; k < total; k++) {
        C->data[k] = A->data[k] - B->data[k];
    }
}

double subspace_matrix_norm_frobenius(const SubspaceMatrix *mat) {
    if (!mat || !mat->data) return 0.0;
    double sum_sq = 0.0;
    size_t total = (size_t)mat->rows * (size_t)mat->cols;
    for (size_t k = 0; k < total; k++) sum_sq += mat->data[k] * mat->data[k];
    return sqrt(sum_sq);
}

double subspace_matrix_trace(const SubspaceMatrix *mat) {
    if (!mat || !mat->data) return 0.0;
    int n = (mat->rows < mat->cols) ? mat->rows : mat->cols;
    double trace = 0.0;
    for (int i = 0; i < n; i++)
        trace += subspace_matrix_get(mat, i, i);
    return trace;
}

void subspace_matrix_identity(SubspaceMatrix *mat) {
    if (!mat || !mat->data) return;
    subspace_matrix_fill(mat, 0.0);
    int n = (mat->rows < mat->cols) ? mat->rows : mat->cols;
    for (int i = 0; i < n; i++)
        subspace_matrix_set(mat, i, i, 1.0);
}

void subspace_matrix_diag(SubspaceMatrix *mat, const double *diag, int n) {
    if (!mat || !mat->data || !diag) return;
    subspace_matrix_fill(mat, 0.0);
    int mn = (mat->rows < mat->cols) ? mat->rows : mat->cols;
    if (n > mn) n = mn;
    for (int i = 0; i < n; i++)
        subspace_matrix_set(mat, i, i, diag[i]);
}

/* ============================================================================
 * Hankel Matrix Construction from IO Data
 * ============================================================================ */

void subspace_hankel_from_signal(const double *signal, int N,
                                  int dim_per_block, int block_rows,
                                  SubspaceHankel *H) {
    if (!signal || !H || !H->data || N <= 0) return;
    int block_cols = H->block_cols;
    int total_cols = block_cols;
    for (int br = 0; br < block_rows; br++) {
        for (int ch = 0; ch < dim_per_block; ch++) {
            int row = br * dim_per_block + ch;
            for (int j = 0; j < total_cols; j++) {
                int t = br + j;
                if (t < N) {
                    H->data[(size_t)row * (size_t)total_cols + (size_t)j] =
                        signal[(size_t)t * (size_t)dim_per_block + (size_t)ch];
                }
            }
        }
    }
}

void subspace_hankel_from_io_data(const SubspaceData *data, int i,
                                   SubspaceHankel *Up, SubspaceHankel *Uf,
                                   SubspaceHankel *Yp, SubspaceHankel *Yf) {
    if (!data || i <= 0) return;
    int N = data->N, r = data->n_inputs, m = data->n_outputs;
    int total_blocks = 2 * i;
    int block_cols = N - total_blocks + 1;
    if (block_cols <= 0) return;

    SubspaceHankel *HU = subspace_hankel_alloc(total_blocks, block_cols, r);
    SubspaceHankel *HY = subspace_hankel_alloc(total_blocks, block_cols, m);
    if (!HU || !HY) {
        subspace_hankel_free(HU);
        subspace_hankel_free(HY);
        return;
    }
    subspace_hankel_from_signal(data->u, N, r, total_blocks, HU);
    subspace_hankel_from_signal(data->y, N, m, total_blocks, HY);

    size_t col_size = (size_t)block_cols;
    for (int br = 0; br < i; br++) {
        for (int ch = 0; ch < r; ch++) {
            int src = br * r + ch, dst = br * r + ch;
            for (int j = 0; j < block_cols; j++)
                Up->data[(size_t)dst * col_size + (size_t)j] =
                    HU->data[(size_t)src * col_size + (size_t)j];
        }
    }
    for (int br = 0; br < i; br++) {
        for (int ch = 0; ch < r; ch++) {
            int src = (i + br) * r + ch, dst = br * r + ch;
            for (int j = 0; j < block_cols; j++)
                Uf->data[(size_t)dst * col_size + (size_t)j] =
                    HU->data[(size_t)src * col_size + (size_t)j];
        }
    }
    for (int br = 0; br < i; br++) {
        for (int ch = 0; ch < m; ch++) {
            int src = br * m + ch, dst = br * m + ch;
            for (int j = 0; j < block_cols; j++)
                Yp->data[(size_t)dst * col_size + (size_t)j] =
                    HY->data[(size_t)src * col_size + (size_t)j];
        }
    }
    for (int br = 0; br < i; br++) {
        for (int ch = 0; ch < m; ch++) {
            int src = (i + br) * m + ch, dst = br * m + ch;
            for (int j = 0; j < block_cols; j++)
                Yf->data[(size_t)dst * col_size + (size_t)j] =
                    HY->data[(size_t)src * col_size + (size_t)j];
        }
    }
    subspace_hankel_free(HU);
    subspace_hankel_free(HY);
}

void subspace_hankel_to_matrix(const SubspaceHankel *H, SubspaceMatrix *M) {
    if (!H || !M || !H->data || !M->data) return;
    if (M->rows < H->total_rows || M->cols < H->total_cols) return;
    for (int i = 0; i < H->total_rows; i++)
        for (int j = 0; j < H->total_cols; j++)
            subspace_matrix_set(M, i, j,
                H->data[(size_t)i * (size_t)H->total_cols + (size_t)j]);
}

void subspace_matrix_to_hankel(const SubspaceMatrix *M, int dim_per_block,
                                SubspaceHankel *H) {
    if (!M || !H || !M->data || !H->data) return;
    if (H->total_rows < M->rows || H->total_cols < M->cols) return;
    for (int i = 0; i < M->rows; i++)
        for (int j = 0; j < M->cols; j++)
            H->data[(size_t)i * (size_t)H->total_cols + (size_t)j] =
                subspace_matrix_get(M, i, j);
    H->total_rows = M->rows;
    H->total_cols = M->cols;
    H->dim_per_block = dim_per_block;
    H->block_rows = M->rows / dim_per_block;
    H->block_cols = M->cols;
}

/* ============================================================================
 * Model Simulation and Response Functions
 * ============================================================================ */

void subspace_model_simulate(const SubspaceModel *model, const double *u,
                              double *y, int N) {
    if (!model || !u || !y || N <= 0) return;
    int n = model->n, r = model->r, m = model->m;
    double *x = (double*)calloc((size_t)n, sizeof(double));
    if (!x) return;
    if (model->x0) {
        for (int i = 0; i < n; i++) x[i] = model->x0[i];
    }
    for (int k = 0; k < N; k++) {
        for (int i = 0; i < m; i++) {
            double yi = 0.0;
            for (int j = 0; j < n; j++)
                yi += model->C[(size_t)i * (size_t)n + (size_t)j] * x[j];
            for (int j = 0; j < r; j++)
                yi += model->D[(size_t)i * (size_t)r + (size_t)j] *
                      u[(size_t)k * (size_t)r + (size_t)j];
            y[(size_t)k * (size_t)m + (size_t)i] = yi;
        }
        if (k < N - 1) {
            double *x_new = (double*)calloc((size_t)n, sizeof(double));
            if (!x_new) break;
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++)
                    x_new[i] += model->A[(size_t)i * (size_t)n + (size_t)j] * x[j];
                for (int j = 0; j < r; j++)
                    x_new[i] += model->B[(size_t)i * (size_t)r + (size_t)j] *
                                u[(size_t)k * (size_t)r + (size_t)j];
            }
            for (int i = 0; i < n; i++) x[i] = x_new[i];
            free(x_new);
        }
    }
    free(x);
}

void subspace_model_predict(const SubspaceModel *model, const double *u,
                             const double *y_measured, double *y_pred, int N) {
    if (!model || !u || !y_measured || !y_pred || N <= 0) return;
    int n = model->n, r = model->r, m = model->m;
    double *x = (double*)calloc((size_t)n, sizeof(double));
    if (!x) return;
    if (model->x0) {
        for (int i = 0; i < n; i++) x[i] = model->x0[i];
    }
    for (int k = 0; k < N; k++) {
        for (int i = 0; i < m; i++) {
            double yi = 0.0;
            for (int j = 0; j < n; j++)
                yi += model->C[(size_t)i * (size_t)n + (size_t)j] * x[j];
            for (int j = 0; j < r; j++)
                yi += model->D[(size_t)i * (size_t)r + (size_t)j] *
                      u[(size_t)k * (size_t)r + (size_t)j];
            y_pred[(size_t)k * (size_t)m + (size_t)i] = yi;
        }
        if (k < N - 1) {
            double *innovation = (double*)calloc((size_t)m, sizeof(double));
            double *x_new = (double*)calloc((size_t)n, sizeof(double));
            if (!innovation || !x_new) { free(innovation); free(x_new); break; }
            for (int i = 0; i < m; i++)
                innovation[i] = y_measured[(size_t)k * (size_t)m + (size_t)i] -
                                y_pred[(size_t)k * (size_t)m + (size_t)i];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++)
                    x_new[i] += model->A[(size_t)i * (size_t)n + (size_t)j] * x[j];
                for (int j = 0; j < r; j++)
                    x_new[i] += model->B[(size_t)i * (size_t)r + (size_t)j] *
                                u[(size_t)k * (size_t)r + (size_t)j];
                for (int j = 0; j < m; j++)
                    x_new[i] += model->K[(size_t)i * (size_t)m + (size_t)j] *
                                innovation[j];
            }
            for (int i = 0; i < n; i++) x[i] = x_new[i];
            free(innovation); free(x_new);
        }
    }
    free(x);
}

void subspace_model_bode(const SubspaceModel *model, double omega,
                          double *mag, double *phase) {
    if (!model || !mag || !phase) return;
    int n = model->n;
    /* SISO frequency response: G(e^{jw}) = C*(e^{jw}*I - A)^{-1}*B + D */
    if (model->m == 1 && model->r == 1) {
        double cos_w = cos(omega), sin_w = sin(omega);
        if (n == 1) {
            double denom_re = cos_w - model->A[0];
            double denom_im = sin_w;
            double denom_mag2 = denom_re * denom_re + denom_im * denom_im;
            if (denom_mag2 < 1e-30) { *mag = 1e10; *phase = 0.0; return; }
            double G_re = model->C[0] * model->B[0] * denom_re / denom_mag2 + model->D[0];
            double G_im = -model->C[0] * model->B[0] * denom_im / denom_mag2;
            *mag = sqrt(G_re * G_re + G_im * G_im);
            *phase = atan2(G_im, G_re);
        } else {
            /* For n > 1, use power iteration to approximate the dominant
             * eigenvalue contribution to the frequency response. */
            double spectral_radius = 0.0;
            double *v = (double*)calloc((size_t)n, sizeof(double));
            if (v) {
                v[0] = 1.0;
                for (int iter = 0; iter < 50; iter++) {
                    double *Av = (double*)calloc((size_t)n, sizeof(double));
                    double nrm = 0.0;
                    for (int i = 0; i < n; i++) {
                        for (int j = 0; j < n; j++)
                            Av[i] += model->A[(size_t)i * (size_t)n + (size_t)j] * v[j];
                        nrm += Av[i] * Av[i];
                    }
                    nrm = sqrt(nrm);
                    if (nrm < 1e-15) break;
                    spectral_radius = nrm;
                    for (int i = 0; i < n; i++) v[i] = Av[i] / nrm;
                    free(Av);
                }
                free(v);
            }
            /* Estimate magnitude from DC gain and spectral radius */
            double dc_gain = fabs(model->D[0]);
            for (int i = 0; i < n; i++) {
                double contrib = 0.0;
                for (int j = 0; j < n; j++) {
                    /* Sum of (I-A)^{-1}[i,j] * B[j] */
                    contrib += fabs(model->C[i]) * fabs(model->B[i]);
                }
                dc_gain += contrib / (1.0 - spectral_radius + 0.01);
            }
            *mag = dc_gain / sqrt(1.0 + omega * omega);
            *phase = -atan2(omega, 1.0 - spectral_radius);
        }
    } else {
        *mag = 1.0; *phase = 0.0;
    }
}

void subspace_model_impulse_response(const SubspaceModel *model,
                                      double *impulse, int len) {
    if (!model || !impulse || len <= 0) return;
    int n = model->n;
    if (model->m == 1 && model->r == 1) {
        impulse[0] = model->D[0];
        if (len <= 1) return;
        double *x = (double*)calloc((size_t)n, sizeof(double));
        double *x_next = (double*)calloc((size_t)n, sizeof(double));
        if (!x || !x_next) { free(x); free(x_next); return; }
        for (int i = 0; i < n; i++) x[i] = model->B[i];
        for (int k = 1; k < len; k++) {
            double yk = 0.0;
            for (int i = 0; i < n; i++) yk += model->C[i] * x[i];
            impulse[k] = yk;
            for (int i = 0; i < n; i++) {
                x_next[i] = 0.0;
                for (int j = 0; j < n; j++)
                    x_next[i] += model->A[(size_t)i * (size_t)n + (size_t)j] * x[j];
            }
            double *tmp = x; x = x_next; x_next = tmp;
        }
        free(x); free(x_next);
    } else {
        for (int k = 0; k < len * model->m * model->r; k++) impulse[k] = 0.0;
    }
}

void subspace_model_step_response(const SubspaceModel *model,
                                   double *step, int len) {
    if (!model || !step || len <= 0) return;
    double *imp = (double*)calloc((size_t)len, sizeof(double));
    if (!imp) return;
    subspace_model_impulse_response(model, imp, len);
    double cumsum = 0.0;
    for (int k = 0; k < len; k++) { cumsum += imp[k]; step[k] = cumsum; }
    free(imp);
}

void subspace_model_poles(const SubspaceModel *model, double *real_part,
                           double *imag_part) {
    if (!model || !real_part || !imag_part) return;
    int n = model->n;
    if (n == 1) {
        real_part[0] = model->A[0]; imag_part[0] = 0.0;
    } else if (n == 2) {
        double a = model->A[0], b = model->A[1];
        double c = model->A[2], d = model->A[3];
        double trace = a + d, det = a * d - b * c;
        double disc = trace * trace - 4.0 * det;
        if (disc >= 0) {
            double sd = sqrt(disc);
            real_part[0] = (trace + sd) / 2.0;
            real_part[1] = (trace - sd) / 2.0;
            imag_part[0] = 0.0; imag_part[1] = 0.0;
        } else {
            real_part[0] = trace / 2.0; real_part[1] = trace / 2.0;
            imag_part[0] = sqrt(-disc) / 2.0;
            imag_part[1] = -sqrt(-disc) / 2.0;
        }
    } else {
        double *A_copy = (double*)malloc((size_t)n * (size_t)n * sizeof(double));
        if (A_copy) {
            memcpy(A_copy, model->A, (size_t)n * (size_t)n * sizeof(double));
            subspace_eigenvalues_real(A_copy, n, real_part, imag_part);
            free(A_copy);
        }
    }
}

/* ============================================================================
 * Validation Functions
 * ============================================================================ */

double subspace_fit_percent(const double *y_true, const double *y_pred, int N) {
    if (!y_true || !y_pred || N <= 1) return 0.0;
    double y_mean = subspace_mean(y_true, N);
    double num = 0.0, den = 0.0;
    for (int i = 0; i < N; i++) {
        double e = y_true[i] - y_pred[i];
        num += e * e;
        double d = y_true[i] - y_mean;
        den += d * d;
    }
    if (den < 1e-15) return 100.0;
    return 100.0 * (1.0 - sqrt(num / den));
}

double subspace_residual_autocorr(const double *residuals, int N, int max_lag) {
    if (!residuals || N <= max_lag) return 1.0;
    double mean_r = subspace_mean(residuals, N);
    double var = 0.0;
    for (int i = 0; i < N; i++) { double d = residuals[i] - mean_r; var += d * d; }
    if (var < 1e-15) return 0.0;
    double max_acf = 0.0;
    for (int lag = 1; lag <= max_lag; lag++) {
        double acf = 0.0;
        for (int i = 0; i < N - lag; i++)
            acf += (residuals[i] - mean_r) * (residuals[i + lag] - mean_r);
        acf /= var;
        if (fabs(acf) > max_acf) max_acf = fabs(acf);
    }
    return max_acf;
}

double subspace_cross_correlation(const double *u, const double *residuals,
                                    int N, int max_lag) {
    if (!u || !residuals || N <= max_lag) return 1.0;
    double mean_u = subspace_mean(u, N), mean_e = subspace_mean(residuals, N);
    double var_u = 0.0, var_e = 0.0;
    for (int i = 0; i < N; i++) {
        double du = u[i] - mean_u, de = residuals[i] - mean_e;
        var_u += du * du; var_e += de * de;
    }
    double denom = sqrt(var_u * var_e);
    if (denom < 1e-15) return 0.0;
    double max_ccf = 0.0;
    for (int lag = -max_lag; lag <= max_lag; lag++) {
        double ccf = 0.0; int count = 0;
        for (int i = 0; i < N; i++) {
            int j = i + lag;
            if (j >= 0 && j < N) {
                ccf += (u[j] - mean_u) * (residuals[i] - mean_e);
                count++;
            }
        }
        if (count > 0 && fabs(ccf / denom) > max_ccf) max_ccf = fabs(ccf / denom);
    }
    return max_ccf;
}

double subspace_variance_accounted_for(const double *y_true,
                                        const double *y_pred, int N) {
    if (!y_true || !y_pred || N <= 1) return 0.0;
    double var_y = subspace_variance(y_true, N, subspace_mean(y_true, N));
    if (var_y < 1e-15) return 100.0;
    double mse = 0.0;
    for (int i = 0; i < N; i++) { double e = y_true[i] - y_pred[i]; mse += e * e; }
    mse /= (double)N;
    return 100.0 * (1.0 - mse / var_y);
}

/* ============================================================================
 * Print Functions
 * ============================================================================ */

void subspace_matrix_print(const SubspaceMatrix *mat, const char *name) {
    if (!mat) { printf("%s: (null)\n", name ? name : "matrix"); return; }
    printf("%s [%d x %d]:\n", name ? name : "matrix", mat->rows, mat->cols);
    int mr = (mat->rows > 12) ? 12 : mat->rows;
    int mc = (mat->cols > 8) ? 8 : mat->cols;
    for (int i = 0; i < mr; i++) {
        printf("  ");
        for (int j = 0; j < mc; j++)
            printf("% .4e ", subspace_matrix_get(mat, i, j));
        if (mat->cols > 8) printf("...");
        printf("\n");
    }
    if (mat->rows > 12) printf("  ... (%d more rows)\n", mat->rows - 12);
}

void subspace_print_model(const SubspaceModel *model) {
    if (!model) { printf("Model: (null)\n"); return; }
    printf("=== State-Space Model ===\n");
    printf("Order n=%d, Inputs r=%d, Outputs m=%d\n", model->n, model->r, model->m);
    printf("Stability: ");
    switch (model->stability) {
        case SS_STABLE:     printf("Stable\n"); break;
        case SS_UNSTABLE:   printf("Unstable\n"); break;
        case SS_MARGINALLY: printf("Marginally stable\n"); break;
        default:            printf("Not checked\n"); break;
    }
    if (model->n > 0) {
        double *re = (double*)malloc((size_t)model->n * sizeof(double));
        double *im = (double*)malloc((size_t)model->n * sizeof(double));
        if (re && im) {
            subspace_model_poles(model, re, im);
            printf("Eigenvalues of A (top 10):\n");
            for (int i = 0; i < model->n && i < 10; i++)
                printf("  lambda[%d] = % .4e %+.4ej (|lambda|=%.4f)\n",
                       i, re[i], im[i], sqrt(re[i]*re[i] + im[i]*im[i]));
        }
        free(re); free(im);
    }
}

void subspace_print_result(const SubspaceResult *result) {
    if (!result) { printf("Result: (null)\n"); return; }
    printf("=== Subspace Identification Result ===\n");
    printf("Order estimated: %d\n", result->order_estimated);
    printf("Loss: %.6e  NRMSE Fit: %.2f%%\n", result->loss, result->fit_percent);
    printf("Cond(A): %.4e  Time: %.4f sec\n",
           result->condition_A, result->elapsed_sec);
    if (result->status_msg[0]) printf("Status: %s\n", result->status_msg);
    if (result->singular_values && result->sv_count > 0) {
        printf("Singular values (top 10):\n");
        for (int i = 0; i < result->sv_count && i < 10; i++)
            printf("  sigma[%d] = %.6e\n", i, result->singular_values[i]);
    }
    if (result->model) subspace_print_model(result->model);
}

/* ============================================================================
 * Utility Functions
 * ============================================================================ */

double subspace_mean(const double *x, int n) {
    if (!x || n <= 0) return 0.0;
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += x[i];
    return sum / (double)n;
}

double subspace_variance(const double *x, int n, double mean) {
    if (!x || n <= 1) return 0.0;
    double sum_sq = 0.0;
    for (int i = 0; i < n; i++) { double d = x[i] - mean; sum_sq += d * d; }
    return sum_sq / (double)(n - 1);
}

double subspace_dot_product(const double *a, const double *b, int n) {
    if (!a || !b || n <= 0) return 0.0;
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += a[i] * b[i];
    return sum;
}

double subspace_norm2(const double *v, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += v[i] * v[i];
    return sqrt(sum);
}

double subspace_condition_number(const double *A_data, int n) {
    if (!A_data || n <= 0) return 0.0;
    double frob = 0.0;
    for (int i = 0; i < n * n; i++) frob += A_data[i] * A_data[i];
    frob = sqrt(frob);
    if (frob < 1e-15) return 0.0;
    double max_abs = 0.0;
    for (int i = 0; i < n; i++) {
        double row_sum = 0.0;
        for (int j = 0; j < n; j++)
            row_sum += fabs(A_data[(size_t)i * (size_t)n + (size_t)j]);
        if (row_sum > max_abs) max_abs = row_sum;
    }
    if (max_abs < 1e-15) return 1e15;
    return frob / max_abs;
}
