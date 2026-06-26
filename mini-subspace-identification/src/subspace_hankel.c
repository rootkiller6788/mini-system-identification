#include "subspace_core.h"
#include "subspace_hankel.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ============================================================================
 * Subspace Identification -- Block Hankel Matrix Implementation
 *
 * Constructs and manipulates block Hankel matrices, which are the
 * fundamental data structures in subspace identification. The block
 * Hankel matrix organizes time series data into a structured form
 * that reveals the underlying state-space dynamics.
 * ============================================================================ */

/* Build block Hankel from single-channel signal */
void subspace_hankel_build(const double *signal, int N, int block_rows,
                            SubspaceHankel *H) {
    subspace_hankel_from_signal(signal, N, 1, block_rows, H);
}

/* Build block Hankel from MIMO signal */
void subspace_hankel_build_mimo(const double *signal, int N, int dim,
                                 int block_rows, SubspaceHankel *H) {
    subspace_hankel_from_signal(signal, N, dim, block_rows, H);
}

/* Build U_p, U_f, Y_p, Y_f from IO data with block row count i */
int subspace_hankel_build_all(const SubspaceData *data, int i,
                               SubspaceHankel *Up, SubspaceHankel *Uf,
                               SubspaceHankel *Yp, SubspaceHankel *Yf) {
    if (!data || i <= 0 || !Up || !Uf || !Yp || !Yf) return -1;
    int total_blocks = 2 * i;
    int block_cols = data->N - total_blocks + 1;
    if (block_cols <= 0) return -2;

    /* Resize Hankel matrices if needed */
    Up->block_rows = i; Up->block_cols = block_cols;
    Up->dim_per_block = data->n_inputs;
    Up->total_rows = i * data->n_inputs;
    Up->total_cols = block_cols;

    Uf->block_rows = i; Uf->block_cols = block_cols;
    Uf->dim_per_block = data->n_inputs;
    Uf->total_rows = i * data->n_inputs;
    Uf->total_cols = block_cols;

    Yp->block_rows = i; Yp->block_cols = block_cols;
    Yp->dim_per_block = data->n_outputs;
    Yp->total_rows = i * data->n_outputs;
    Yp->total_cols = block_cols;

    Yf->block_rows = i; Yf->block_cols = block_cols;
    Yf->dim_per_block = data->n_outputs;
    Yf->total_rows = i * data->n_outputs;
    Yf->total_cols = block_cols;

    subspace_hankel_from_io_data(data, i, Up, Uf, Yp, Yf);
    return 0;
}

/* Build W_p = [U_p; Y_p] as a standard matrix */
void subspace_hankel_build_wp(const SubspaceHankel *Up, const SubspaceHankel *Yp,
                               SubspaceMatrix *Wp) {
    if (!Up || !Yp || !Wp) return;
    int total_rows = Up->total_rows + Yp->total_rows;
    int total_cols = Up->total_cols;
    if (Wp->rows < total_rows || Wp->cols < total_cols) return;

    for (int j = 0; j < total_cols; j++) {
        /* Copy U_p rows */
        for (int i = 0; i < Up->total_rows; i++) {
            subspace_matrix_set(Wp, i, j,
                Up->data[(size_t)i * (size_t)Up->total_cols + (size_t)j]);
        }
        /* Copy Y_p rows */
        for (int i = 0; i < Yp->total_rows; i++) {
            subspace_matrix_set(Wp, Up->total_rows + i, j,
                Yp->data[(size_t)i * (size_t)Yp->total_cols + (size_t)j]);
        }
    }
}

/* Build W_f = [U_f; Y_f] as a standard matrix */
void subspace_hankel_build_wf(const SubspaceHankel *Uf, const SubspaceHankel *Yf,
                               SubspaceMatrix *Wf) {
    if (!Uf || !Yf || !Wf) return;
    int total_rows = Uf->total_rows + Yf->total_rows;
    int total_cols = Uf->total_cols;
    if (Wf->rows < total_rows || Wf->cols < total_cols) return;

    for (int j = 0; j < total_cols; j++) {
        for (int i = 0; i < Uf->total_rows; i++) {
            subspace_matrix_set(Wf, i, j,
                Uf->data[(size_t)i * (size_t)Uf->total_cols + (size_t)j]);
        }
        for (int i = 0; i < Yf->total_rows; i++) {
            subspace_matrix_set(Wf, Uf->total_rows + i, j,
                Yf->data[(size_t)i * (size_t)Yf->total_cols + (size_t)j]);
        }
    }
}

/* Convert Hankel to standard matrix */
void subspace_hankel_export_matrix(const SubspaceHankel *H, SubspaceMatrix *M,
                                    bool row_major) {
    if (!H || !M) return;
    int rows = H->total_rows, cols = H->total_cols;
    if (M->rows < rows || M->cols < cols) return;
    if (row_major) {
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                subspace_matrix_set(M, i, j,
                    H->data[(size_t)i * (size_t)cols + (size_t)j]);
    } else {
        /* Column-major: direct copy if data layout matches, else transpose */
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                subspace_matrix_set(M, i, j,
                    H->data[(size_t)i * (size_t)cols + (size_t)j]);
    }
}

/* Import from matrix to Hankel */
void subspace_hankel_import_matrix(const SubspaceMatrix *M, int dim_per_block,
                                    SubspaceHankel *H) {
    subspace_matrix_to_hankel(M, dim_per_block, H);
}

/* Shift operation: remove first block row, add zeros at end */
void subspace_hankel_shift(const SubspaceHankel *H, SubspaceHankel *H_shifted) {
    if (!H || !H_shifted || H->block_rows <= 1) return;
    int new_block_rows = H->block_rows - 1;
    int dim = H->dim_per_block;
    int cols = H->total_cols;

    H_shifted->block_rows = new_block_rows;
    H_shifted->block_cols = H->block_cols;
    H_shifted->dim_per_block = dim;
    H_shifted->total_rows = new_block_rows * dim;
    H_shifted->total_cols = cols;

    size_t src_offset = (size_t)dim * (size_t)cols;
    size_t src_total = (size_t)H->total_rows * (size_t)cols;
    size_t copy_bytes = (src_total - src_offset) * sizeof(double);
    memcpy(H_shifted->data, H->data + src_offset / sizeof(double) /* pointer arithmetic */,
           copy_bytes > 0 ? copy_bytes : 0);
}

/* Extract sub-block */
void subspace_hankel_subblock(const SubspaceHankel *H, int start_block,
                               int num_blocks, SubspaceHankel *sub) {
    if (!H || !sub || start_block < 0 ||
        start_block + num_blocks > H->block_rows) return;
    int dim = H->dim_per_block, cols = H->total_cols;
    sub->block_rows = num_blocks;
    sub->block_cols = H->block_cols;
    sub->dim_per_block = dim;
    sub->total_rows = num_blocks * dim;
    sub->total_cols = cols;

    for (int br = 0; br < num_blocks; br++) {
        int src_br = start_block + br;
        for (int ch = 0; ch < dim; ch++) {
            int src_row = src_br * dim + ch;
            int dst_row = br * dim + ch;
            for (int j = 0; j < cols; j++) {
                sub->data[(size_t)dst_row * (size_t)cols + (size_t)j] =
                    H->data[(size_t)src_row * (size_t)cols + (size_t)j];
            }
        }
    }
}

/* ============================================================================
 * Data Preprocessing
 * ============================================================================ */

/* Remove mean and/or linear trend from IO data */
void subspace_data_detrend(SubspaceData *data, bool remove_mean,
                            bool remove_linear) {
    if (!data || !data->u || !data->y) return;
    int N = data->N, r = data->n_inputs, m = data->n_outputs;
    double t_mean = (double)(N - 1) / 2.0;

    for (int ch = 0; ch < r; ch++) {
        if (remove_mean) {
            double mu = 0.0;
            for (int k = 0; k < N; k++)
                mu += data->u[(size_t)k * (size_t)r + (size_t)ch];
            mu /= (double)N;
            for (int k = 0; k < N; k++)
                data->u[(size_t)k * (size_t)r + (size_t)ch] -= mu;
        }
        if (remove_linear) {
            double sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0;
            for (int k = 0; k < N; k++) {
                double t = (double)k - t_mean;
                double v = data->u[(size_t)k * (size_t)r + (size_t)ch];
                sx += t; sy += v; sxx += t * t; sxy += t * v;
            }
            double slope = (sxy - sx * sy / N) / (sxx - sx * sx / N + 1e-15);
            double intercept = (sy - slope * sx) / N;
            for (int k = 0; k < N; k++) {
                double t = (double)k - t_mean;
                data->u[(size_t)k * (size_t)r + (size_t)ch] -= (slope * t + intercept);
            }
        }
    }
    for (int ch = 0; ch < m; ch++) {
        if (remove_mean) {
            double mu = 0.0;
            for (int k = 0; k < N; k++)
                mu += data->y[(size_t)k * (size_t)m + (size_t)ch];
            mu /= (double)N;
            for (int k = 0; k < N; k++)
                data->y[(size_t)k * (size_t)m + (size_t)ch] -= mu;
        }
        if (remove_linear) {
            double sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0;
            for (int k = 0; k < N; k++) {
                double t = (double)k - t_mean;
                double v = data->y[(size_t)k * (size_t)m + (size_t)ch];
                sx += t; sy += v; sxx += t * t; sxy += t * v;
            }
            double slope = (sxy - sx * sy / N) / (sxx - sx * sx / N + 1e-15);
            double intercept = (sy - slope * sx) / N;
            for (int k = 0; k < N; k++) {
                double t = (double)k - t_mean;
                data->y[(size_t)k * (size_t)m + (size_t)ch] -= (slope * t + intercept);
            }
        }
    }
}

/* Scale data to unit variance */
void subspace_data_normalize(SubspaceData *data) {
    if (!data || !data->u || !data->y) return;
    int N = data->N, r = data->n_inputs, m = data->n_outputs;
    for (int ch = 0; ch < r; ch++) {
        double mu = 0.0, var = 0.0;
        for (int k = 0; k < N; k++)
            mu += data->u[(size_t)k * (size_t)r + (size_t)ch];
        mu /= (double)N;
        for (int k = 0; k < N; k++) {
            double d = data->u[(size_t)k * (size_t)r + (size_t)ch] - mu;
            var += d * d;
        }
        var /= (double)(N - 1);
        double stddev = sqrt(var);
        if (stddev > 1e-15) {
            for (int k = 0; k < N; k++)
                data->u[(size_t)k * (size_t)r + (size_t)ch] =
                    (data->u[(size_t)k * (size_t)r + (size_t)ch] - mu) / stddev;
        }
    }
    for (int ch = 0; ch < m; ch++) {
        double mu = 0.0, var = 0.0;
        for (int k = 0; k < N; k++)
            mu += data->y[(size_t)k * (size_t)m + (size_t)ch];
        mu /= (double)N;
        for (int k = 0; k < N; k++) {
            double d = data->y[(size_t)k * (size_t)m + (size_t)ch] - mu;
            var += d * d;
        }
        var /= (double)(N - 1);
        double stddev = sqrt(var);
        if (stddev > 1e-15) {
            for (int k = 0; k < N; k++)
                data->y[(size_t)k * (size_t)m + (size_t)ch] =
                    (data->y[(size_t)k * (size_t)m + (size_t)ch] - mu) / stddev;
        }
    }
}

/* Validate data */
int subspace_data_validate(const SubspaceData *data) {
    if (!data) return -1;
    if (data->N < 10) return -2;  /* Too few samples */
    if (data->n_inputs <= 0) return -3;
    if (data->n_outputs <= 0) return -4;
    if (!data->u || !data->y) return -5;
    /* Check for constant signals (no excitation) */
    double input_var = 0.0;
    int r = data->n_inputs;
    for (int ch = 0; ch < r; ch++) {
        double mu = 0.0, var = 0.0;
        for (int k = 0; k < data->N; k++)
            mu += data->u[(size_t)k * (size_t)r + (size_t)ch];
        mu /= (double)data->N;
        for (int k = 0; k < data->N; k++) {
            double d = data->u[(size_t)k * (size_t)r + (size_t)ch] - mu;
            var += d * d;
        }
        var /= (double)(data->N - 1);
        if (var > input_var) input_var = var;
    }
    if (input_var < 1e-15) return -6;  /* No input excitation */
    return 0;
}
