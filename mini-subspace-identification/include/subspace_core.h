#ifndef SUBSPACE_CORE_H
#define SUBSPACE_CORE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ============================================================================
 * Subspace Identification -- Core Types and Utilities
 *
 * Based on foundational works:
 *   Van Overschee & De Moor (1996) -- Subspace Identification for Linear Systems
 *   Verhaegen & Dewilde (1992) -- Subspace Model Identification
 *   Larimore (1990) -- Canonical Variate Analysis in Identification
 *   Ljung (1999) -- System Identification: Theory for the User, 2nd ed.
 *   Katayama (2005) -- Subspace Methods for System Identification
 *
 * Subspace identification estimates state-space models directly from
 * input-output data using numerical linear algebra (QR, SVD) without
 * iterative optimization. The key insight: the extended observability
 * matrix Gamma_i (and/or the state sequence X) can be recovered from the
 * row space of certain projected data matrices.
 *
 * Mathematical Foundation:
 *   State-space model (innovation form):
 *     x(k+1) = A x(k) + B u(k) + K e(k)
 *     y(k)   = C x(k) + D u(k) + e(k)
 *
 *   Block Hankel matrices from data Z^N = {u(1),y(1),...,u(N),y(N)}:
 *     U_p = past inputs,  U_f = future inputs
 *     Y_p = past outputs, Y_f = future outputs
 *
 *   Key subspace relation:
 *     Y_f = Gamma_i X_f + H_i^d U_f + G_i E_f
 *   where Gamma_i = [C; CA; ...; CA^{i-1}] extended observability matrix.
 *
 *   Oblique projection: O_i = Y_f /_{U_f} W_p  =  Gamma_i Xhat_i
 *   SVD of weighted projection: W_1 O_i W_2 = U Sigma V^T
 *   Order n is determined by the rank of O_i (number of non-zero singular vals).
 * ============================================================================ */

/* --- Subspace Algorithm Variants --- */
typedef enum {
    SS_N4SID    = 0,  /* Numerical algorithms for Subspace State Space System ID */
    SS_MOESP    = 1,  /* Multivariable Output-Error State sPace */
    SS_CVA      = 2,  /* Canonical Variate Analysis */
    SS_PO_MOESP = 3   /* Past-Observable MOESP */
} SubspaceAlgorithm;

/* --- Weighting Schemes (for W_1 O_i W_2) --- */
typedef enum {
    SS_WGT_N4SID = 0,  /* W_1 = I, W_2 = I (original N4SID) */
    SS_WGT_MOESP = 1,  /* W_1 = I, W_2 = Pi_{U_f^bot} (MOESP weighting) */
    SS_WGT_CVA   = 2   /* W_1 = (Y_f Pi_{U_f^bot} Y_f^T)^{-1/2}, W_2 = Pi_{U_f^bot} */
} SubspaceWeighting;

/* --- System Order Selection Criterion --- */
typedef enum {
    SS_ORDER_SVD_GAP   = 0,  /* Singular value gap detection */
    SS_ORDER_AIC       = 1,  /* Akaike Information Criterion */
    SS_ORDER_NIC       = 2,  /* Normalized Information Criterion (Bauer 2001) */
    SS_ORDER_SVC       = 3   /* Singular Value Criterion (ratio threshold) */
} SubspaceOrderCriterion;

/* --- Stability Status of Identified System --- */
typedef enum {
    SS_STABLE        = 0,
    SS_UNSTABLE      = 1,
    SS_MARGINALLY    = 2,  /* eigenvalues on unit circle */
    SS_NOT_CHECKED   = 3
} SubspaceStability;

/* --- Real-valued dense matrix (column-major storage) --- */
typedef struct {
    int      rows;
    int      cols;
    double  *data;    /* length = rows * cols, column-major */
} SubspaceMatrix;

/* --- Singular Value Decomposition Result --- */
typedef struct {
    int              n;  /* number of singular values */
    double          *S;  /* singular values [n] in descending order */
    SubspaceMatrix  *U;  /* left singular vectors (m x n thin) */
    SubspaceMatrix  *V;  /* right singular vectors (n x n thin) */
} SubspaceSVD;

/* --- Block Hankel Matrix ---
 * Constructed from signal s = [s(0),...,s(N-1)] with i block rows and j cols:
 *   H_{i,j} = [ s(0)    s(1)    ... s(j-1)    ]
 *             [ s(1)    s(2)    ... s(j)      ]
 *             [ ...     ...     ... ...       ]
 *             [ s(i-1)  s(i)    ... s(i+j-2)  ]
 * j = N - i + 1. For MIMO (m outputs, r inputs), each block has m or r rows.
 */
typedef struct {
    int      block_rows;      /* i = number of block rows */
    int      block_cols;      /* j = number of block columns */
    int      dim_per_block;   /* m (outputs) or r (inputs) per block row */
    int      total_rows;      /* block_rows * dim_per_block */
    int      total_cols;      /* block_cols */
    double  *data;            /* length = total_rows * total_cols, row-major */
} SubspaceHankel;

/* --- Input-Output Data Set --- */
typedef struct {
    int      N;           /* number of samples */
    int      n_inputs;    /* r = input dimension */
    int      n_outputs;   /* m = output dimension */
    double   Ts;          /* sampling period */
    double  *u;           /* inputs,  size = N * n_inputs,  row-major */
    double  *y;           /* outputs, size = N * n_outputs, row-major */
    char    *name;
} SubspaceData;

/* --- State-Space Model (Discrete-Time, Innovation Form) ---
 * x(k+1) = A x(k) + B u(k) + K e(k)
 * y(k)   = C x(k) + D u(k) + e(k)
 */
typedef struct {
    int               n;          /* system order = state dimension */
    int               r;          /* input dimension */
    int               m;          /* output dimension */
    double           *A;          /* n x n system matrix, row-major */
    double           *B;          /* n x r input matrix, row-major */
    double           *C;          /* m x n output matrix, row-major */
    double           *D;          /* m x r feedthrough matrix, row-major */
    double           *K;          /* n x m Kalman gain, row-major */
    double           *x0;         /* n initial state */
    SubspaceStability stability;
    double            Ts;
} SubspaceModel;

/* --- Subspace Identification Options --- */
typedef struct {
    int                   i;                  /* block rows (future horizon) */
    int                   max_order;          /* maximum system order to test */
    SubspaceAlgorithm     algorithm;          /* N4SID, MOESP, or CVA */
    SubspaceWeighting     weighting;
    SubspaceOrderCriterion order_crit;
    double                sv_threshold;       /* singular value cutoff ratio */
    bool                  estimate_D;         /* estimate D matrix */
    bool                  estimate_K;         /* estimate Kalman gain */
    bool                  enforce_stability;
    bool                  verbose;
    int                   past_horizon_mult;  /* past horizon = mult * i */
} SubspaceOptions;

/* --- Subspace Identification Result --- */
typedef struct {
    SubspaceModel  *model;            /* identified state-space model */
    double         *singular_values;  /* SVs from projection SVD */
    int             sv_count;
    int             order_estimated;
    double         *eigenvalues;      /* eigenvalues of A (length n) */
    double          loss;             /* output prediction error variance */
    double          fit_percent;      /* NRMSE fit (0-100) */
    double          condition_A;
    double          elapsed_sec;
    char            status_msg[256];
} SubspaceResult;

/* ============================================================================
 * Core API Function Declarations
 * ============================================================================ */

/* --- Memory Management --- */
SubspaceData*     subspace_data_alloc(int N, int n_inputs, int n_outputs);
void              subspace_data_free(SubspaceData *data);
SubspaceMatrix*   subspace_matrix_alloc(int rows, int cols);
void              subspace_matrix_free(SubspaceMatrix *mat);
SubspaceHankel*   subspace_hankel_alloc(int block_rows, int block_cols,
                                        int dim_per_block);
void              subspace_hankel_free(SubspaceHankel *H);
SubspaceModel*    subspace_model_alloc(int n, int r, int m);
void              subspace_model_free(SubspaceModel *model);
SubspaceResult*   subspace_result_alloc(void);
void              subspace_result_free(SubspaceResult *result);
SubspaceSVD*      subspace_svd_alloc(int m, int n);
void              subspace_svd_free(SubspaceSVD *svd);
SubspaceOptions   subspace_options_default(void);

/* --- Matrix Operations --- */
void   subspace_matrix_set(SubspaceMatrix *mat, int i, int j, double val);
double subspace_matrix_get(const SubspaceMatrix *mat, int i, int j);
void   subspace_matrix_fill(SubspaceMatrix *mat, double val);
void   subspace_matrix_copy(const SubspaceMatrix *src, SubspaceMatrix *dst);
void   subspace_matrix_transpose(const SubspaceMatrix *src, SubspaceMatrix *dst);
void   subspace_matrix_multiply(const SubspaceMatrix *A, const SubspaceMatrix *B,
                                 SubspaceMatrix *C);
void   subspace_matrix_add(const SubspaceMatrix *A, const SubspaceMatrix *B,
                            SubspaceMatrix *C);
void   subspace_matrix_subtract(const SubspaceMatrix *A, const SubspaceMatrix *B,
                                 SubspaceMatrix *C);
double subspace_matrix_norm_frobenius(const SubspaceMatrix *mat);
double subspace_matrix_trace(const SubspaceMatrix *mat);
void   subspace_matrix_identity(SubspaceMatrix *mat);
void   subspace_matrix_diag(SubspaceMatrix *mat, const double *diag, int n);

/* --- Linear Algebra Utilities --- */
void   subspace_qr_decompose(const SubspaceMatrix *A, SubspaceMatrix *Q,
                              SubspaceMatrix *R);
int    subspace_svd_compute(const SubspaceMatrix *A, SubspaceSVD *result);
int    subspace_cholesky(const SubspaceMatrix *A, SubspaceMatrix *L);
void   subspace_solve_triangular(const SubspaceMatrix *R, const SubspaceMatrix *B,
                                  SubspaceMatrix *X, bool upper, bool trans);
int    subspace_solve_linear(const SubspaceMatrix *A, const SubspaceMatrix *B,
                              SubspaceMatrix *X);
double subspace_eigenvalues_real(double *A_data, int n, double *eig_real,
                                  double *eig_imag);
void   subspace_matrix_print(const SubspaceMatrix *mat, const char *name);
void   subspace_matrix_printf(const SubspaceMatrix *mat, const char *fmt);

/* --- Hankel Matrix Operations --- */
void   subspace_hankel_from_signal(const double *signal, int N,
                                    int dim_per_block, int block_rows,
                                    SubspaceHankel *H);
void   subspace_hankel_from_io_data(const SubspaceData *data, int i,
                                     SubspaceHankel *Up, SubspaceHankel *Uf,
                                     SubspaceHankel *Yp, SubspaceHankel *Yf);
void   subspace_hankel_to_matrix(const SubspaceHankel *H, SubspaceMatrix *M);
void   subspace_matrix_to_hankel(const SubspaceMatrix *M, int dim_per_block,
                                  SubspaceHankel *H);

/* --- Projection Operations --- */
void   subspace_orthogonal_projection(const SubspaceMatrix *A,
                                       const SubspaceMatrix *B,
                                       SubspaceMatrix *P);
void   subspace_oblique_projection(const SubspaceMatrix *A,
                                    const SubspaceMatrix *B,
                                    const SubspaceMatrix *C, SubspaceMatrix *O);

/* --- Model Operations --- */
void   subspace_model_simulate(const SubspaceModel *model, const double *u,
                                double *y, int N);
void   subspace_model_predict(const SubspaceModel *model, const double *u,
                               const double *y_measured, double *y_pred, int N);
void   subspace_model_bode(const SubspaceModel *model, double omega,
                            double *mag, double *phase);
void   subspace_model_impulse_response(const SubspaceModel *model,
                                        double *impulse, int len);
void   subspace_model_step_response(const SubspaceModel *model,
                                     double *step, int len);
void   subspace_model_poles(const SubspaceModel *model, double *real_part,
                             double *imag_part);

/* --- Order Estimation --- */
int    subspace_estimate_order(const double *singular_values, int n_sv,
                                SubspaceOrderCriterion criterion,
                                double threshold, int max_order);
void   subspace_order_selection_plot(const double *singular_values, int n_sv);

/* --- System Matrices Extraction --- */
int    subspace_extract_n4sid(const SubspaceMatrix *Gamma,
                               const SubspaceMatrix *X_i,
                               const SubspaceMatrix *X_ip1,
                               const SubspaceData *data, int i,
                               SubspaceModel *model);
int    subspace_extract_moesp(const SubspaceMatrix *Gamma,
                               const SubspaceData *data, int i,
                               SubspaceModel *model);
int    subspace_extract_cva(const SubspaceMatrix *Gamma,
                             const SubspaceMatrix *X_i,
                             const SubspaceMatrix *X_ip1,
                             const SubspaceData *data, int i,
                             SubspaceModel *model);

/* --- High-Level Identification --- */
int    subspace_identify(const SubspaceData *data,
                          const SubspaceOptions *options,
                          SubspaceResult *result);

/* --- Validation --- */
double subspace_fit_percent(const double *y_true, const double *y_pred, int N);
double subspace_residual_autocorr(const double *residuals, int N, int max_lag);
double subspace_cross_correlation(const double *u, const double *residuals,
                                    int N, int max_lag);
double subspace_variance_accounted_for(const double *y_true,
                                        const double *y_pred, int N);

/* --- Utility Functions --- */
double subspace_mean(const double *x, int n);
double subspace_variance(const double *x, int n, double mean);
double subspace_dot_product(const double *a, const double *b, int n);
double subspace_norm2(const double *v, int n);
void   subspace_print_model(const SubspaceModel *model);
void   subspace_print_result(const SubspaceResult *result);
double subspace_condition_number(const double *A_data, int n);

/* --- Inline helpers --- */
static inline double subspace_safe_div(double num, double den, double fallback) {
    return (fabs(den) < 1e-15) ? fallback : (num / den);
}

static inline int subspace_min_int(int a, int b) { return (a < b) ? a : b; }
static inline int subspace_max_int(int a, int b) { return (a > b) ? a : b; }

#endif /* SUBSPACE_CORE_H */