#ifndef RLS_CORE_H
#define RLS_CORE_H

#include <stdbool.h>
#include <stddef.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ============================================================================
 * Regularized Least Squares for System Identification -- Core Types & Algebra
 *
 * [Ljung99]  Ljung, L. "System Identification: Theory for the User" 2/e, 1999
 * [Golub96]  Golub, G.H., Van Loan, C.F. "Matrix Computations" 3/e, 1996
 * [Hoerl70]  Hoerl & Kennard "Ridge Regression" Technometrics 12(1), 1970
 * [Tibsh96]  Tibshirani, R. "Lasso" JRSS-B 58(1), 1996
 * [Pill10]   Pillonetto et al. "Kernel-based linear system ID" Automatica, 2010
 * [Zou05]    Zou & Hastie "Elastic net" JRSS-B 67(2), 2005
 * ============================================================================ */

/* ---------- L1: Core Definitions ---------- */
typedef struct {
    double *data;
    int     rows, cols, capacity;
} RLSMatrix;

typedef struct {
    double *data;
    int     dim, capacity;
} RLSVector;

typedef enum {
    RLS_REG_NONE=0, RLS_REG_RIDGE=1, RLS_REG_LASSO=2,
    RLS_REG_ELASTICNET=3, RLS_REG_NUCLEAR=4, RLS_REG_KERNEL=5,
    RLS_REG_GROUP_LASSO=6, RLS_REG_FUSED=7
} RLSRegType;

typedef struct {
    RLSRegType type;
    double lambda, lambda2, alpha;
    int group_count, *group_sizes, *group_indices;
    RLSMatrix *D;
} RLSRegularizer;

typedef enum {
    RLS_MODEL_FIR=0, RLS_MODEL_ARX=1, RLS_MODEL_OE=2,
    RLS_MODEL_ARMAX=3, RLS_MODEL_BJ=4, RLS_MODEL_SS=5,
    RLS_MODEL_NARX=6
} RLSModelType;

typedef struct {
    RLSModelType type;
    int na, nb, nc, nd, nf, nk, nx;
    double ts;
} RLSModelOrder;

typedef struct {
    double *u, *y, *t;
    int N;
    double ts, snr;
    char *name;
} RLSData;

typedef struct {
    double *theta, *theta_true, *std_errors, *p_values;
    int n_params;
    double loss, mse, r2, aic, bic, cond_number, effective_df;
    int iterations;
    bool converged;
} RLSEstimate;

typedef struct {
    RLSRegularizer *regularizer;
    int max_iterations;
    double tolerance;
    bool center_data, scale_data, compute_stats, verbose;
    int verbosity_level, random_seed;
} RLSOptions;

typedef enum {
    RLS_SOLVER_CHOLESKY=0, RLS_SOLVER_SVD=1, RLS_SOLVER_QR=2,
    RLS_SOLVER_CG=3, RLS_SOLVER_CD=4, RLS_SOLVER_ADMM=5, RLS_SOLVER_LSQR=6
} RLSSolverType;

typedef struct {
    RLSSolverType type;
    double cg_tol, cd_tol, admm_tol, lsqr_tol;
    int cg_max_iter, cd_max_iter, admm_max_iter, lsqr_max_iter;
    double admm_rho;
} RLSSolverConfig;

typedef enum {
    RLS_LAMBDA_FIXED=0, RLS_LAMBDA_GCV=1, RLS_LAMBDA_LCURVE=2,
    RLS_LAMBDA_KFOLD_CV=3, RLS_LAMBDA_AICC=4, RLS_LAMBDA_ML=5,
    RLS_LAMBDA_STEIN=6
} RLSLambdaMethod;

typedef struct {
    RLSLambdaMethod method;
    double lambda, lambda_min, lambda_max, lambda_opt, lambda_opt_score;
    int n_lambda, k_folds;
    double *lambda_grid, *cv_scores;
} RLSLambdaSelection;

/* ---------- L3: Matrix/Vector Algebra ---------- */
RLSMatrix *rls_matrix_alloc(int rows, int cols);
RLSMatrix *rls_matrix_alloc_data(double *data, int rows, int cols);
void       rls_matrix_free(RLSMatrix *M);
RLSMatrix *rls_matrix_copy(const RLSMatrix *M);
void       rls_matrix_set(RLSMatrix *M, int i, int j, double val);
double     rls_matrix_get(const RLSMatrix *M, int i, int j);
void       rls_matrix_zero(RLSMatrix *M);
void       rls_matrix_identity(RLSMatrix *M);
void       rls_matrix_scale(RLSMatrix *M, double s);
void       rls_matrix_add_diag(RLSMatrix *M, double val);
void       rls_matrix_fill(RLSMatrix *M, double val);

RLSVector *rls_vector_alloc(int dim);
RLSVector *rls_vector_alloc_data(double *data, int dim);
void       rls_vector_free(RLSVector *v);
RLSVector *rls_vector_copy(const RLSVector *v);
void       rls_vector_set(RLSVector *v, int i, double val);
double     rls_vector_get(const RLSVector *v, int i);
void       rls_vector_zero(RLSVector *v);

double rls_vector_dot(const RLSVector *a, const RLSVector *b);
double rls_vector_nrm2(const RLSVector *v);
double rls_vector_asum(const RLSVector *v);
int    rls_vector_iamax(const RLSVector *v);
double rls_vector_norm_inf(const RLSVector *v);
void   rls_vector_axpy(RLSVector *y, double alpha, const RLSVector *x);
void   rls_vector_scal(RLSVector *v, double alpha);
void   rls_vector_copy_to(RLSVector *dst, const RLSVector *src);
void   rls_vector_sub(RLSVector *r, const RLSVector *a, const RLSVector *b);

void rls_matrix_vector_mul(RLSVector *y, const RLSMatrix *A, const RLSVector *x);
void rls_matrix_t_vector_mul(RLSVector *y, const RLSMatrix *A, const RLSVector *x);
void rls_rank1_update(RLSMatrix *A, double alpha, const RLSVector *x, const RLSVector *y);
void rls_matrix_multiply(RLSMatrix *C, const RLSMatrix *A, const RLSMatrix *B);
void rls_matrix_transpose(RLSMatrix *AT, const RLSMatrix *A);
void rls_gram_matrix(RLSMatrix *G, const RLSMatrix *A);
void rls_compute_residual(RLSVector *r, const RLSVector *y, const RLSMatrix *Phi, const RLSVector *theta);

/* ---------- L4: Factorizations ---------- */
int  rls_cholesky_decompose(RLSMatrix *A);
void rls_cholesky_solve(RLSVector *x, const RLSMatrix *L, const RLSVector *b);
int  rls_ldlt_decompose(RLSMatrix *A);
void rls_ldlt_solve(RLSVector *x, const RLSMatrix *LDLt, const RLSVector *b);
int  rls_qr_decompose(RLSMatrix *A, RLSVector *tau);
void rls_qr_multiply_q(RLSMatrix *QR, const RLSVector *tau, RLSVector *x, bool transpose);
void rls_qr_solve(RLSVector *x, const RLSMatrix *QR, const RLSVector *tau, const RLSVector *b);
int  rls_svd_decompose(RLSMatrix *U, RLSVector *S, RLSMatrix *Vt, const RLSMatrix *A);
void rls_svd_solve(RLSVector *x, const RLSMatrix *U, const RLSVector *S, const RLSMatrix *Vt, const RLSVector *b, double lambda);
int  rls_eigen_symm(RLSVector *eval, RLSMatrix *evec, const RLSMatrix *A);
int  rls_matrix_inverse_spd(RLSMatrix *Ainv, const RLSMatrix *A);
int  rls_pinv(RLSMatrix *Apinv, const RLSMatrix *A, double tol);

double rls_matrix_trace(const RLSMatrix *A);
double rls_matrix_frobenius_norm(const RLSMatrix *A);
double rls_matrix_norm_inf(const RLSMatrix *A);
double rls_cond_estimate(const RLSMatrix *A, int max_iter, double tol);
double rls_effective_df(const RLSMatrix *Phi, double lambda);
void   rls_matrix_print(const RLSMatrix *A, const char *name);
void   rls_vector_print(const RLSVector *v, const char *name);

/* Data utilities */
RLSData *rls_data_alloc(int N);
void     rls_data_free(RLSData *data);
void     rls_data_generate_sine(RLSData *data, double freq, double amp, double noise_std);
void     rls_data_generate_step(RLSData *data, double amp, double t_step, double noise_std);
void     rls_data_generate_prbs(RLSData *data, int reg_len, double amp, double noise_std);
void     rls_data_generate_random_walk(RLSData *data, double step_std, double noise_std);
void     rls_data_split(RLSData *train, RLSData *test, const RLSData *full, double train_ratio);
void     rls_data_center(RLSData *data);
void     rls_data_normalize(RLSData *data, double *u_mean, double *u_std, double *y_mean, double *y_std);
double   rls_data_estimate_snr(const RLSData *data);

RLSEstimate *rls_estimate_alloc(int n_params);
void         rls_estimate_free(RLSEstimate *est);
void         rls_estimate_compute_stats(RLSEstimate *est, const RLSMatrix *Phi, const RLSVector *y, double sigma2, double lambda);
void         rls_estimate_print(const RLSEstimate *est);

RLSOptions        rls_options_default(void);
RLSSolverConfig   rls_solver_config_default(RLSSolverType type);
RLSLambdaSelection rls_lambda_selection_default(RLSLambdaMethod method, double lambda);
RLSRegularizer    rls_regularizer_default(RLSRegType type, double lambda);
void              rls_regularizer_free(RLSRegularizer *reg);

#endif
