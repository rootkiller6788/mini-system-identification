#ifndef SUBSPACE_LINALG_H
#define SUBSPACE_LINALG_H
#include "subspace_core.h"

/* ============================================================================
 * Subspace Identification -- Linear Algebra Utilities
 *
 * Implements core numerical linear algebra operations needed by subspace
 * identification algorithms: matrix multiplication, QR factorization via
 * modified Gram-Schmidt, SVD via Golub-Kahan bidiagonalization + QR iteration,
 * Cholesky factorization, linear system solvers, eigenvalue computation
 * via the QR algorithm with double shifts.
 *
 * References:
 *   Golub & Van Loan (2013) -- Matrix Computations, 4th ed.
 *   Trefethen & Bau (1997) -- Numerical Linear Algebra
 *   Demmel (1997) -- Applied Numerical Linear Algebra
 * ============================================================================ */

/* --- BLAS-like Level 1 operations --- */
double subspace_dot(const double *x, const double *y, int n);
double subspace_nrm2(const double *x, int n);
void   subspace_axpy(double alpha, const double *x, double *y, int n);
void   subspace_scal(double alpha, double *x, int n);
double subspace_asum(const double *x, int n);

/* --- BLAS-like Level 2 operations --- */
void subspace_gemv(bool trans, double alpha, const SubspaceMatrix *A,
                    const double *x, double beta, double *y);
void subspace_ger(double alpha, const double *x, const double *y,
                   SubspaceMatrix *A);

/* --- BLAS-like Level 3 operations --- */
void subspace_gemm(bool transA, bool transB, double alpha,
                    const SubspaceMatrix *A, const SubspaceMatrix *B,
                    double beta, SubspaceMatrix *C);

/* --- Householder reflection --- */
double subspace_householder(double *x, int n);
void   subspace_householder_apply(double tau, const double *v, SubspaceMatrix *A,
                                   int row_start, int col_start);

/* --- QR factorization (Modified Gram-Schmidt with reorthogonalization) --- */
int subspace_qr_mgs(const SubspaceMatrix *A, SubspaceMatrix *Q, SubspaceMatrix *R);
int subspace_qr_householder(const SubspaceMatrix *A, SubspaceMatrix *Q,
                             SubspaceMatrix *R, double *tau);

/* --- SVD via Golub-Reinsch (bidiagonalization + implicit QR) --- */
int subspace_svd_golub_reinsch(const SubspaceMatrix *A, double *S,
                                SubspaceMatrix *U, SubspaceMatrix *Vt);
int subspace_svd_power_iteration(const SubspaceMatrix *A, int k,
                                  double *S, SubspaceMatrix *U, SubspaceMatrix *V);

/* --- Symmetric eigenvalue problem --- */
int subspace_syev(double *A_data, int n, double *eigenvalues,
                   SubspaceMatrix *eigenvectors);
int subspace_syevd(double *A_data, int n, double *eigenvalues);

/* --- LDL^T factorization for symmetric indefinite matrices --- */
int subspace_ldlt(const double *A_data, int n, double *L, double *D);

/* --- Matrix square root and inverse square root --- */
int subspace_sqrtm(const SubspaceMatrix *A, SubspaceMatrix *sqrtA);
int subspace_inv_sqrtm(const SubspaceMatrix *A, SubspaceMatrix *invSqrtA);

/* --- Pseudoinverse via SVD --- */
int subspace_pinv(const SubspaceMatrix *A, double tol, SubspaceMatrix *Aplus);

/* --- Rank computation --- */
int subspace_rank(const double *singular_values, int n, double tol);

/* --- Numerical utilities --- */
double subspace_machine_epsilon(void);
void   subspace_sort_descending(double *values, int n, int *perm);
double subspace_hypot(double a, double b);
void   subspace_givens(double a, double b, double *c, double *s);

#endif /* SUBSPACE_LINALG_H */
