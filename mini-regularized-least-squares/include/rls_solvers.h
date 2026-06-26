#ifndef RLS_SOLVERS_H
#define RLS_SOLVERS_H

#include "rls_core.h"

/* ============================================================================
 * Regularized Least Squares Solvers (L5: Algorithms)
 *
 * Problem: minimize (1/2)*||y - Phi*theta||_2^2 + P(theta)
 *
 * Ridge (L2):       (Phi^T Phi + lambda I) theta = Phi^T y   [Normal Equations]
 * LASSO (L1):       No closed form; solved via coordinate descent, ADMM
 * Elastic Net:      Combined L1+L2 penalty; coordinate descent
 * Group LASSO:      Block coordinate descent or ADMM
 * Kernel:           theta = Phi^T alpha, solve (K + lambda I) alpha = y
 *
 * Each solver returns an RLSEstimate with final theta, loss, and diagnostics.
 * ============================================================================ */

/* ---------- Ridge Regression (L2) ---------- */

/** Ridge via Cholesky on normal equations: (Phi^T Phi + lambda I) theta = Phi^T y.
 *  Complexity: O(np^2 + p^3/3) with n samples, p parameters.
 *  Best for: p <= 1000, well-conditioned Phi^T Phi. */
RLSEstimate *rls_solve_ridge_cholesky(const RLSMatrix *Phi, const RLSVector *y,
                                       double lambda, const RLSOptions *opt);

/** Ridge via SVD: theta = V * diag(s_i/(s_i^2+lambda)) * U^T * y.
 *  Complexity: O(np^2) or O(n^2 p) depending on dimensions.
 *  Best for: rank-deficient problems, ill-conditioned Phi. Also yields
 *  effective degrees of freedom and generalized cross-validation scores. */
RLSEstimate *rls_solve_ridge_svd(const RLSMatrix *Phi, const RLSVector *y,
                                  double lambda, const RLSOptions *opt);

/** Ridge via QR: R theta = Q^T y with Phi = Q R.
 *  More numerically stable than Cholesky for moderately ill-conditioned problems.
 *  Complexity: O(2np^2 - 2p^3/3). */
RLSEstimate *rls_solve_ridge_qr(const RLSMatrix *Phi, const RLSVector *y,
                                 double lambda, const RLSOptions *opt);

/** Ridge via Conjugate Gradient (iterative).
 *  Complexity: O(k * nnz(Phi)) per iteration.
 *  Best for: large sparse design matrices where p is large. */
RLSEstimate *rls_solve_ridge_cg(const RLSMatrix *Phi, const RLSVector *y,
                                 double lambda, const RLSOptions *opt,
                                 const RLSSolverConfig *solver);

/** Generic ridge solver: auto-selects best method based on problem size. */
RLSEstimate *rls_solve_ridge(const RLSMatrix *Phi, const RLSVector *y,
                              double lambda, const RLSOptions *opt);

/* ---------- LASSO (L1) ---------- */

/** LASSO via Coordinate Descent.
 *  Minimizes (1/2n)*||y - Phi*theta||_2^2 + lambda*||theta||_1
 *  Uses cyclical coordinate descent with active set acceleration.
 *  Complexity: O(k * n * p) for k iterations.
 *  Ref: Friedman, Hastie, Tibshirani (2010) "Regularization Paths via CD" */
RLSEstimate *rls_solve_lasso_cd(const RLSMatrix *Phi, const RLSVector *y,
                                 double lambda, const RLSOptions *opt,
                                 const RLSSolverConfig *solver);

/** LASSO via ADMM (Alternating Direction Method of Multipliers).
 *  Decomposes into: theta-update (ridge), z-update (soft-threshold).
 *  Robust for general regularizers, moderate convergence speed.
 *  Ref: Boyd et al. (2011) "Distributed Optimization via ADMM" */
RLSEstimate *rls_solve_lasso_admm(const RLSMatrix *Phi, const RLSVector *y,
                                   double lambda, const RLSOptions *opt,
                                   const RLSSolverConfig *solver);

/** Generic LASSO solver with auto method selection. */
RLSEstimate *rls_solve_lasso(const RLSMatrix *Phi, const RLSVector *y,
                              double lambda, const RLSOptions *opt);

/* ---------- Elastic Net ---------- */

/** Elastic Net via Coordinate Descent.
 *  Minimizes (1/2n)*||y - Phi*theta||_2^2 + lambda*(alpha*||theta||_1 + (1-alpha)*||theta||_2^2/2)
 *  Each coordinate update: theta_j = S(sum_i Phi_ij*r_i + theta_j, lambda*alpha) / (1 + lambda*(1-alpha)) */
RLSEstimate *rls_solve_elasticnet_cd(const RLSMatrix *Phi, const RLSVector *y,
                                      double alpha, double lambda,
                                      const RLSOptions *opt,
                                      const RLSSolverConfig *solver);

/* ---------- Group LASSO ---------- */

/** Group LASSO via Block Coordinate Descent.
 *  Minimizes (1/2)*||y - Phi*theta||_2^2 + lambda * sum_g ||theta_g||_2
 *  Block update: theta_g = (1 - lambda/||s_g||_2)_+ * s_g where s_g is the
 *  unpenalized block solution. */
RLSEstimate *rls_solve_group_lasso(const RLSMatrix *Phi, const RLSVector *y,
                                    const RLSRegularizer *reg,
                                    const RLSOptions *opt,
                                    const RLSSolverConfig *solver);

/* ---------- Fused LASSO ---------- */

/** Fused LASSO via ADMM.
 *  Minimizes (1/2)*||y - Phi*theta||_2^2 + lambda1*||theta||_1 + lambda2*||D theta||_1
 *  Uses ADMM with splitting: u = theta, v = D theta. */
RLSEstimate *rls_solve_fused_lasso(const RLSMatrix *Phi, const RLSVector *y,
                                    const RLSRegularizer *reg,
                                    const RLSOptions *opt,
                                    const RLSSolverConfig *solver);

/* ---------- LSQR (Sparse Iterative) ---------- */

/** LSQR: Iterative method for sparse least squares based on Golub-Kahan
 *  bidiagonalization. Handles regularization naturally.
 *  Complexity: O(k * nnz(Phi)) for k iterations.
 *  Best for: very large sparse problems (n > 10^5).
 *  Ref: Paige & Saunders (1982) "LSQR" ACM TOMS 8(1) */
RLSEstimate *rls_solve_lsqr(const RLSMatrix *Phi, const RLSVector *y,
                             double lambda, const RLSOptions *opt,
                             const RLSSolverConfig *solver);

/* ---------- Warm-start and Solution Path ---------- */

/** Compute solution path: solve for a sequence of lambda values.
 *  Uses warm-starting: solution for lambda_k initializes lambda_{k+1}.
 *  Returns array of estimates (caller frees each). */
RLSEstimate **rls_solution_path(const RLSMatrix *Phi, const RLSVector *y,
                                 const double *lambdas, int n_lambdas,
                                 RLSRegType reg_type, double alpha,
                                 const RLSOptions *opt);

/** Compute regularization path with decreasing lambda sequence.
 *  lambda_max = max_j |Phi_j^T y| / (n * alpha) for Elastic Net
 *  lambda_min = epsilon * lambda_max
 *  Generates n_lambdas log-spaced values. */
double *rls_lambda_path(double lambda_max, double lambda_min, int n_lambdas);

/** Compute lambda_max (smallest lambda giving all-zero solution for LASSO) */
double rls_lambda_max_lasso(const RLSMatrix *Phi, const RLSVector *y, int n_samples);

#endif
