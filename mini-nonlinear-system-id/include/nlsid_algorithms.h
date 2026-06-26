#ifndef NLSID_ALGORITHMS_H
#define NLSID_ALGORITHMS_H

#include "nlsid_core.h"
#include "nlsid_models.h"

/* ============================================================================
 * mini-nonlinear-system-id: Parameter Estimation Algorithms
 *
 * Nonlinear optimization methods for estimating model parameters from
 * input-output data. Implements the Prediction Error Method (PEM)
 * framework with various numerical optimization strategies.
 *
 * Key algorithms:
 *   - Gauss-Newton: second-order approximation via Jacobian
 *   - Levenberg-Marquardt: damped GN for robustness
 *   - Orthogonal Least Squares: basis selection for linear-in-params models
 *   - Steepest Descent with line search
 *   - Conjugate Gradient
 *   - Recursive Least Squares for online identification
 *
 * Reference:
 *   Ljung (1999) Chapters 7, 10
 *   Nocedal & Wright (2006) Numerical Optimization
 *   Fletcher (1987) Practical Methods of Optimization
 * ============================================================================ */

/* ============================================================================
 * Part 1: Cost Function and Residuals
 * ============================================================================ */

/** Compute residuals e(t,theta) = y(t) - y_hat(t|t-1, theta) for a model.
 *  Returns number of residuals computed. Caller must free e. */
int nlsid_compute_residuals(const NLSIDModel* model,
                             const NLSIDDataset* ds,
                             double** e, int* n);

/** Compute cost function value V_N(theta) for given data and model.
 *  Cost = (1/N) Σ L(e(t)), where L is the chosen loss function. */
double nlsid_compute_cost(const NLSIDModel* model,
                           const NLSIDDataset* ds,
                           const NLSIDConfig* config);

/** Compute cost with L2 regularization: V = V_data + lambda * ||theta||^2 */
double nlsid_compute_cost_regularized(const NLSIDModel* model,
                                       const NLSIDDataset* ds,
                                       const NLSIDConfig* config,
                                       double lambda);

/** Evaluate Huber loss: L(e) = 0.5*e^2 for |e|<=delta, delta*(|e|-0.5*delta) otherwise */
double nlsid_huber_loss(double e, double delta);

/** Evaluate epsilon-insensitive loss: L(e) = max(0, |e| - eps) */
double nlsid_epsilon_insensitive_loss(double e, double eps);

/* ============================================================================
 * Part 2: Numerical Differentiation (Gradient Computation)  [L4 Theorem]
 * ============================================================================ */

/** Compute numerical gradient of cost V w.r.t. parameters using central
 *  finite differences. O(n_params * n_data) complexity.
 *  Theorem: Under sufficient smoothness, error = O(h^2) for step size h. */
int nlsid_compute_gradient_fd(const NLSIDModel* model,
                               const NLSIDDataset* ds,
                               const NLSIDConfig* config,
                               double* gradient, int n_params);

/** Compute analytical gradient for NARX model with basis expansion.
 *  dV/dθ_j = -(2/N) Σ e(t) * φ_j(x(t)) where φ_j is the j-th basis. */
int nlsid_compute_gradient_narx(const NARXModel* narx,
                                 const NLSIDDataset* ds,
                                 double* gradient, int n_params);

/** Compute approximate Hessian using Gauss-Newton approximation.
 *  H ≈ J^T J where J is the Jacobian of residuals.
 *  Theorem (Gauss-Newton): For small residuals, H_GN ≈ true Hessian. */
int nlsid_compute_hessian_gn(const NLSIDModel* model,
                              const NLSIDDataset* ds,
                              const NLSIDConfig* config,
                              double** H, int n_params);

/** Compute Jacobian matrix J of residuals: J[t][j] = ∂e(t)/∂θ_j.
 *  Dimensions: n_residuals × n_params. Caller must free. */
int nlsid_compute_jacobian(const NLSIDModel* model,
                            const NLSIDDataset* ds,
                            double*** J, int* n_rows, int* n_cols);

/* ============================================================================
 * Part 3: Optimization Algorithms
 * ============================================================================ */

/** Gauss-Newton iteration: θ_{k+1} = θ_k - (J^T J)^{-1} J^T e
 *  For linear-in-parameters models (NARX with basis), this converges
 *  in one iteration. For general nonlinear models, it requires multiple steps.
 *  Complexity: O(N * n_params^2) per iteration.
 *  Reference: Ljung (1999) §10.2 */
int nlsid_optimize_gauss_newton(NLSIDModel* model,
                                 const NLSIDDataset* ds,
                                 NLSIDConfig* config,
                                 NLSIDResult* result);

/** Levenberg-Marquardt: θ_{k+1} = θ_k - (J^T J + λI)^{-1} J^T e
 *  Interpolates between Gauss-Newton (λ→0) and steepest descent (λ→∞).
 *  λ is adjusted based on whether cost decreases.
 *  Reference: Marquardt (1963), More (1978) */
int nlsid_optimize_levenberg_marquardt(NLSIDModel* model,
                                        const NLSIDDataset* ds,
                                        NLSIDConfig* config,
                                        NLSIDResult* result);

/** Steepest descent with Armijo line search for step size.
 *  θ_{k+1} = θ_k - α_k ∇V
 *  Armijo condition: V(θ_{k+1}) ≤ V(θ_k) + c₁ α_k ∇V^T p_k
 *  Complexity: O(N * n_params) per iteration.
 *  Reference: Nocedal & Wright (2006) §3.1 */
int nlsid_optimize_steepest_descent(NLSIDModel* model,
                                     const NLSIDDataset* ds,
                                     NLSIDConfig* config,
                                     NLSIDResult* result);

/** Conjugate Gradient (Fletcher-Reeves):
 *  Builds conjugate search directions for faster convergence than SD.
 *  Complexity: O(N * n_params) per iteration.
 *  Reference: Nocedal & Wright (2006) §5.2 */
int nlsid_optimize_conjugate_gradient(NLSIDModel* model,
                                       const NLSIDDataset* ds,
                                       NLSIDConfig* config,
                                       NLSIDResult* result);

/** Orthogonal Least Squares (OLS) for basis selection.
 *  For linear-in-parameters models y = Σ w_i φ_i(x), selects the most
 *  significant basis functions by orthogonalizing the regressor matrix.
 *  This addresses the structural identification problem.
 *  Theorem (OLS): Each step selects the basis with maximum error reduction
 *                  ratio (ERR). ERR_i = g_i^2 * (w_i^T w_i) / (y^T y).
 *  Reference: Chen, Billings & Luo (1989) */
int nlsid_optimize_ols(NARXModel* narx,
                        const NLSIDDataset* ds,
                        int max_bases,
                        double err_threshold,
                        NLSIDResult* result);

/** Recursive Least Squares (RLS) for online nonlinear identification.
 *  For linear-in-parameters models:
 *    K(t) = P(t-1)φ(t) / (λ + φ(t)^T P(t-1) φ(t))
 *    θ(t) = θ(t-1) + K(t) (y(t) - φ(t)^T θ(t-1))
 *    P(t) = (1/λ) (I - K(t)φ(t)^T) P(t-1)
 *  λ = forgetting factor ∈ (0, 1].
 *  Complexity: O(n_params^2) per time step.
 *  Reference: Ljung (1999) §11.2 */
int nlsid_optimize_recursive_ls(NARXModel* narx,
                                 const NLSIDDataset* ds,
                                 double forgetting_factor,
                                 double* theta_final);

/** Line search using Armijo backtracking.
 *  Finds step size α such that: f(x + αp) ≤ f(x) + c₁ α ∇f^T p.
 *  Returns the step size found. */
double nlsid_line_search_armijo(NLSIDModel* model,
                                 const NLSIDDataset* ds,
                                 const NLSIDConfig* config,
                                 const double* params_current,
                                 const double* direction,
                                 const double* gradient,
                                 double cost_current);

/** Line search using quadratic/cubic interpolation (More-Thuente).
 *  Strong Wolfe conditions: sufficient decrease + curvature condition.
 *  Returns the step size found. */
double nlsid_line_search_wolfe(NLSIDModel* model,
                                const NLSIDDataset* ds,
                                const NLSIDConfig* config,
                                const double* params_current,
                                const double* direction,
                                const double* gradient,
                                double cost_current);

/* ============================================================================
 * Part 4: Parameter Initialization
 * ============================================================================ */

/** Initialize parameters randomly: θ_j ~ N(0, sigma^2) */
void nlsid_init_random(double* theta, int n_params, double sigma,
                        unsigned int* seed);

/** Initialize NARX model parameters using linear least squares
 *  on a linear-in-parameters basis expansion. */
int nlsid_init_narx_ls(NARXModel* narx, const NLSIDDataset* ds);

/** Heuristic initialization: small random values with zero mean. */
void nlsid_init_heuristic(double* theta, int n_params, unsigned int* seed);

/* ============================================================================
 * Part 5: Main Identification Interface
 * ============================================================================ */

/** Run full nonlinear system identification:
 *  1. Initialize parameters
 *  2. Run selected optimization algorithm
 *  3. Compute fit statistics and residual tests
 *  4. Populate NLSIDResult
 *
 *  This is the main entry point for nonlinear system identification.
 *
 *  Algorithm selector:
 *    0 = Gauss-Newton
 *    1 = Levenberg-Marquardt (default)
 *    2 = Steepest Descent
 *    3 = Conjugate Gradient
 *
 *  Returns 0 on success, -1 on failure. */
int nlsid_identify(NLSIDModel* model,
                    const NLSIDDataset* estimation_data,
                    const NLSIDDataset* validation_data,
                    NLSIDConfig* config,
                    NLSIDResult* result);

/** Cross-validate model: identify on k-1 folds, test on held-out fold.
 *  Returns average fit percentage across all folds. */
double nlsid_cross_validate(NLSIDModel* model,
                             const NLSIDDataset* ds,
                             NLSIDConfig* config,
                             int n_folds);

/* ============================================================================
 * Part 6: Utility Functions
 * ============================================================================ */

/** Solve linear system Ax = b using Gaussian elimination with partial pivoting.
 *  A is n×n stored in row-major format. Returns 0 on success. */
int nlsid_solve_linear_system(double* A, double* b, int n, double* x);

/** Compute matrix inverse using Gaussian elimination.
 *  A is n×n, inverse is stored in A_inv. Returns 0 on success. */
int nlsid_matrix_inverse(double* A, int n, double* A_inv);

/** Compute condition number of matrix A (n×n) using power iteration.
 *  κ = |λ_max| / |λ_min|. Returns 0 on success. */
int nlsid_matrix_condition(const double* A, int n, double* condition);

/** Matrix-vector multiplication: y = A*x. A is n×m (row-major), x is m×1. */
void nlsid_matrix_mult(const double* A, const double* x, int n, int m,
                        double* y);

/** Matrix transpose multiply: y = A^T * x. A is n×m (row-major). */
void nlsid_matrix_transpose_mult(const double* A, const double* x, int n, int m,
                                  double* y);

/** Matrix-matrix multiply: C = A * B. A is m×k, B is k×n, C is m×n. */
void nlsid_matrix_matmul(const double* A, const double* B, int m, int k, int n,
                          double* C);

/** Cholesky decomposition: A = L * L^T. A is n×n SPD.
 *  L is lower triangular. Returns 0 on success. */
int nlsid_cholesky(const double* A, int n, double* L);

/** Solve A*x = b using Cholesky factor L (precomputed). */
void nlsid_cholesky_solve(const double* L, const double* b, int n, double* x);

#endif /* NLSID_ALGORITHMS_H */