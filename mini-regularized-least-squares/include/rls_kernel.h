#ifndef RLS_KERNEL_H
#define RLS_KERNEL_H

#include "rls_core.h"

/* ============================================================================
 * Kernel-Based Regularization for System Identification (L8: Advanced)
 *
 * Instead of regularizing theta directly, kernel methods place a Gaussian
 * prior over the impulse response: theta ~ N(0, lambda*K) where K encodes
 * smoothness, stability, and decay properties of the system.
 *
 * Key kernels (Pillonetto et al., 2010, 2014):
 *
 * 1. Stable Spline (SS):
 *    K(i,j) = beta^(i+j) * beta^max(i,j) / 2 - beta^(3*max(i,j)) / 6
 *    Encodes BIBO stability with exponential decay.
 *
 * 2. Tuned/Correlated (TC):
 *    K(i,j) = beta^max(i,j)
 *    Simpler decay kernel. Tuned variant: K(i,j) = beta^(|i-j|) * gamma^(i+j-2)
 *
 * 3. Diagonal (DI):
 *    K(i,j) = beta^i * delta(i,j)
 *    Independent decay, no correlation between coefficients.
 *
 * 4. DC (Diagonal + Correlated):
 *    K(i,j) = c * beta^(i+j) + (1-c) * beta^i * delta(i,j)
 *
 * 5. RBF (Radial Basis Function / Gaussian):
 *    K(i,j) = exp(-|i-j|^2 / (2*ell^2))
 *    Smoothness kernel for nonlinear FIR.
 *
 * The kernel hyperparameters (beta, gamma, ell, c) are typically estimated
 * via marginal likelihood maximization (Empirical Bayes).
 *
 * Ref: [Pill10] Pillonetto & De Nicolao, Automatica 46(1), 2010
 *      [Pill14] Pillonetto et al., Automatica 50(3), 2014
 * ============================================================================ */

typedef enum {
    RLS_KERNEL_SS = 0,    /* Stable Spline */
    RLS_KERNEL_TC = 1,    /* Tuned/Correlated */
    RLS_KERNEL_DI = 2,    /* Diagonal */
    RLS_KERNEL_DC = 3,    /* Diagonal + Correlated */
    RLS_KERNEL_RBF = 4,   /* Gaussian RBF */
    RLS_KERNEL_SE = 5,    /* Squared Exponential */
    RLS_KERNEL_MATERN32 = 6, /* Matern nu=3/2 */
    RLS_KERNEL_MATERN52 = 7, /* Matern nu=5/2 */
    RLS_KERNEL_CUSTOM = 8    /* User-supplied kernel function */
} RLSKernelType;

typedef struct {
    RLSKernelType type;
    double beta;       /* decay rate (0<beta<1 for stability) */
    double gamma;      /* correlation parameter for TC */
    double ell;        /* length-scale for RBF/SE/Matern */
    double c;          /* mixing weight for DC (0<=c<=1) */
    double nu;         /* smoothness for Matern family */
    double sigma_f2;   /* signal variance (scale factor) */
    int    dim;        /* kernel matrix dimension */
    double (*custom_fn)(int i, int j, void *params);  /* custom kernel */
    void  *custom_params;
} RLSKernel;

/* ---------- Kernel Matrix Construction ---------- */

/** Allocate and compute kernel matrix K of size dim x dim. */
RLSMatrix *rls_kernel_matrix(const RLSKernel *kernel);

/** Compute single kernel entry K(i,j). */
double rls_kernel_eval(const RLSKernel *kernel, int i, int j);

/** Stable Spline kernel: K(i,j) = beta^(i+j+max(i,j))/2 - beta^(3*max(i,j))/6 */
double rls_kernel_ss(int i, int j, double beta);

/** Tuned/Correlated kernel: K(i,j) = beta^max(i,j) * gamma^(|i-j|) */
double rls_kernel_tc(int i, int j, double beta, double gamma);

/** Diagonal kernel: K(i,j) = beta^i * delta(i,j) */
double rls_kernel_di(int i, int j, double beta);

/** DC kernel: K(i,j) = c*beta^(i+j) + (1-c)*beta^i*delta(i,j) */
double rls_kernel_dc(int i, int j, double beta, double c);

/** RBF kernel: K(i,j) = exp(-|i-j|^2 / (2*ell^2)) */
double rls_kernel_rbf(int i, int j, double ell);

/** Matern nu=3/2 kernel */
double rls_kernel_matern32(int i, int j, double ell);

/** Matern nu=5/2 kernel */
double rls_kernel_matern52(int i, int j, double ell);

/* ---------- Kernel Ridge Regression ---------- */

/** Solve kernel ridge regression: alpha = (K + lambda*I)^{-1} y.
 *  Then theta = Phi^T * alpha (representer theorem).
 *  This is the dual formulation, efficient when p > n. */
RLSEstimate *rls_solve_kernel_ridge(const RLSMatrix *K, const RLSVector *y,
                                     double lambda, const RLSOptions *opt);

/** Full kernel-based FIR identification:
 *  1. Build kernel matrix K from kernel parameters.
 *  2. Solve kernel ridge regression to get alpha.
 *  3. Compute impulse response estimate theta. */
RLSEstimate *rls_kernel_fir_identify(const RLSData *data, const RLSKernel *kernel,
                                      double lambda, const RLSOptions *opt);

/* ---------- Hyperparameter Optimization ---------- */

/** Marginal likelihood for kernel hyperparameters.
 *  log p(y|beta,lambda) = -0.5*y^T*(K+lambda*I)^{-1}*y - 0.5*log det(K+lambda*I) - n/2*log(2*pi)
 *  This is the key objective for Empirical Bayes. */
double rls_kernel_marginal_likelihood(const RLSMatrix *K, const RLSVector *y,
                                       double lambda);

/** Gradient of marginal likelihood w.r.t. kernel parameter beta.
 *  d/dbeta log p(y|beta) = 0.5*y^T*S^{-1}*dK/dbeta*S^{-1}*y - 0.5*tr(S^{-1}*dK/dbeta)
 *  where S = K + lambda*I. */
double rls_kernel_marginal_gradient(const RLSMatrix *K, const RLSVector *y,
                                     double lambda, const RLSMatrix *dK);

/** Optimize kernel hyperparameters via gradient-based marginal likelihood max.
 *  Uses BFGS or gradient descent. Returns optimal kernel config. */
int  rls_kernel_optimize_hyperparams(RLSKernel *kernel, const RLSData *data,
                                      double lambda, const RLSOptions *opt);

/* ---------- Kernel Utilities ---------- */

/** Free kernel structure */
void rls_kernel_free(RLSKernel *kernel);

/** Default stable spline kernel for FIR of given length */
RLSKernel rls_kernel_default_ss(int dim, double beta);

/** Default TC kernel */
RLSKernel rls_kernel_default_tc(int dim, double beta, double gamma);

/** Compute log-determinant of K + lambda*I using Cholesky.
 *  Returns log|K + lambda*I|. Used for marginal likelihood. */
double rls_kernel_logdet(const RLSMatrix *K, double lambda);

#endif
