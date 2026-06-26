#ifndef RLS_REGULARIZERS_H
#define RLS_REGULARIZERS_H

#include "rls_core.h"

/* ============================================================================
 * Regularization Penalty Functions and Proximal Operators
 *
 * L2 Ridge:     P(theta) = (lambda/2) * ||theta||_2^2
 *   Gradient:   lambda * theta
 *   Prox:       theta / (1 + lambda*step)
 *
 * L1 LASSO:     P(theta) = lambda * ||theta||_1
 *   Subgradient: lambda * sign(theta)
 *   Prox:       soft_threshold(theta, lambda*step)
 *
 * Elastic Net:  P(theta) = alpha*lambda*||theta||_1 + (1-alpha)*lambda/2*||theta||_2^2
 *   Combined penalty, solved via coordinate descent.
 *
 * Nuclear Norm: P(Theta) = lambda * ||Theta||_* = lambda * sum(sigma_i)
 *   Prox:       singular value soft-thresholding
 *
 * Group LASSO:  P(theta) = lambda * sum_g ||theta_g||_2
 *   Prox:       block soft-thresholding per group
 *
 * Fused LASSO:  P(theta) = lambda1*||theta||_1 + lambda2*||D*theta||_1
 *   Total variation penalty for ordered parameters.
 *
 * Ref: [Tibsh96], [Zou05], [Parikh13] Proximal Algorithms, Boyd
 * ============================================================================ */

/* --- Penalty evaluation --- */
double rls_penalty_eval(const RLSVector *theta, const RLSRegularizer *reg);
double rls_penalty_ridge(const RLSVector *theta, double lambda);
double rls_penalty_lasso(const RLSVector *theta, double lambda);
double rls_penalty_elasticnet(const RLSVector *theta, double alpha, double lambda);
double rls_penalty_nuclear(const RLSMatrix *Theta, double lambda);
double rls_penalty_group_lasso(const RLSVector *theta, const RLSRegularizer *reg);
double rls_penalty_fused(const RLSVector *theta, const RLSRegularizer *reg);

/* --- Gradient / Subgradient --- */
void rls_gradient_ridge(RLSVector *grad, const RLSVector *theta, double lambda);
void rls_subgradient_lasso(RLSVector *subg, const RLSVector *theta, double lambda);
void rls_gradient_elasticnet(RLSVector *grad, const RLSVector *theta, double alpha, double lambda);

/* --- Proximal Operators (L5: Algorithms) --- */
/** Soft-thresholding: S(x, lambda) = sign(x) * max(|x|-lambda, 0) */
double rls_soft_threshold(double x, double lambda);
void   rls_prox_lasso(RLSVector *out, const RLSVector *theta, double lambda);
void   rls_prox_ridge(RLSVector *out, const RLSVector *theta, double lambda);
void   rls_prox_elasticnet(RLSVector *out, const RLSVector *theta, double alpha, double lambda);
void   rls_prox_group_lasso(RLSVector *out, const RLSVector *theta, const RLSRegularizer *reg);

/** Singular value soft-thresholding for nuclear norm */
void rls_prox_nuclear(RLSMatrix *out, const RLSMatrix *Theta, double lambda);

/** Prox for fused LASSO (via ADMM subproblem) */
void rls_prox_fused(RLSVector *out, const RLSVector *theta, const RLSRegularizer *reg, double rho);

/* --- Effective degrees of freedom per regularizer (L4: Gauss-Markov extension) --- */
double rls_df_ridge(const RLSMatrix *Phi, double lambda);
double rls_df_lasso(const RLSMatrix *Phi, const RLSVector *theta_hat);
double rls_df_elasticnet(const RLSMatrix *Phi, const RLSVector *theta_hat, double alpha, double lambda);

/* --- Dual norm (for subgradient optimality checks) --- */
double rls_dual_norm_lasso(const RLSVector *v);
double rls_dual_norm_group_lasso(const RLSVector *v, const RLSRegularizer *reg);

/* --- KKT optimality condition check --- */
bool rls_kkt_check_lasso(const RLSVector *theta, const RLSVector *grad_loss, double lambda, double tol);
bool rls_kkt_check_elasticnet(const RLSVector *theta, const RLSVector *grad_loss, double alpha, double lambda, double tol);

#endif
