#ifndef SUBSPACE_ORDER_H
#define SUBSPACE_ORDER_H
#include "subspace_core.h"

/* ============================================================================
 * Subspace Identification -- System Order Estimation
 *
 * Determining the system order n is one of the most critical steps in
 * subspace identification. The singular values from the SVD of the
 * (weighted) projection matrix provide the primary information source.
 *
 * Theoretical basis: For a noise-free system of order n, the projection
 * matrix O_i has rank exactly n. With noise, O_i is full rank but the
 * first n singular values are significantly larger than the rest.
 *
 * Methods implemented:
 *   1. SVD Gap -- Find the maximum ratio sigma_k / sigma_{k+1}
 *   2. AIC (Akaike 1974) -- Balance model fit vs. complexity
 *   3. NIC (Bauer 2001) -- Normalized Information Criterion
 *   4. SVC (Singular Value Criterion) -- Cumulative energy threshold
 *
 * References:
 *   Bauer, D. (2001) "Order estimation for subspace methods", Automatica
 *   Akaike, H. (1974) "A new look at the statistical model identification"
 *   Stoica, P. & Selen, Y. (2004) "Model-order selection: a review"
 * ============================================================================ */

/* --- Singular Value Gap Method ---
 * Find k that maximizes sigma_k / sigma_{k+1}.
 * Most widely used method in practice. Robust when there is a clear gap. */
int subspace_order_svd_gap(const double *S, int n_sv, int max_order);

/* --- Akaike Information Criterion ---
 * AIC(k) = N * log(V_k) + 2 * k * (m + r)
 * where V_k is the variance unexplained by the k-th order model.
 * Returns the order minimizing AIC(k). */
int subspace_order_aic(const double *S, int n_sv, int N, int m, int r,
                        int max_order);

/* --- Normalized Information Criterion (Bauer, 2001) ---
 * NIC(k) = log(V_k) + (k * d_k * log(N)) / N
 * where d_k accounts for the degrees of freedom.
 * Consistent criterion: selects true order as N -> infinity. */
int subspace_order_nic(const double *S, int n_sv, int N, int m, int r,
                        int max_order);

/* --- Singular Value Criterion ---
 * Order = min{k : sum_{i=1}^k sigma_i^2 / sum_{i=1}^{n_sv} sigma_i^2 > threshold}
 * Typical threshold: 0.95 - 0.99 */
int subspace_order_svc(const double *S, int n_sv, double threshold,
                        int max_order);

/* --- Bayesian Information Criterion ---
 * BIC(k) = N * log(V_k) + k * (m + r) * log(N) */
int subspace_order_bic(const double *S, int n_sv, int N, int m, int r,
                        int max_order);

/* --- Minimum Description Length ---
 * MDL(k) = N * log(V_k) + 0.5 * k * (2*(m+r) - k) * log(N) / N */
int subspace_order_mdl(const double *S, int n_sv, int N, int m, int r,
                        int max_order);

/* --- Subspace-based Information Criterion (SBC) ---
 * Based on the ratio of successive singular values with penalty term */
int subspace_order_sbc(const double *S, int n_sv, int N, int m, int r,
                        int max_order);

/* --- Hankel Singular Value Analysis ---
 * For the deterministic part: reconstruct Gamma_i, then compute
 * the Hankel singular values (square roots of eigenvalues of
 * the product of controllability and observability Gramians). */
int subspace_order_hankel_sv(const SubspaceModel *model, double *hsv,
                              int max_order);

/* --- Balanced Truncation Order Reduction ---
 * Given a model identified at high order, reduce to lower order
 * by balanced truncation (Moore, 1981). */
int subspace_balanced_truncation(SubspaceModel *model, int reduced_order);

/* --- Model Order Consensus ---
 * Run multiple criteria and return the consensus order.
 * Consensus = simple majority of the methods that produce a result
 * within [1, max_order]. */
int subspace_order_consensus(const double *S, int n_sv, int N, int m, int r,
                              int max_order);

#endif /* SUBSPACE_ORDER_H */
