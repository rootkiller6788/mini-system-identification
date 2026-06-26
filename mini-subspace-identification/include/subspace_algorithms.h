#ifndef SUBSPACE_ALGORITHMS_H
#define SUBSPACE_ALGORITHMS_H
#include "subspace_core.h"

/* ============================================================================
 * Subspace Identification -- Algorithm Implementations
 *
 * Three main families of subspace identification algorithms:
 *
 * 1. N4SID (Numerical algorithms for Subspace State Space System ID)
 *    - Van Overschee & De Moor (1994, 1996)
 *    - Recovers state sequence X_i first, then estimates A,B,C,D via LS
 *    - Uses oblique projection: O_i = Y_f /_{U_f} W_p
 *    - SVD: O_i = U Sigma V^T -> Gamma_i = U_1 Sigma_1^{1/2}, X_i = Sigma_1^{1/2} V_1^T
 *
 * 2. MOESP (Multivariable Output-Error State sPace)
 *    - Verhaegen & Dewilde (1992), Verhaegen (1994)
 *    - Recovers extended observability matrix Gamma_i first
 *    - Uses LQ decomposition + SVD of projected output
 *    - PO-MOESP variant uses past outputs as instruments
 *
 * 3. CVA (Canonical Variate Analysis)
 *    - Larimore (1990), Peternell et al. (1996)
 *    - Statistical approach: maximize correlation between past and future
 *    - Uses weighting: W_1 = (Y_f Pi_{U_f^bot} Y_f^T)^{-1/2}
 *    - Equivalent to canonical correlation between past and future
 *
 * Each algorithm implements the same conceptual pipeline:
 *   Data -> Hankel -> Projection -> SVD -> Order -> Extract -> Validate
 *
 * References:
 *   Van Overschee & De Moor (1996) -- Subspace Identification, Ch. 3-4
 *   Katayama (2005) -- Subspace Methods for System Identification, Ch. 7-9
 *   Verhaegen & Verdult (2007) -- Filtering and System Identification, Ch. 10
 * ============================================================================ */

/* ============================================================================
 * N4SID Algorithm (Van Overschee & De Moor, 1994)
 *
 * Steps:
 *   1. Build Hankel matrices U_p, U_f, Y_p, Y_f from data
 *   2. Compute oblique projection O_i = Y_f /_{U_f} [U_p; Y_p]
 *   3. SVD of O_i = U Sigma V^T
 *   4. Determine order n from singular values
 *   5. Recover state: Gamma_i = U(:,1:n) * Sigma(1:n,1:n)^{1/2}
 *      and X_i = Sigma(1:n,1:n)^{1/2} * V(:,1:n)^T
 *   6. Estimate A, C from Gamma_i (shift structure)
 *   7. Estimate B, D by solving linear regression with known states
 *
 * Theorem (Consistency): Under persistence of excitation and
 *   white noise innovations, the N4SID estimates of (A,B,C,D)
 *   are strongly consistent as N -> infinity.
 * ============================================================================ */

/* Full N4SID implementation, returns SubspaceModel through result */
int subspace_n4sid(const SubspaceData *data, const SubspaceOptions *options,
                    SubspaceResult *result);

/* N4SID variant: Robust N4SID with instrumental variables */
int subspace_n4sid_iv(const SubspaceData *data, const SubspaceOptions *options,
                       SubspaceResult *result);

/* N4SID with combined deterministic-stochastic identification */
int subspace_n4sid_combined(const SubspaceData *data,
                             const SubspaceOptions *options,
                             SubspaceResult *result);

/* ============================================================================
 * MOESP Algorithm (Verhaegen & Dewilde, 1992)
 *
 * Steps (PO-MOESP variant):
 *   1. Build block Hankel matrices from past/future IO data
 *   2. Form LQ decomposition of [U_f; U_p; Y_p; Y_f]
 *   3. Extract the (3,4) block after eliminating U_f influence
 *   4. SVD of the projected output block
 *   5. Estimate order n from singular value drop
 *   6. Extended observability matrix Gamma_i from left singular vectors
 *   7. Estimate A and C from Gamma_i shift structure
 *   8. Estimate B and D from least squares on the output equation
 *
 * Theorem (Asymptotic Properties): Under mild conditions, MOESP
 *   estimates are consistent and asymptotically normal with
 *   convergence rate O(1/sqrt(N)).
 * ============================================================================ */

/* Full MOESP implementation */
int subspace_moesp(const SubspaceData *data, const SubspaceOptions *options,
                    SubspaceResult *result);

/* PO-MOESP (Past Outputs MOESP) -- uses past outputs as instruments */
int subspace_po_moesp(const SubspaceData *data, const SubspaceOptions *options,
                       SubspaceResult *result);

/* PI-MOESP (Past Inputs MOESP) -- uses past inputs as instruments */
int subspace_pi_moesp(const SubspaceData *data, const SubspaceOptions *options,
                       SubspaceResult *result);

/* ============================================================================
 * CVA Algorithm (Larimore, 1990)
 *
 * CVA computes the canonical variates between the "past" W_p
 * and the conditional future (Y_f given U_f). It solves:
 *
 *   max_{a,b} corr(a^T W_p, b^T Y_f | U_f)
 *
 * This is equivalent to the generalized eigenvalue problem:
 *   Sigma_{pf} Sigma_{ff}^{-1} Sigma_{fp} v = lambda^2 Sigma_{pp} v
 *
 * where Sigma_{pp} = E[W_p W_p^T], Sigma_{ff} = E[Y_f Y_f^T | U_f], etc.
 *
 * Steps:
 *   1. Remove U_f influence: Y_f^{res} = Y_f / U_f^bot, W_p^{res} = W_p / U_f^bot
 *   2. Compute covariances Sigma_{pp}, Sigma_{fp}, Sigma_{ff}
 *   3. Compute SVD: Sigma_{pp}^{-1/2} Sigma_{pf} Sigma_{ff}^{-1/2} = U Sigma V^T
 *   4. The canonical variates are: a_k = Sigma_{pp}^{-1/2} u_k, b_k = Sigma_{ff}^{-1/2} v_k
 *   5. States: X_i = Sigma_n V_n^T Sigma_{pp}^{-1/2} W_p
 *   6. Gamma_i = Sigma_{ff}^{1/2} U_n Sigma_n^{1/2}
 * ============================================================================ */

/* Full CVA implementation */
int subspace_cva(const SubspaceData *data, const SubspaceOptions *options,
                  SubspaceResult *result);

/* CVA with AIC-based order selection (Larimore, 1990) */
int subspace_cva_aic(const SubspaceData *data, const SubspaceOptions *options,
                      SubspaceResult *result);

/* ============================================================================
 * State Sequence Estimation (Common to all algorithms)
 * ============================================================================ */

/* Estimate state sequence using Kalman filter approach */
int subspace_kalman_states(const SubspaceModel *model,
                            const SubspaceData *data,
                            SubspaceMatrix *X_estimated);

/* Smooth state sequence using Rauch-Tung-Striebel smoother */
int subspace_rts_smoother(const SubspaceModel *model,
                           const SubspaceData *data,
                           SubspaceMatrix *X_smoothed);

#endif /* SUBSPACE_ALGORITHMS_H */
