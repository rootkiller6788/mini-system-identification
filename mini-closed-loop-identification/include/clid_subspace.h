/**
 * clid_subspace.h — Subspace Methods for Closed-Loop System Identification
 *
 * Subspace identification methods identify state-space models directly
 * from input-output data without iterative optimization. Closed-loop
 * subspace methods extend the open-loop algorithms (MOESP, N4SID, CVA)
 * by handling the correlation between input and noise.
 *
 * Key challenge: In open-loop subspace ID, we project future outputs
 * onto past I/O data. Under feedback, past inputs are correlated with
 * future noise, biasing the projection. Solutions include:
 *   - Using past reference/reference as instruments (r-based methods)
 *   - Predictor-Based Subspace ID (PBSID) — uses a high-order ARX predictor
 *   - SSARX — combines subspace with ARX pre-estimation
 *   - Two-stage/projection approach
 *
 * References:
 *   Van Overschee & De Moor (1996) "Subspace Identification for Linear Systems"
 *   Verhaegen (1994) Automatica 30(1)
 *   Chiuso & Picci (2005) Automatica 41(5)
 *   Katayama (2005) "Subspace Methods for System Identification", Ch.9
 */
#ifndef CLID_SUBSPACE_H
#define CLID_SUBSPACE_H

#include "clid_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * MOESP (Multivariable Output-Error State sPace) for closed-loop data.
 *
 * Uses the PO-MOESP scheme with instrumental variables:
 *   Z_past = [r(t-1) ... r(t-p)]^T  (past references as instruments)
 *   Y_future projected onto Z_past yields the extended observability matrix.
 *
 * The key insight: r(t) is uncorrelated with e(t), so using r as
 * instrument breaks the input-noise correlation problem.
 *
 * Algorithm steps:
 *   1. Form block-Hankel matrices U_p, U_f, Y_p, Y_f, R_p, R_f
 *   2. Compute instrumental projection: O_i = Y_f /_{R_p} [U_p; Y_p]
 *   3. SVD of weighted O_i to determine order n and observability Gamma_i
 *   4. Extract A,C from Gamma_i, then B,D,K from linear regression
 *
 * Reference: Verhaegen (1994); Van Overschee & De Moor (1996) Ch.4
 *
 * @param data      Closed-loop data with reference r(t) known
 * @param opts      Options: block size from na_max, order from nb_max
 * @param ss_out    Output: identified state-space model
 * @return          0 on success, -1 on failed SVD or rank detection
 */
int clid_subspace_moesp_cld(const CLID_Dataset *data,
                             const CLID_Options *opts,
                             CLID_StateSpace *ss_out);

/**
 * N4SID variant for closed-loop systems.
 *
 * The N4SID algorithm estimates (A,C) from the extended observability
 * matrix via projection, then (B,D,K) by linear regression.
 *
 * Closed-loop adaptation: Use a combined projection that includes
 * past references as instrumental variables to decorrelate the
 * input from the noise.  The projection is:
 *   O_i = Z_f / [R_p; U_p; Y_p]  where Z_f = [U_f; Y_f]
 *
 * Different weightings (CVA, MOESP, N4SID) yield different algorithms:
 *   CVA:  W1 = L_y^{-1/2}, W2 = I  (Canonical Variate Analysis)
 *   N4SID: W1 = I, W2 = I
 *   MOESP: W1 = I, W2 = Pi_perp(U_f)
 *
 * Reference: Van Overschee & De Moor (1996) Ch.5
 */
int clid_subspace_n4sid_cld(const CLID_Dataset *data,
                             const CLID_Options *opts,
                             CLID_StateSpace *ss_out);

/**
 * CVA (Canonical Variate Analysis) for closed-loop subspace identification.
 *
 * CVA weighting maximizes the canonical correlation between past and
 * future, making it optimal in a prediction error sense (Larimore 1990).
 *
 * In closed loop, CVA with r-based instruments produces the most
 * statistically efficient subspace estimates among the three classical
 * algorithms (MOESP, N4SID, CVA).
 *
 * Reference: Larimore (1990); Katayama (2005) Ch.9
 */
int clid_subspace_cva_cld(const CLID_Dataset *data,
                           const CLID_Options *opts,
                           CLID_StateSpace *ss_out);

/**
 * PBSID — Predictor-Based Subspace Identification for closed-loop systems.
 *
 * PBSID avoids the instrumental variable issue entirely by:
 *   1. First fitting a high-order ARX model: y(t) = B_{ARX}(q) u(t) + A_{ARX}(q) y(t)
 *      The ARX residual is approximately white even under feedback.
 *   2. Constructing the state sequence from the ARX predictor:
 *      x(t) = [y(t-1) ... y(t-na) u(t-nk) ... u(t-nk-nb+1)]^T
 *   3. Estimating (A,B,C,D) from the state sequence by LS regression.
 *
 * Key advantage: No reference signal r(t) needed! Works with only (u,y) data.
 *
 * Reference: Chiuso & Picci (2005); Chiuso (2007) Automatica
 */
int clid_subspace_pbsid(const CLID_Dataset *data,
                         const CLID_Options *opts,
                         CLID_StateSpace *ss_out);

/**
 * SSARX — Subspace method via ARX pre-estimation.
 *
 * Similar to PBSID but uses a two-stage approach:
 *   1. Estimate a high-order ARX model to get the predictor Markov parameters
 *   2. Realize a state-space model from these Markov parameters
 *      via Kung's realization algorithm (Hankel matrix SVD)
 *
 * Reference: Jansson (2003); Ljung & McKelvey (1996)
 */
int clid_subspace_ssarx(const CLID_Dataset *data,
                         const CLID_Options *opts,
                         CLID_StateSpace *ss_out);

/**
 * Order selection for closed-loop subspace identification.
 *
 * Uses information criteria corrected for the feedback structure:
 *   - SVD singular value gap detection
 *   - AIC_c: AIC with small-sample correction for closed loop
 *   - NIC (Normalized Information Criterion) adapted for CL data
 *
 * The closed-loop nature affects the noise coloring, requiring
 * modification of standard order selection criteria.
 *
 * @param data      Closed-loop data
 * @param s         Singular values from subspace projection
 * @param n_s       Number of singular values
 * @param n_order   Output: recommended model order
 * @return          0 on success
 */
int clid_subspace_order_select(const CLID_Dataset *data,
                                const double *s, int n_s,
                                int *n_order);

#ifdef __cplusplus
}
#endif

#endif /* CLID_SUBSPACE_H */
