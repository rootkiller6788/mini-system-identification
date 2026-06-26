/**
 * clid_iv.h — Instrumental Variable Methods for Closed-Loop Identification
 *
 * Instrumental Variable (IV) methods address the fundamental problem of
 * closed-loop identification: the correlation between input u(t) and
 * noise e(t) caused by feedback.  By using instruments z(t) that are
 * correlated with the input but uncorrelated with the noise, IV methods
 * produce consistent estimates without requiring a correct noise model.
 *
 * Instrument condition: E[z(t) e(t)] = 0  (instrument-noise orthogonality)
 *                       E[z(t) phi^T(t)] full rank  (relevance)
 *
 * In closed loop, the external reference r(t) is a natural instrument
 * since r(t) is uncorrelated with e(t) but correlated with u(t) through
 * the feedback path.
 *
 * References:
 *   Young (2011) "Recursive Estimation and Time-Series Analysis"
 *   Soderstrom & Stoica (1989) "System Identification", Ch.8
 *   Gilson & Van den Hof (2005) Automatica 41(3)
 *   Ljung (1999) Section 7.6
 */
#ifndef CLID_IV_H
#define CLID_IV_H

#include "clid_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Basic IV method for closed-loop ARX identification.
 *
 * Model: A(q) y(t) = B(q) u(t) + v(t)  (v = colored noise, uncorrelated with r)
 *
 * IV estimate solves: [ (1/N) SUM_t z(t) phi^T(t) ] * theta_hat_IV = 0
 * in the overdetermined case giving:
 *   theta_hat_IV = (Z^T Phi)^{-1} Z^T Y
 *
 * where Z = [z(1) ... z(N)]^T is the instrument matrix and
 * Phi = [phi(1) ... phi(N)]^T is the regression matrix.
 *
 * Uses r(t-lag) as instruments: z(t) = [r(t-lag), ..., r(t-lag-na-nb+1)]^T
 *
 * Complexity: O(N * (na+nb) * (na+nb+lag))
 */
int clid_iv_basic(const CLID_Dataset *data,
                  const CLID_Options *opts,
                  CLID_TransferFcn *tf_out);

/**
 * IV4 method — Four-step Instrumental Variable algorithm.
 *
 * Step 1: Basic IV to get initial theta1 (using r as instrument)
 * Step 2: Generate auxiliary model output: y_sim = G(theta1) u
 *         Filter y_sim using AR model of noise
 * Step 3: Extended IV with simulated instrument:
 *         z(t) = [y_sim(t-1), ..., y_sim(t-na), u(t-nk), ..., u(t-nk-nb+1)]^T
 *         Re-estimate theta2 using IV with these refined instruments
 * Step 4: Iterate Steps 2-3 until convergence or compute optimal IV.
 *
 * The IV4 method is optimal among IV methods when the noise model is
 * correctly specified (Young 2011, Theorem 7.1).
 *
 * Complexity: O(iv4_iters * N * (na+nb)^2)
 */
int clid_iv4(const CLID_Dataset *data,
             const CLID_Options *opts,
             CLID_Estimate *est_out);

/**
 * Refined Instrumental Variable (RIV) — iterative IV with noise estimation.
 *
 * Extends IV4 by simultaneously estimating the noise model parameters,
 * then using the noise-filtered auxiliary model to generate optimal
 * instruments at each iteration.
 *
 * The refined instrument is:
 *   z_opt(t) = C^{-1}(q) * [G(q) applied to the auxiliary model]
 *
 * This yields asymptotically efficient estimates (reaches Cramer-Rao bound)
 * when the model structure is correct.
 *
 * Reference: Young (2011) Ch.7; Young & Jakeman (1979)
 */
int clid_iv_refined(const CLID_Dataset *data,
                    const CLID_Options *opts,
                    CLID_Estimate *est_out);

/**
 * Compute optimal instruments for a given model structure.
 *
 * The optimal instrument (reaching the Cramer-Rao lower bound) is:
 *   z_opt(t) = H^{-1}(q, theta) * [psi_u(t) ; psi_e(t)]
 * where psi_u and psi_e are the gradients of the predictor w.r.t.
 * the plant and noise parameters respectively.
 *
 * This computes the asymptotic covariance achievable with IV:
 *   Cov(theta_hat_IV) = (1/N) * [E{z phi^T}]^{-1} * E{zz^T} * sigma_e^2 * [E{phi z^T}]^{-1}
 *
 * @param model     Current model estimate (for gradient computation)
 * @param data      Data (for computing expectations empirically)
 * @param cov_out   Output: asymptotic covariance of IV estimate
 */
int clid_iv_optimal_instruments(const CLID_Estimate *model,
                                 const CLID_Dataset *data,
                                 CLID_AsymptoticCov *cov_out);

/**
 * Young-Wahlberg IV approach — uses delayed inputs as instruments.
 *
 * When external reference r(t) is NOT available (only u,y known),
 * delayed inputs u(t-tau) can serve as instruments provided tau is
 * large enough that u(t-tau) is uncorrelated with e(t).
 *
 * This works because:
 *   u(t-tau) is correlated with u(t) (through system dynamics)
 *   u(t-tau) is uncorrelated with e(t) for tau > 0 (causality)
 *
 * The Wahlberg-Johansson approach uses the delayed input vector.
 *
 * Reference: Wahlberg (1989); Johansson (1994)
 */
int clid_iv_young_wahlberg(const CLID_Dataset *data,
                            const CLID_Options *opts,
                            CLID_TransferFcn *tf_out);

/**
 * Check IV consistency conditions.
 *
 * Requirements for consistent IV estimate:
 *   1. E[z(t) v(t)] = 0  (instrument-noise orthogonality)
 *   2. rank( E[z(t) phi^T(t)] ) = dim(theta)  (full rank)
 *   3. E[z(t) z^T(t)] is nonsingular
 *
 * Returns detailed diagnostic on which conditions are (un)satisfied.
 */
CLID_Identifiability clid_iv_consistency_check(const CLID_Dataset *data,
                                                const CLID_Options *opts);

#ifdef __cplusplus
}
#endif

#endif /* CLID_IV_H */
