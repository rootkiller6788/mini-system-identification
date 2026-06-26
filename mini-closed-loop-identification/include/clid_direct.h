/**
 * clid_direct.h — Direct Closed-Loop Identification Methods
 *
 * The direct approach applies open-loop PEM directly to the closed-loop
 * input-output data (u,y), ignoring the presence of feedback. When the
 * noise model H(q,theta) is sufficiently flexible (correctly parameterized),
 * consistent estimates are obtained despite the feedback.
 *
 * Key insight (Ljung 1999, Theorem 13.1): The direct method gives
 * consistent estimates iff the noise model H(q,theta) contains the true
 * noise model.
 *
 * References:
 *   Ljung (1999) Ch.13.4
 *   Forssell & Ljung (1999) Automatica 35(7)
 */
#ifndef CLID_DIRECT_H
#define CLID_DIRECT_H

#include "clid_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Direct ARX identification on closed-loop data.
 *
 * Model: A(q) y(t) = B(q) u(t) + e(t)
 * This is the simplest direct method. Biased unless true system is in model
 * set and noise is white (rare in closed loop).
 *
 * Algorithm: Least-squares solution to the linear regression:
 *   y(t) = phi^T(t) theta + e(t)
 *
 * Solves normal equations: (Phi^T Phi) theta_hat = Phi^T Y
 * Complexity: O(N * (na+nb)^2)
 */
int clid_direct_arx(const CLID_Dataset *data,
                    const CLID_Options *opts,
                    CLID_TransferFcn *tf_out);

/**
 * Direct ARMAX via Prediction Error Method (Gauss-Newton).
 *
 * Model: A(q) y(t) = B(q) u(t) + C(q) e(t)
 * The MA part C(q) absorbs noise correlations caused by feedback,
 * making this method consistent when true noise is in model set.
 *
 * References: Ljung (1999) Section 10.2, Section 13.4
 */
int clid_direct_armax(const CLID_Dataset *data,
                      const CLID_Options *opts,
                      CLID_Estimate *est_out);

/**
 * Direct Output Error (OE) identification.
 *
 * Model: y(t) = [B(q)/F(q)] u(t) + e(t)
 * No noise model - makes OE generally inconsistent in closed loop
 * unless noise is negligible or special conditions hold.
 *
 * Uses simulation-based prediction (not one-step-ahead).
 * Complexity: O(N * (nf+nb)^2 * max_iter)
 */
int clid_direct_oe(const CLID_Dataset *data,
                   const CLID_Options *opts,
                   CLID_Estimate *est_out);

/**
 * Direct Box-Jenkins (BJ) identification.
 *
 * Model: y(t) = [B(q)/F(q)] u(t) + [C(q)/D(q)] e(t)
 * Most flexible polynomial model - plant and noise are independent.
 * Gold standard for direct closed-loop identification because it
 * can capture arbitrary noise correlation from feedback.
 *
 * Reference: Box, Jenkins, Reinsel (1994); Ljung (1999) Section 4.2
 */
int clid_direct_bj(const CLID_Dataset *data,
                   const CLID_Options *opts,
                   CLID_Estimate *est_out);

/**
 * Direct state-space identification via Prediction Error Method.
 *
 * Uses innovations form with Kalman gain K to absorb feedback correlation -
 * the state-space equivalent of having a flexible noise model.
 *
 * Reference: Ljung (1999) Section 7.4; McKelvey et al. (1996)
 */
int clid_direct_ss(const CLID_Dataset *data,
                   const CLID_Options *opts,
                   CLID_StateSpace *ss_out);

/**
 * Check whether the direct method will be consistent for given setup.
 *
 * Theorem (Ljung 1999, Thm 13.1):
 *   The direct PEM estimate is consistent iff the noise model H(q,theta)
 *   contains the true noise model and the dataset is informative enough.
 *
 * Checks: noise model adequacy, PE condition, no algebraic loop.
 */
CLID_Identifiability clid_direct_consistency_check(const CLID_Dataset *data,
                                                    const CLID_FeedbackLoop *feedback,
                                                    const CLID_Options *opts);

/**
 * Compute asymptotic bias when noise model is misspecified.
 *
 * Formula (Ljung 1999, Eq. 13.56):
 *   theta* = argmin integral |G0 - G(theta) + B(theta,omega)|^2 * Phi_u / |H|^2 domega
 *   where B captures the bias from feedback correlation Phi_eu != 0.
 */
int clid_direct_bias_compute(const CLID_TransferFcn *true_plant,
                              const CLID_TransferFcn *true_noise,
                              const CLID_FeedbackLoop *feedback,
                              const CLID_Estimate *estimated,
                              CLID_BiasReport *report_out);

/**
 * Direct identification with data prefiltering.
 *
 * Prefiltering modifies frequency weighting of PEM criterion.
 * In closed loop, prefiltering can: emphasize high-SNR bands,
 * reduce bias in specific frequency ranges, or approximate
 * the indirect method when L = 1/(1+C*G_hat).
 *
 * Reference: Ljung (1999) Section 13.5; Gevers (1993)
 */
int clid_direct_with_prefilter(const CLID_Dataset *data,
                                const CLID_Options *opts,
                                const double *filter_num, int filter_num_order,
                                const double *filter_den, int filter_den_order,
                                CLID_TransferFcn *tf_out);

#ifdef __cplusplus
}
#endif

#endif /* CLID_DIRECT_H */
