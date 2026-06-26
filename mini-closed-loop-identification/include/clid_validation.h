/**
 * clid_validation.h — Model Validation for Closed-Loop Identification
 *
 * Model validation in closed loop differs from open-loop validation
 * because standard residual tests are affected by feedback. A good
 * model in open-loop may fail closed-loop validation and vice versa.
 *
 * The fundamental principle (Ljung 1999, Sec. 16.5):
 *   In closed loop, the residual epsilon(t,theta_hat) should be
 *   uncorrelated with the reference r(t-tau) for all tau, but
 *   NOT necessarily uncorrelated with u(t-tau) — feedback
 *   correlation may remain even for a perfect model.
 *
 * Validation tests:
 *   1. Residual whiteness (adapted for CL)
 *   2. Cross-correlation: epsilon(t) vs r(t-tau) — MUST be zero
 *   3. Cross-correlation: epsilon(t) vs u(t-tau) — MAY be nonzero
 *   4. Closed-loop stability of identified model with controller
 *   5. Frequency-domain validation using spectral analysis
 *   6. Cross-validation on independent dataset
 *
 * References:
 *   Ljung (1999) Ch.16
 *   Bombois et al. (2001) "Least costly identification experiment for control"
 *   Hjalmarsson (2005) Automatica 41(3)
 */
#ifndef CLID_VALIDATION_H
#define CLID_VALIDATION_H

#include "clid_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Residual whiteness test adapted for closed-loop data.
 *
 * Computes the Ljung-Box Q statistic on the prediction error sequence:
 *   Q = N(N+2) * SUM_{tau=1}^{m} rho_epsilon(tau)^2 / (N-tau)
 * where rho_epsilon(tau) is the sample autocorrelation at lag tau.
 *
 * Under the null hypothesis of whiteness, Q ~ chi^2(m - p) where
 * p is the number of estimated parameters.
 *
 * In closed loop, a well-estimated flexible noise model should produce
 * white residuals. If the noise model is too simple, residuals will
 * show autocorrelation due to unmodeled feedback dynamics.
 *
 * @param pe      Prediction error sequence from identified model
 * @return        0 = residuals are white (p > 0.05), 1 = not white
 */
int clid_validate_residual_whiteness(CLID_PredictionError *pe);

/**
 * Cross-correlation test: prediction error vs external reference.
 *
 * Tests: E[epsilon(t) r(t-tau)] = 0 for tau = -20,...20
 *
 * THIS IS THE KEY CLOSED-LOOP VALIDATION TEST.
 * Even if the model is imperfect, epsilon(t) should be uncorrelated
 * with the external excitation r(t). If cross-correlation is nonzero,
 * the model has NOT captured all dynamics from r to y — the model is
 * invalid for control design (Ljung 1999, Sec. 16.5).
 *
 * @param epsilon    Prediction errors
 * @param N          Number of samples
 * @param r          Reference signal
 * @param max_lag    Maximum correlation lag to test
 * @param p_value    Output: p-value of joint significance test
 * @return           0 = passes (uncorrelated), 1 = fails (correlated)
 */
int clid_validate_crosscorr_ref(const double *epsilon, int N,
                                 const double *r, int max_lag,
                                 double *p_value);

/**
 * Cross-correlation test: prediction error vs input.
 *
 * Tests: E[epsilon(t) u(t-tau)] = 0 for tau = -20,...,20
 *
 * WARNING: In closed loop, this test may FAIL even for a perfect model,
 * because feedback creates correlation between u(t) and e(t-tau).
 * A nonzero cross-correlation here does NOT necessarily indicate
 * a bad model — it may indicate the feedback path.
 *
 * However, for indirect or two-stage methods, this test should pass
 * because those methods explicitly decorrelate u from the noise.
 *
 * @param epsilon    Prediction errors
 * @param N          Number of samples
 * @param u          Input signal
 * @param max_lag    Maximum correlation lag
 * @param p_value    Output: p-value
 * @return           0 = uncorrelated, 1 = correlated
 */
int clid_validate_crosscorr_input(const double *epsilon, int N,
                                   const double *u, int max_lag,
                                   double *p_value);

/**
 * Validate closed-loop stability of identified model with controller.
 *
 * Computes the closed-loop poles of the interconnection G_hat * C:
 *   poles_CL = roots( 1 + C(q) G_hat(q) = 0 )
 *
 * If any closed-loop pole has magnitude >= 1 (or real part >= 0 for CT),
 * the identified model predicts an unstable closed loop — which is
 * a strong indicator of model mismatch if the real system is stable.
 *
 * Note: This is a NECESSARY condition for a good model, not sufficient.
 * A stable closed loop does not guarantee the model is good.
 *
 * @param plant_hat    Identified plant model
 * @param controller   Known (or assumed) controller
 * @param stability    Output: 1 = stable, 0 = unstable
 * @param max_pole_mag Output: maximum closed-loop pole magnitude
 * @return             0 on success
 */
int clid_validate_stability(const CLID_TransferFcn *plant_hat,
                             const CLID_Controller *controller,
                             int *stability,
                             double *max_pole_mag);

/**
 * Frequency-domain model validation.
 *
 * Compares the frequency response of the identified model with a
 * nonparametric ETFE (Empirical Transfer Function Estimate) from
 * the data.  Computes:
 *   J_freq = (1/M) SUM_{k=1}^{M} |G_hat(jw_k) - G_etfe(jw_k)|^2 / Phi_y(w_k)
 *
 * The ETFE is unbiased but has high variance; the identified model
 * should lie within the ETFE confidence bounds.
 *
 * Reference: Ljung (1999) Section 16.4; Pintelon & Schoukens (2012)
 *
 * @param data         Closed-loop data
 * @param plant_hat    Identified plant
 * @param n_freqs      Number of frequency points
 * @param J_freq       Output: frequency-domain cost
 * @return             0 on success
 */
int clid_validate_frequency(const CLID_Dataset *data,
                             const CLID_TransferFcn *plant_hat,
                             int n_freqs,
                             double *J_freq);

/**
 * Cross-validation for closed-loop models.
 *
 * Splits data into estimation set (N_est samples) and validation set
 * (N_val samples). Computes the prediction error on the validation set:
 *   V_val = (1/N_val) SUM_{t=1}^{N_val} epsilon_val^2(t, theta_hat)
 *
 * In closed loop, cross-validation should be done on datasets from
 * DIFFERENT experiments (or with different excitation) to avoid
 * overfitting the specific feedback path.
 *
 * @param data         Full dataset
 * @param est          Estimated model
 * @param split_ratio  Fraction for estimation (e.g., 0.7)
 * @param v_val        Output: validation cost
 * @param fit_val      Output: NRMSE fit on validation data
 * @return             0 on success
 */
int clid_validate_crossval(const CLID_Dataset *data,
                            const CLID_Estimate *est,
                            double split_ratio,
                            double *v_val,
                            double *fit_val);

/**
 * Uncertainty region estimation via inverse chi-squared test.
 *
 * Computes the ellipsoidal uncertainty region in parameter space:
 *   U_alpha = {theta : (theta - theta_hat)^T P^{-1} (theta - theta_hat) <= chi2_alpha(p)}
 *
 * where P is the estimated parameter covariance matrix.
 *
 * In closed loop, the covariance P must account for the correlation
 * structure induced by feedback. Uses the sandwich formula:
 *   P = (1/N) * H^{-1} * J * H^{-1}
 * where H = Hessian of V_N, J = E[psi(t) psi^T(t)] * sigma_e^2
 *
 * Reference: Ljung (1999) Section 9.5; Bombois et al. (2001)
 */
int clid_validate_uncertainty(const CLID_Estimate *est,
                               const CLID_Dataset *data,
                               double confidence,
                               CLID_UncertaintyRegion *ur_out);

/**
 * Control-relevant validation: assess model quality for controller design.
 *
 * A model that fits well in open-loop sense may be poor for control.
 * The control-relevant criterion (Gevers 1993; Hjalmarsson 2005):
 *   J_CR = || (G_hat - G_0) * W_c ||_2
 * where W_c depends on the control objective (e.g., W_c = C*S for tracking).
 *
 * If controller C is known, computes the estimated control performance
 * degradation due to model error.
 *
 * Reference: Gevers (1993); Hjalmarsson (2005)
 *
 * @param plant_hat     Identified plant
 * @param controller    Controller to be used
 * @param data          Data for nonparametric reference G_etfe
 * @param perf_degrad   Output: estimated performance degradation (%)
 * @return              0 on success
 */
int clid_validate_control_relevance(const CLID_TransferFcn *plant_hat,
                                     const CLID_Controller *controller,
                                     const CLID_Dataset *data,
                                     double *perf_degrad);

#ifdef __cplusplus
}
#endif

#endif /* CLID_VALIDATION_H */
