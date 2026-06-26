#ifndef RLS_MODELS_H
#define RLS_MODELS_H

#include "rls_core.h"

/* ============================================================================
 * Model Structure Construction for System Identification (L3/L5)
 *
 * Builds the regressor matrix Phi and predictor from input-output data
 * for various model structures.
 *
 * FIR:    y(t) = b_1 u(t-1) + ... + b_nb u(t-nb) + e(t)
 *         theta = [b_1, ..., b_nb]^T
 *         phi(t) = [u(t-1), ..., u(t-nb)]^T
 *
 * ARX:    y(t) + a_1 y(t-1) + ... + a_na y(t-na) = b_1 u(t-nk) + ... + b_nb u(t-nk-nb+1) + e(t)
 *         theta = [a_1,...,a_na, b_1,...,b_nb]^T
 *         phi(t) = [-y(t-1),...,-y(t-na), u(t-nk),...,u(t-nk-nb+1)]^T
 *
 * OE:     y(t) = (B(q)/F(q)) u(t-nk) + e(t)  [pseudo-linear regression]
 *
 * ARMAX:  A(q) y(t) = B(q) u(t-nk) + C(q) e(t)
 *
 * BJ:     y(t) = [B(q)/F(q)] u(t-nk) + [C(q)/D(q)] e(t)
 *
 * SS:     x(t+1) = A x(t) + B u(t); y(t) = C x(t) + e(t)
 *         Innovation form: x(t+1) = A x(t) + B u(t) + K e(t); y(t) = C x(t) + e(t)
 *
 * Ref: [Ljung99] Chapters 4, 7, 10
 * ============================================================================ */

/* ---------- Regressor Matrix Construction ---------- */

/** Build FIR regressor matrix.
 *  Phi: (N-nb) x nb matrix.
 *  y_vec: output vector y(nb:N-1).
 *  Uses the first nb samples as initial conditions. */
void rls_build_fir_regressor(RLSMatrix *Phi, RLSVector *y_vec,
                              const RLSData *data, int nb);

/** Build ARX regressor matrix.
 *  Phi: (N-max(na,nb+nk)) x (na+nb) matrix.
 *  Columns: -y(t-1)...-y(t-na) then u(t-nk)...u(t-nk-nb+1) */
void rls_build_arx_regressor(RLSMatrix *Phi, RLSVector *y_vec,
                              const RLSData *data, const RLSModelOrder *order);

/** Build OE pseudo-linear regressor.
 *  Uses the simulated (noise-free) output for the regressor.
 *  Requires an initial parameter estimate; performs iterative refinement. */
void rls_build_oe_regressor(RLSMatrix *Phi, RLSVector *y_vec,
                             const RLSData *data, const RLSModelOrder *order,
                             const RLSVector *theta_init);

/** Build ARMAX pseudo-linear regressor.
 *  Includes estimated residuals e(t-k) in the regressor. */
void rls_build_armax_regressor(RLSMatrix *Phi, RLSVector *y_vec,
                                const RLSData *data, const RLSModelOrder *order,
                                const RLSVector *theta_init);

/** Build state-space regressor from I/O data (subspace preliminary step). */
void rls_build_ss_regressor(RLSMatrix *Phi, RLSVector *y_vec,
                             const RLSData *data, const RLSModelOrder *order);

/** Build nonlinear ARX regressor with polynomial basis expansion.
 *  Includes monomials u(t-i)^d and cross-terms u(t-i)*y(t-j). */
void rls_build_narx_regressor(RLSMatrix *Phi, RLSVector *y_vec,
                               const RLSData *data, const RLSModelOrder *order,
                               int poly_degree);

/* ---------- Model Simulation ---------- */

/** Simulate FIR model output given input sequence and parameters. */
void rls_simulate_fir(RLSVector *y_sim, const RLSData *data,
                       const RLSVector *theta, int nb);

/** Simulate ARX model output. */
void rls_simulate_arx(RLSVector *y_sim, const RLSData *data,
                       const RLSVector *theta, const RLSModelOrder *order);

/** Simulate OE model output (requires simulation of dynamics). */
void rls_simulate_oe(RLSVector *y_sim, const RLSData *data,
                      const RLSVector *theta, const RLSModelOrder *order);

/** Simulate state-space model output. */
void rls_simulate_ss(RLSVector *y_sim, const RLSData *data,
                      const RLSVector *theta, const RLSModelOrder *order);

/* ---------- Model Validation Metrics ---------- */

/** Compute prediction fit: 100*(1 - ||y - y_hat|| / ||y - mean(y)||) */
double rls_fit_percent(const RLSVector *y, const RLSVector *y_hat);

/** Mean squared error */
double rls_mse(const RLSVector *y, const RLSVector *y_hat);

/** Residual whiteness test (Ljung-Box Q statistic for auto-correlation).
 *  Returns p-value. Low p-value (<0.05) indicates model inadequacy. */
double rls_whiteness_test(const RLSVector *residuals, int max_lag);

/** Cross-correlation test between residuals and input.
 *  Returns p-value for independence hypothesis. */
double rls_independence_test(const RLSVector *residuals, const RLSVector *u, int max_lag);

/* ---------- Regressor Utilities ---------- */

/** Number of parameters for a given model order */
int rls_model_num_params(const RLSModelOrder *order);

/** Effective number of samples after accounting for delays */
int rls_model_effective_samples(const RLSData *data, const RLSModelOrder *order);

/** Print model order specification */
void rls_model_order_print(const RLSModelOrder *order);

#endif
