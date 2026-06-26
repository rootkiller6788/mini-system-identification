#include "pem_predictor.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/* ============================================================================
 * PEM Predictor — One-Step-Ahead Prediction Implementations
 *
 * Implements predictors for ARX, ARMAX, OE, BJ, and FIR model structures.
 * Each predictor computes y_hat(t|theta) given data up to time t-1.
 *
 * Key concept: The one-step-ahead predictor is the optimal (minimum-variance)
 * predictor of y(t) given past data, under the assumption that e(t) is white
 * noise. For linear models, this is also the conditional expectation.
 * ============================================================================ */

/* --- Parameter Counts --- */

int pem_arx_nparam(int na, int nb) { return na + nb; }
int pem_armax_nparam(int na, int nb, int nc) { return na + nb + nc; }
int pem_oe_nparam(int nb, int nf) { return nb + nf; }
int pem_bj_nparam(int nb, int nc, int nd, int nf) { return nb + nc + nd + nf; }

/* --- Predictor State Management --- */

PEMPredictorState* pem_predictor_state_alloc(int max_lag) {
    if (max_lag < 1) max_lag = 1;
    PEMPredictorState *state = (PEMPredictorState*)calloc(1, sizeof(PEMPredictorState));
    if (!state) return NULL;
    state->max_lag = max_lag;
    state->head = 0;
    state->eps_history = (double*)calloc((size_t)max_lag, sizeof(double));
    state->w_history = (double*)calloc((size_t)max_lag, sizeof(double));
    if (!state->eps_history || !state->w_history) {
        pem_predictor_state_free(state);
        return NULL;
    }
    return state;
}

void pem_predictor_state_free(PEMPredictorState *state) {
    if (!state) return;
    free(state->eps_history);
    free(state->w_history);
    free(state);
}

void pem_predictor_state_reset(PEMPredictorState *state) {
    if (!state) return;
    memset(state->eps_history, 0, (size_t)state->max_lag * sizeof(double));
    memset(state->w_history, 0, (size_t)state->max_lag * sizeof(double));
    state->head = 0;
}

/* Helper: store value in circular buffer at position head, advance head */
static void circ_store(double *buf, int max_lag, int *head, double value) {
    buf[*head] = value;
    *head = (*head + 1) % max_lag;
}

/* Helper: read from circular buffer at offset k (k steps ago)
 * k=1 means the most recently stored value, k=2 is one before that, etc. */
static double circ_read(const double *buf, int max_lag, int head, int k) {
    /* head points to NEXT write position. The most recent value is at (head-1).
     * For offset k (k>=1): index = (head - k) mod max_lag */
    int idx = (head - k) % max_lag;
    if (idx < 0) idx += max_lag;
    return buf[idx];
}

/* ================================================================
 * ARX Predictor
 *
 * Model: A(q) y(t) = B(q) u(t-nk) + e(t)
 * Predictor: y_hat(t) = -sum_{i=1}^{na} a_i y(t-i) + sum_{i=1}^{nb} b_i u(t-nk-i+1)
 *
 * This is linear in parameters (linear regression).
 * theta = [a_1, ..., a_na, b_1, ..., b_nb]
 *
 * Complexity: O(na + nb) per prediction step.
 * ================================================================ */

double pem_predict_arx(const double *theta, int na, int nb, int nk,
                       const double *u, const double *y, int t) {
    double y_hat = 0.0;

    /* AR part: -sum a_i y(t-i) */
    for (int i = 1; i <= na; i++) {
        if (t - i >= 0) y_hat -= theta[i - 1] * y[t - i];
    }

    /* X part: sum b_i u(t-nk-i+1) */
    for (int i = 1; i <= nb; i++) {
        int idx = t - nk - i + 1;
        if (idx >= 0) y_hat += theta[na + i - 1] * u[idx];
    }

    return y_hat;
}

double pem_residual_arx(const double *theta, int na, int nb, int nk,
                        const double *u, const double *y, int t) {
    return y[t] - pem_predict_arx(theta, na, nb, nk, u, y, t);
}

void pem_predict_arx_batch(const double *theta, int na, int nb, int nk,
                           const double *u, const double *y, int N,
                           double *y_hat) {
    for (int t = 0; t < N; t++) {
        y_hat[t] = pem_predict_arx(theta, na, nb, nk, u, y, t);
    }
}

void pem_residuals_arx(const double *theta, int na, int nb, int nk,
                       const double *u, const double *y, int N,
                       double *epsilon) {
    for (int t = 0; t < N; t++) {
        epsilon[t] = pem_residual_arx(theta, na, nb, nk, u, y, t);
    }
}

/* ================================================================
 * ARMAX Predictor
 *
 * Model: A(q) y(t) = B(q) u(t-nk) + C(q) e(t)
 *
 * Predictor:
 *   y_hat(t|theta) = -sum a_i y(t-i) + sum b_i u(t-nk-i+1)
 *                    + sum c_i eps(t-i, theta)
 *
 * where eps(t,theta) = y(t) - y_hat(t|theta).
 *
 * This is a pseudo-linear regression: the regressor includes past prediction
 * errors, which depend on theta. This makes the predictor nonlinear in
 * the parameters despite the model being linear.
 *
 * theta = [a_1..a_na, b_1..b_nb, c_1..c_nc]
 *
 * The state stores past eps values in a circular buffer.
 *
 * Complexity: O(na + nb + nc) per prediction step.
 * ================================================================ */

double pem_predict_armax(const double *theta, int na, int nb, int nc, int nk,
                         const double *u, const double *y, int t,
                         PEMPredictorState *state) {
    double y_hat = 0.0;

    /* AR part: -sum a_i y(t-i) */
    for (int i = 1; i <= na; i++) {
        if (t - i >= 0) y_hat -= theta[i - 1] * y[t - i];
    }

    /* X part: sum b_i u(t-nk-i+1) */
    for (int i = 1; i <= nb; i++) {
        int idx = t - nk - i + 1;
        if (idx >= 0) y_hat += theta[na + i - 1] * u[idx];
    }

    /* MA part: sum c_i eps(t-i, theta)
     * The c parameters start at offset na+nb in theta */
    for (int i = 1; i <= nc; i++) {
        /* Read eps(t-i) from circular buffer.
         * eps(t-1) is at offset 1, eps(t-2) at offset 2, etc.
         * But only if we've already stored values for those times. */
        if (t >= i) {
            double eps_past = circ_read(state->eps_history, state->max_lag, state->head, i);
            y_hat += theta[na + nb + i - 1] * eps_past;
        }
    }

    return y_hat;
}

void pem_predict_armax_batch(const double *theta, int na, int nb, int nc, int nk,
                             const double *u, const double *y, int N,
                             double *y_hat) {
    /* For ARMAX batch, we need state to track eps recursively.
     * Allocate temporary state with max_lag = max(na, nb+nk, nc) + 1 */
    int max_lag = na;
    if (nb + nk > max_lag) max_lag = nb + nk;
    if (nc > max_lag) max_lag = nc;
    max_lag = max_lag + 2;
    if (max_lag < 10) max_lag = 10;

    PEMPredictorState *state = pem_predictor_state_alloc(max_lag);
    if (!state) return;

    for (int t = 0; t < N; t++) {
        double yh = pem_predict_armax(theta, na, nb, nc, nk, u, y, t, state);
        y_hat[t] = yh;
        double eps = y[t] - yh;
        circ_store(state->eps_history, max_lag, &state->head, eps);
    }
    pem_predictor_state_free(state);
}

void pem_residuals_armax(const double *theta, int na, int nb, int nc, int nk,
                         const double *u, const double *y, int N,
                         double *epsilon) {
    int max_lag = na;
    if (nb + nk > max_lag) max_lag = nb + nk;
    if (nc > max_lag) max_lag = nc;
    max_lag = max_lag + 2;
    if (max_lag < 10) max_lag = 10;

    PEMPredictorState *state = pem_predictor_state_alloc(max_lag);
    if (!state) return;

    for (int t = 0; t < N; t++) {
        double yh = pem_predict_armax(theta, na, nb, nc, nk, u, y, t, state);
        epsilon[t] = y[t] - yh;
        circ_store(state->eps_history, max_lag, &state->head, epsilon[t]);
    }
    pem_predictor_state_free(state);
}

/* ================================================================
 * Output Error Predictor
 *
 * Model: y(t) = B(q)/F(q) u(t-nk) + e(t)
 *        where F(q) = 1 + f_1 q^{-1} + ... + f_nf q^{-nf}
 *
 * The predictor is:
 *   w(t,theta) = B(q)/F(q) u(t-nk)
 *   y_hat(t|theta) = w(t,theta)
 *
 * w(t) is computed recursively via the difference equation:
 *   w(t) = -f_1 w(t-1) - ... - f_nf w(t-nf) + b_1 u(t-nk) + ... + b_nb u(t-nk-nb+1)
 *
 * theta = [b_1..b_nb, f_1..f_nf]
 *
 * Key property: The OE predictor does NOT use measured outputs y, only inputs u.
 * This means the OE criterion sum(y - w)^2 can have local minima.
 *
 * Complexity: O(nb + nf) per prediction.
 * ================================================================ */

double pem_predict_oe(const double *theta, int nb, int nf, int nk,
                      const double *u, int t, PEMPredictorState *state) {
    double w = 0.0;

    /* Input part: sum b_i u(t-nk-i+1) */
    for (int i = 1; i <= nb; i++) {
        int idx = t - nk - i + 1;
        if (idx >= 0) w += theta[i - 1] * u[idx];
    }

    /* Output (filter) part: -sum f_i w(t-i) */
    for (int i = 1; i <= nf; i++) {
        if (t >= i) {
            double w_past = circ_read(state->w_history, state->max_lag, state->head, i);
            w -= theta[nb + i - 1] * w_past;
        }
    }

    return w;
}

void pem_predict_oe_batch(const double *theta, int nb, int nf, int nk,
                          const double *u, int N, double *y_hat) {
    int max_lag = nb + nk + 1;
    if (nf + 1 > max_lag) max_lag = nf + 1;
    if (max_lag < 10) max_lag = 10;

    PEMPredictorState *state = pem_predictor_state_alloc(max_lag);
    if (!state) return;

    for (int t = 0; t < N; t++) {
        double w = pem_predict_oe(theta, nb, nf, nk, u, t, state);
        y_hat[t] = w;
        circ_store(state->w_history, max_lag, &state->head, w);
    }
    pem_predictor_state_free(state);
}

void pem_residuals_oe(const double *theta, int nb, int nf, int nk,
                      const double *u, const double *y, int N,
                      double *epsilon) {
    int max_lag = nb + nk + 1;
    if (nf + 1 > max_lag) max_lag = nf + 1;
    if (max_lag < 10) max_lag = 10;

    PEMPredictorState *state = pem_predictor_state_alloc(max_lag);
    if (!state) return;

    for (int t = 0; t < N; t++) {
        double w = pem_predict_oe(theta, nb, nf, nk, u, t, state);
        epsilon[t] = y[t] - w;
        circ_store(state->w_history, max_lag, &state->head, w);
    }
    pem_predictor_state_free(state);
}

void pem_kstep_predict_oe(const double *theta, int nb, int nf, int nk,
                          const double *u, int N, int k,
                          double *y_hat) {
    /* k-step-ahead prediction for OE:
     * For t where we have data, use measured past w(t-i).
     * For t+k, only use previously PREDICTED w values,
     * not measured outputs. This is pure simulation beyond step k=1.
     *
     * For k=1: standard one-step-ahead predictor.
     * For k=N: pure simulation (infinite-step prediction).
     */
    int max_lag = nb + nk + 1;
    if (nf + 1 > max_lag) max_lag = nf + 1;
    if (max_lag < 10) max_lag = 10;

    PEMPredictorState *state = pem_predictor_state_alloc(max_lag);
    if (!state) return;

    for (int t = 0; t < N; t++) {
        double w = 0.0;

        /* Input terms always use actual u (known input) */
        for (int i = 1; i <= nb; i++) {
            int idx = t - nk - i + 1;
            if (idx >= 0) w += theta[i - 1] * u[idx];
        }

        /* Filter terms: for first k steps use actual w(t-i) if available
         * (measured/one-step), then switch to predicted w values */
        for (int i = 1; i <= nf; i++) {
            if (t >= i) {
                double w_past;
                if (i <= k && (t - i) >= 0) {
                    /* Use stored one-step predictions for recent steps */
                    w_past = circ_read(state->w_history, max_lag, state->head, i);
                } else if (t - i >= 0) {
                    /* Use previously PREDICTED multi-step values */
                    w_past = y_hat[t - i];
                } else {
                    w_past = 0.0;
                }
                w -= theta[nb + i - 1] * w_past;
            }
        }

        y_hat[t] = w;
        circ_store(state->w_history, max_lag, &state->head, w);
    }
    pem_predictor_state_free(state);
}

/* ================================================================
 * Box-Jenkins Predictor
 *
 * Model: y(t) = B(q)/F(q) u(t-nk) + C(q)/D(q) e(t)
 *
 * Predictor:
 *   w(t) = B(q)/F(q) u(t-nk)                     (deterministic part)
 *   y_hat(t) = [1 - D(q)/C(q)] y(t) + D(q)/C(q) w(t)  (stochastic part)
 *
 * Equivalently:
 *   y_hat(t) = w(t) + [1 - D(q)/C(q)] (y(t) - w(t))
 *
 * where D(q) = 1 + d_1 q^{-1} + ... + d_nd q^{-nd}
 *       C(q) = 1 + c_1 q^{-1} + ... + c_nc q^{-nc}
 *
 * theta = [b_1..b_nb, c_1..c_nc, d_1..d_nd, f_1..f_nf]
 *
 * This is the most general linear polynomial model structure used in PEM.
 * All other structures (ARX, ARMAX, OE) are special cases.
 *
 * Reference: Ljung (1999), Section 4.2 "Families of Transfer Function Models"
 * ================================================================ */

double pem_predict_bj(const double *theta, int nb, int nc, int nd, int nf, int nk,
                      const double *u, const double *y, int t,
                      PEMPredictorState *state) {
    /* Step 1: Compute deterministic part w(t) = B(q)/F(q) u(t-nk) */
    double w = 0.0;

    /* B part: sum b_i u(t-nk-i+1) */
    for (int i = 1; i <= nb; i++) {
        int idx = t - nk - i + 1;
        if (idx >= 0) w += theta[i - 1] * u[idx];
    }

    /* F part: -sum f_i w(t-i) */
    for (int i = 1; i <= nf; i++) {
        if (t >= i) {
            double w_past = circ_read(state->w_history, state->max_lag, state->head, i);
            w -= theta[nb + nc + nd + i - 1] * w_past;
        }
    }

    /* Store w in history */
    circ_store(state->w_history, state->max_lag, &state->head, w);

    /* Step 2: Compute prediction error contribution
     *
     * y_hat(t) = [1 - D(q)/C(q)] y(t) + D(q)/C(q) w(t)
     *
     * Let eps(t) = y(t) - w(t). Then:
     * y_hat(t) = w(t) + [1 - D/C] eps(t)
     *
     * Since C(q) y_hat(t) = C(q) w(t) + [C(q) - D(q)] (y(t)-w(t))
     * Actually, directly implementing:
     * y_hat(t) = w(t) + (C(q)-D(q))/C(q) * eps(t) where eps(t)=y(t)-w(t)
     *
     * Or: y_hat(t) = y(t) - D(q)/C(q) * (y(t)-w(t))
     *
     * Simplifying to the standard form used in practice:
     * eps_bar(t) = y(t) - w(t)   (deterministic residual)
     * Then: C(q) e_hat(t) = (C(q) - D(q)) eps_bar(t) + D(q) (y(t)-y_hat(t))
     *
     * The cleanest recursive implementation:
     *   e(t) = y(t) - y_hat(t)
     *   y_hat(t) = w(t) + sum_{i=1}^{nc} c_i e(t-i) - sum_{i=1}^{nd} d_i (y(t-i)-w(t-i))
     */

    double yh = w;
    int cp = nb;                          /* Start of C params */
    int dp = nb + nc;                     /* Start of D params */

    /* C part: + sum c_i * eps(t-i) where eps = y - y_hat */
    for (int i = 1; i <= nc; i++) {
        if (t >= i) {
            double eps_past = circ_read(state->eps_history, state->max_lag, state->head, i);
            yh += theta[cp + i - 1] * eps_past;
        }
    }

    /* D part: - sum d_i * eps_bar(t-i) where eps_bar = y - w
     * Actually for the standard BJ predictor:
     * y_hat(t) = w(t) + sum c_i eps(t-i) - sum d_i (y(t-i) - w(t-i))
     */
    for (int i = 1; i <= nd; i++) {
        if (t >= i) {
            double y_past = y[t - i];
            double w_past = circ_read(state->w_history, state->max_lag, state->head, i + 1);
            yh -= theta[dp + i - 1] * (y_past - w_past);
        }
    }

    return yh;
}

void pem_predict_bj_batch(const double *theta, int nb, int nc, int nd, int nf, int nk,
                          const double *u, const double *y, int N,
                          double *y_hat) {
    int max_lag = nb + nk + 1;
    if (nf + 1 > max_lag) max_lag = nf + 1;
    if (nc + 1 > max_lag) max_lag = nc + 1;
    if (nd + 1 > max_lag) max_lag = nd + 1;
    if (max_lag < 20) max_lag = 20;

    PEMPredictorState *state = pem_predictor_state_alloc(max_lag);
    if (!state) return;

    for (int t = 0; t < N; t++) {
        double yh = pem_predict_bj(theta, nb, nc, nd, nf, nk, u, y, t, state);
        y_hat[t] = yh;
        double eps = y[t] - yh;
        circ_store(state->eps_history, max_lag, &state->head, eps);
    }
    pem_predictor_state_free(state);
}

void pem_residuals_bj(const double *theta, int nb, int nc, int nd, int nf, int nk,
                      const double *u, const double *y, int N,
                      double *epsilon) {
    int max_lag = nb + nk + 1;
    if (nf + 1 > max_lag) max_lag = nf + 1;
    if (nc + 1 > max_lag) max_lag = nc + 1;
    if (nd + 1 > max_lag) max_lag = nd + 1;
    if (max_lag < 20) max_lag = 20;

    PEMPredictorState *state = pem_predictor_state_alloc(max_lag);
    if (!state) return;

    for (int t = 0; t < N; t++) {
        double yh = pem_predict_bj(theta, nb, nc, nd, nf, nk, u, y, t, state);
        epsilon[t] = y[t] - yh;
        circ_store(state->eps_history, max_lag, &state->head, epsilon[t]);
    }
    pem_predictor_state_free(state);
}

/* ================================================================
 * FIR Predictor
 *
 * Model: y(t) = B(q) u(t-nk) + e(t)
 * Predictor: y_hat(t) = sum b_i u(t-nk-i+1)
 *
 * This is the simplest case: purely feedforward, no feedback.
 * Linear regression (no recursion needed).
 *
 * Complexity: O(nb) per prediction.
 * ================================================================ */

double pem_predict_fir(const double *theta, int nb, int nk,
                       const double *u, int t) {
    double y_hat = 0.0;
    for (int i = 1; i <= nb; i++) {
        int idx = t - nk - i + 1;
        if (idx >= 0) y_hat += theta[i - 1] * u[idx];
    }
    return y_hat;
}