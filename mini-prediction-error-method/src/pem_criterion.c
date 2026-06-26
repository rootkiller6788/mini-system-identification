#include "pem_criterion.h"
#include "pem_predictor.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/* ============================================================================
 * PEM Criterion — Loss Functions, Criterion, Gradient, and Hessian
 *
 * Implements the core computations needed by the PEM optimizer:
 *   V_N(theta) = (1/N) sum l(eps(t,theta))
 *   g(theta) = dV/d(theta)                       [gradient]
 *   H(theta) ~ (1/N) sum psi(t) psi^T(t)         [Gauss-Newton Hessian]
 *
 * Reference: Ljung (1999), Chapter 7 "Computing the Estimate"
 * ============================================================================ */

/* ================================================================
 * Loss Functions
 * ================================================================ */

double pem_loss_eval(double eps, PEMLossFunction loss_type, double param) {
    switch (loss_type) {
        case PEM_LOSS_QUADRATIC:
            return 0.5 * eps * eps;
        case PEM_LOSS_ABSOLUTE:
            return fabs(eps);
        case PEM_LOSS_HUBER:
            /* Huber: quadratic for |eps| <= param, linear beyond
             * l(eps) = 0.5*eps^2              if |eps| <= delta
             *        = delta*(|eps| - 0.5*delta)  if |eps| > delta */
            if (fabs(eps) <= param) return 0.5 * eps * eps;
            return param * (fabs(eps) - 0.5 * param);
        case PEM_LOSS_VAPNIK:
            /* Vapnik epsilon-insensitive: l(eps) = max(0, |eps| - epsilon) */
            {
                double d = fabs(eps) - param;
                return (d > 0.0) ? d : 0.0;
            }
        default:
            return 0.5 * eps * eps;
    }
}

double pem_loss_derivative(double eps, PEMLossFunction loss_type, double param) {
    switch (loss_type) {
        case PEM_LOSS_QUADRATIC:
            return eps;
        case PEM_LOSS_ABSOLUTE:
            return (eps > 0.0) ? 1.0 : ((eps < 0.0) ? -1.0 : 0.0);
        case PEM_LOSS_HUBER:
            if (fabs(eps) <= param) return eps;
            return (eps > 0.0) ? param : -param;
        case PEM_LOSS_VAPNIK:
            if (fabs(eps) <= param) return 0.0;
            return (eps > 0.0) ? 1.0 : -1.0;
        default:
            return eps;
    }
}

/* ================================================================
 * Criterion Functions
 *
 * V_N(theta) = (1/(2N)) * sum eps(t,theta)^2
 *
 * The 1/2 factor makes the derivative simpler: l'(eps) = eps.
 * ================================================================ */

double pem_criterion_arx(const double *theta, int na, int nb, int nk,
                         const PEMData *data) {
    if (!data || data->N <= 0) return 0.0;
    double sum_sq = 0.0;
    int N = data->N;
    for (int t = 0; t < N; t++) {
        double eps = pem_residual_arx(theta, na, nb, nk, data->u, data->y, t);
        sum_sq += eps * eps;
    }
    return 0.5 * sum_sq / (double)N;
}

/* Helper: circular buffer operations (local copies for this translation unit) */
static void circ_store_local(double *buf, int max_lag, int *head, double value) {
    buf[*head] = value;
    *head = (*head + 1) % max_lag;
}
__attribute__((unused))
static double circ_read_local(const double *buf, int max_lag, int head, int k) {
    int idx = (head - k) % max_lag;
    if (idx < 0) idx += max_lag;
    return buf[idx];
}

double pem_criterion_armax(const double *theta, int na, int nb, int nc, int nk,
                           const PEMData *data) {
    if (!data || data->N <= 0) return 0.0;
    int N = data->N;
    int max_lag = na;
    if (nb + nk > max_lag) max_lag = nb + nk;
    if (nc > max_lag) max_lag = nc;
    max_lag += 2;
    if (max_lag < 10) max_lag = 10;

    PEMPredictorState *state = pem_predictor_state_alloc(max_lag);
    if (!state) return 1e100;

    double sum_sq = 0.0;
    for (int t = 0; t < N; t++) {
        double yh = pem_predict_armax(theta, na, nb, nc, nk, data->u, data->y, t, state);
        double eps = data->y[t] - yh;
        sum_sq += eps * eps;
        circ_store_local(state->eps_history, max_lag, &state->head, eps);
    }
    pem_predictor_state_free(state);
    return 0.5 * sum_sq / (double)N;
}

double pem_criterion_oe(const double *theta, int nb, int nf, int nk,
                        const PEMData *data) {
    if (!data || data->N <= 0) return 0.0;
    int N = data->N;
    int max_lag = nb + nk + 1;
    if (nf + 1 > max_lag) max_lag = nf + 1;
    if (max_lag < 10) max_lag = 10;

    PEMPredictorState *state = pem_predictor_state_alloc(max_lag);
    if (!state) return 1e100;

    double sum_sq = 0.0;
    for (int t = 0; t < N; t++) {
        double w = pem_predict_oe(theta, nb, nf, nk, data->u, t, state);
        double eps = data->y[t] - w;
        sum_sq += eps * eps;
        circ_store_local(state->w_history, max_lag, &state->head, w);
    }
    pem_predictor_state_free(state);
    return 0.5 * sum_sq / (double)N;
}

double pem_criterion_bj(const double *theta, int nb, int nc, int nd, int nf, int nk,
                        const PEMData *data) {
    if (!data || data->N <= 0) return 0.0;
    int N = data->N;
    int max_lag = nb + nk + 1;
    if (nf + 1 > max_lag) max_lag = nf + 1;
    if (nc + 1 > max_lag) max_lag = nc + 1;
    if (nd + 1 > max_lag) max_lag = nd + 1;
    if (max_lag < 20) max_lag = 20;

    PEMPredictorState *state = pem_predictor_state_alloc(max_lag);
    if (!state) return 1e100;

    double sum_sq = 0.0;
    for (int t = 0; t < N; t++) {
        double yh = pem_predict_bj(theta, nb, nc, nd, nf, nk, data->u, data->y, t, state);
        double eps = data->y[t] - yh;
        sum_sq += eps * eps;
        circ_store_local(state->eps_history, max_lag, &state->head, eps);
    }
    pem_predictor_state_free(state);
    return 0.5 * sum_sq / (double)N;
}

double pem_criterion_general(const double *eps, int N,
                             PEMLossFunction loss_type, double param) {
    if (N <= 0) return 0.0;
    double sum = 0.0;
    for (int t = 0; t < N; t++)
        sum += pem_loss_eval(eps[t], loss_type, param);
    return sum / (double)N;
}

/* ================================================================
 * Gradient Computation
 *
 * g_k = dV/d(theta_k) = -(1/N) * sum_t eps(t,theta) * psi_k(t,theta)
 *
 * where psi_k(t,theta) = -d(eps(t,theta))/d(theta_k) is the negative
 * gradient of the prediction error (pseudo-regressor).
 *
 * For quadratic loss: g = -(1/N) * Phi^T * epsilon
 * ================================================================ */

void pem_gradient_arx(const double *theta, int na, int nb, int nk,
                      const PEMData *data, double *g) {
    /* ARX: eps(t) = y(t) - phi^T(t) theta
     * psi(t) = phi(t) = [-y(t-1),...,-y(t-na), u(t-nk),...,u(t-nk-nb+1)]^T
     * g = -(1/N) sum eps(t) * phi(t)
     *
     * This is O(N * (na+nb)) for gradient computation. */
    int npar = na + nb;
    int N = data->N;

    /* Zero gradient */
    for (int i = 0; i < npar; i++) g[i] = 0.0;

    for (int t = 0; t < N; t++) {
        double eps = pem_residual_arx(theta, na, nb, nk, data->u, data->y, t);
        /* psi for a_i: -y(t-i) */
        for (int i = 1; i <= na; i++) {
            if (t - i >= 0) g[i - 1] -= eps * (-data->y[t - i]);
        }
        /* psi for b_i: u(t-nk-i+1) */
        for (int i = 1; i <= nb; i++) {
            int idx = t - nk - i + 1;
            if (idx >= 0) g[na + i - 1] -= eps * data->u[idx];
        }
    }

    /* Scale by 1/N */
    double invN = 1.0 / (double)N;
    for (int i = 0; i < npar; i++) g[i] *= invN;
}

void pem_gradient_armax(const double *theta, int na, int nb, int nc, int nk,
                        const PEMData *data, double *g) {
    /* ARMAX gradient via filtered pseudo-regressors.
     *
     * For ARMAX, d(eps)/d(theta_k) = psi_k(t) where
     * C(q) psi_k(t) = phi_k(t) depends on which parameter theta_k is:
     *   For a_i: psi_{a_i}(t) filtered through C(q): C(q) psi_{a_i}(t) = -y(t-i)
     *   For b_i: psi_{b_i}(t) filtered through C(q): C(q) psi_{b_i}(t) = u(t-nk-i+1)
     *   For c_i: psi_{c_i}(t) filtered through C(q): C(q) psi_{c_i}(t) = eps(t-i)
     *
     * This gives: g_k = -(1/N) * sum eps(t) * psi_k(t)
     *
     * Numerical approach: Use finite differences if C(q) filtering is complex,
     * but for small nc, we implement the exact recursive filtering. */
    int npar = na + nb + nc;
    int N = data->N;

    for (int i = 0; i < npar; i++) g[i] = 0.0;

    /* First, compute all residuals */
    double *eps = (double*)malloc((size_t)N * sizeof(double));
    if (!eps) return;

    /* Compute residuals using ARMAX predictor */
    int max_lag = na;
    if (nb + nk > max_lag) max_lag = nb + nk;
    if (nc > max_lag) max_lag = nc;
    max_lag += 2;
    if (max_lag < 10) max_lag = 10;

    PEMPredictorState *state = pem_predictor_state_alloc(max_lag);
    if (!state) { free(eps); return; }

    for (int t = 0; t < N; t++) {
        double yh = pem_predict_armax(theta, na, nb, nc, nk, data->u, data->y, t, state);
        eps[t] = data->y[t] - yh;
        circ_store_local(state->eps_history, max_lag, &state->head, eps[t]);
    }

    /* Compute filtered pseudo-regressors psi_k(t) for each parameter
     * C(q) psi_k(t) = phi_k(t)
     * psi_k(t) = phi_k(t) - sum_{j=1}^{nc} c_j * psi_k(t-j)
     *
     * We compute psi for all parameters simultaneously at each time step. */
    double **psi = (double**)malloc((size_t)npar * sizeof(double*));
    for (int k = 0; k < npar; k++)
        psi[k] = (double*)calloc((size_t)N, sizeof(double));

    /* C coefficients */
    const double *c = theta + na + nb;

    for (int t = 0; t < N; t++) {
        for (int k = 0; k < npar; k++) {
            double phi_kt;

            if (k < na) {
                /* a_i: phi_kt = -y(t-i-1) */
                int i = k + 1;
                phi_kt = (t - i >= 0) ? -data->y[t - i] : 0.0;
            } else if (k < na + nb) {
                /* b_i: phi_kt = u(t-nk-i+1) */
                int i = k - na + 1;
                int idx = t - nk - i + 1;
                phi_kt = (idx >= 0) ? data->u[idx] : 0.0;
            } else {
                /* c_i: phi_kt = eps(t-i) */
                int i = k - na - nb + 1;
                phi_kt = (t - i >= 0) ? eps[t - i] : 0.0;
            }

            /* Filter through C(q): psi_kt = phi_kt - sum c_j * psi_k(t-j) */
            double psi_kt = phi_kt;
            for (int j = 1; j <= nc; j++) {
                if (t - j >= 0) psi_kt -= c[j - 1] * psi[k][t - j];
            }
            psi[k][t] = psi_kt;

            /* Accumulate gradient: g_k += eps(t) * psi_k(t) */
            g[k] += eps[t] * psi_kt;
        }
    }

    double invN = 1.0 / (double)N;
    for (int k = 0; k < npar; k++) {
        g[k] *= -invN;
        free(psi[k]);
    }
    free(psi);
    free(eps);
    pem_predictor_state_free(state);
}

void pem_gradient_oe(const double *theta, int nb, int nf, int nk,
                     const PEMData *data, double *g) {
    /* OE gradient via filtered pseudo-regressors.
     *
     * For OE: eps(t) = y(t) - w(t) where w(t) = B(q)/F(q) u(t-nk)
     *
     * Pseudo-regressors (filtered through 1/F(q)):
     *   For b_i: F(q) psi_{b_i}(t) = u(t-nk-i+1)
     *   For f_i: F(q) psi_{f_i}(t) = -w(t-i)
     *
     * g_k = -(1/N) sum eps(t) * psi_k(t)
     */
    int npar = nb + nf;
    int N = data->N;
    for (int i = 0; i < npar; i++) g[i] = 0.0;

    /* Compute all w(t) and eps(t) */
    int max_lag = nb + nk + 1;
    if (nf + 1 > max_lag) max_lag = nf + 1;
    if (max_lag < 10) max_lag = 10;

    PEMPredictorState *state = pem_predictor_state_alloc(max_lag);
    if (!state) return;

    double *w = (double*)malloc((size_t)N * sizeof(double));
    double *eps = (double*)malloc((size_t)N * sizeof(double));
    if (!w || !eps) { free(w); free(eps); pem_predictor_state_free(state); return; }

    for (int t = 0; t < N; t++) {
        w[t] = pem_predict_oe(theta, nb, nf, nk, data->u, t, state);
        eps[t] = data->y[t] - w[t];
        circ_store_local(state->w_history, max_lag, &state->head, w[t]);
    }

    /* Compute psi_k(t) for each parameter via filtering through F(q):
     * psi_k(t) = phi_k(t) - sum_{j=1}^{nf} f_j * psi_k(t-j) */
    double **psi = (double**)malloc((size_t)npar * sizeof(double*));
    for (int k = 0; k < npar; k++)
        psi[k] = (double*)calloc((size_t)N, sizeof(double));

    const double *f = theta + nb;

    for (int t = 0; t < N; t++) {
        for (int k = 0; k < npar; k++) {
            double phi_kt;
            if (k < nb) {
                /* b_i */
                int i = k + 1;
                int idx = t - nk - i + 1;
                phi_kt = (idx >= 0) ? data->u[idx] : 0.0;
            } else {
                /* f_i */
                int i = k - nb + 1;
                phi_kt = (t - i >= 0) ? -w[t - i] : 0.0;
            }

            /* Filter: psi_kt = phi_kt - sum f_j * psi_k(t-j) */
            double psi_kt = phi_kt;
            for (int j = 1; j <= nf; j++) {
                if (t - j >= 0) psi_kt -= f[j - 1] * psi[k][t - j];
            }
            psi[k][t] = psi_kt;

            g[k] += eps[t] * psi_kt;
        }
    }

    double invN = 1.0 / (double)N;
    for (int k = 0; k < npar; k++) {
        g[k] *= -invN;
        free(psi[k]);
    }
    free(psi);
    free(w); free(eps);
    pem_predictor_state_free(state);
}

/* ================================================================
 * Gauss-Newton Hessian Approximation
 *
 * H_GN = (1/N) * sum_t psi(t,theta) * psi^T(t,theta)
 *
 * This neglects the second-derivative term sum eps(t)*d^2(eps)/d(theta)^2,
 * which is justified because:
 *   1. Near the optimum, eps(t) -> white noise uncorrelated with d^2(eps)
 *   2. For ARX, d^2(eps)/d(theta)^2 = 0 exactly
 *   3. For other structures, the contribution is O(1/sqrt(N)) smaller
 *
 * The GN Hessian is always positive semidefinite, ensuring descent directions.
 *
 * Stored in row-major order: H[i*npar + j]
 * ================================================================ */

void pem_hessian_arx(const double *theta, int na, int nb, int nk,
                     const PEMData *data, double *H) {
    (void)theta; /* ARX Hessian is independent of theta (linear regression) */
    /* ARX Hessian: H = (1/N) * sum phi(t) * phi^T(t)
     * phi(t) = [-y(t-1),...,-y(t-na), u(t-nk),...,u(t-nk-nb+1)]
     *
     * This is the same as (1/N)*Phi^T*Phi, which is the normal equation matrix. */
    int npar = na + nb;
    int N = data->N;

    /* Zero Hessian */
    for (int i = 0; i < npar * npar; i++) H[i] = 0.0;

    for (int t = 0; t < N; t++) {
        /* Build phi(t) vector */
        double *phi = (double*)malloc((size_t)npar * sizeof(double));
        int idx = 0;
        for (int i = 1; i <= na; i++) {
            phi[idx++] = (t - i >= 0) ? -data->y[t - i] : 0.0;
        }
        for (int i = 1; i <= nb; i++) {
            int ui = t - nk - i + 1;
            phi[idx++] = (ui >= 0) ? data->u[ui] : 0.0;
        }

        /* Accumulate phi * phi^T */
        for (int i = 0; i < npar; i++)
            for (int j = 0; j < npar; j++)
                H[i * npar + j] += phi[i] * phi[j];

        free(phi);
    }

    double invN = 1.0 / (double)N;
    for (int i = 0; i < npar * npar; i++) H[i] *= invN;
}

void pem_hessian_oe(const double *theta, int nb, int nf, int nk,
                    const PEMData *data, double *H) {
    /* OE Gauss-Newton Hessian: H = (1/N) sum psi(t) * psi^T(t)
     * Uses the same psi computation as the gradient. */
    int npar = nb + nf;
    int N = data->N;
    for (int i = 0; i < npar * npar; i++) H[i] = 0.0;

    int max_lag = nb + nk + 1;
    if (nf + 1 > max_lag) max_lag = nf + 1;
    if (max_lag < 10) max_lag = 10;

    PEMPredictorState *state = pem_predictor_state_alloc(max_lag);
    if (!state) return;

    double *w = (double*)malloc((size_t)N * sizeof(double));
    if (!w) { pem_predictor_state_free(state); return; }

    for (int t = 0; t < N; t++) {
        w[t] = pem_predict_oe(theta, nb, nf, nk, data->u, t, state);
        circ_store_local(state->w_history, max_lag, &state->head, w[t]);
    }

    const double *f = theta + nb;
    double **psi = (double**)malloc((size_t)npar * sizeof(double*));
    for (int k = 0; k < npar; k++)
        psi[k] = (double*)calloc((size_t)N, sizeof(double));

    for (int t = 0; t < N; t++) {
        for (int k = 0; k < npar; k++) {
            double phi_kt;
            if (k < nb) {
                int i = k + 1;
                int idx = t - nk - i + 1;
                phi_kt = (idx >= 0) ? data->u[idx] : 0.0;
            } else {
                int i = k - nb + 1;
                phi_kt = (t - i >= 0) ? -w[t - i] : 0.0;
            }
            double psi_kt = phi_kt;
            for (int j = 1; j <= nf; j++) {
                if (t - j >= 0) psi_kt -= f[j - 1] * psi[k][t - j];
            }
            psi[k][t] = psi_kt;
        }
    }

    /* Accumulate H = (1/N) sum psi * psi^T */
    for (int t = 0; t < N; t++) {
        for (int i = 0; i < npar; i++) {
            for (int j = 0; j < npar; j++) {
                H[i * npar + j] += psi[i][t] * psi[j][t];
            }
        }
    }

    double invN = 1.0 / (double)N;
    for (int i = 0; i < npar * npar; i++) H[i] *= invN;

    for (int k = 0; k < npar; k++) free(psi[k]);
    free(psi); free(w);
    pem_predictor_state_free(state);
}

void pem_hessian_armax(const double *theta, int na, int nb, int nc, int nk,
                       const PEMData *data, double *H) {
    /* ARMAX Hessian: H = (1/N) sum psi(t) * psi^T(t)
     * Computed similarly to gradient but accumulating outer products. */
    int npar = na + nb + nc;
    int N = data->N;
    for (int i = 0; i < npar * npar; i++) H[i] = 0.0;

    /* Compute residuals first */
    int max_lag = na;
    if (nb + nk > max_lag) max_lag = nb + nk;
    if (nc > max_lag) max_lag = nc;
    max_lag += 2;
    if (max_lag < 10) max_lag = 10;

    PEMPredictorState *state = pem_predictor_state_alloc(max_lag);
    if (!state) return;

    double *eps = (double*)malloc((size_t)N * sizeof(double));
    if (!eps) { pem_predictor_state_free(state); return; }

    for (int t = 0; t < N; t++) {
        double yh = pem_predict_armax(theta, na, nb, nc, nk, data->u, data->y, t, state);
        eps[t] = data->y[t] - yh;
        circ_store_local(state->eps_history, max_lag, &state->head, eps[t]);
    }

    const double *c = theta + na + nb;
    double **psi = (double**)malloc((size_t)npar * sizeof(double*));
    for (int k = 0; k < npar; k++)
        psi[k] = (double*)calloc((size_t)N, sizeof(double));

    for (int t = 0; t < N; t++) {
        for (int k = 0; k < npar; k++) {
            double phi_kt;
            if (k < na) {
                int i = k + 1;
                phi_kt = (t - i >= 0) ? -data->y[t - i] : 0.0;
            } else if (k < na + nb) {
                int i = k - na + 1;
                int idx = t - nk - i + 1;
                phi_kt = (idx >= 0) ? data->u[idx] : 0.0;
            } else {
                int i = k - na - nb + 1;
                phi_kt = (t - i >= 0) ? eps[t - i] : 0.0;
            }
            double psi_kt = phi_kt;
            for (int j = 1; j <= nc; j++) {
                if (t - j >= 0) psi_kt -= c[j - 1] * psi[k][t - j];
            }
            psi[k][t] = psi_kt;
        }
    }

    for (int t = 0; t < N; t++) {
        for (int i = 0; i < npar; i++)
            for (int j = 0; j < npar; j++)
                H[i * npar + j] += psi[i][t] * psi[j][t];
    }

    double invN = 1.0 / (double)N;
    for (int i = 0; i < npar * npar; i++) H[i] *= invN;

    for (int k = 0; k < npar; k++) free(psi[k]);
    free(psi); free(eps);
    pem_predictor_state_free(state);
}

void pem_hessian_bj(const double *theta, int nb, int nc, int nd, int nf, int nk,
                    const PEMData *data, double *H) {
    /* BJ Hessian via numerical approximation.
     * For the full BJ model, exact psi computation is very involved.
     * We use the fact that near the optimum, the GN approximation with
     * numerically-derived psi is valid.
     *
     * For an initial implementation, we compute psi via finite differences
     * on eps(t,theta) with respect to each parameter. */
    int npar = nb + nc + nd + nf;
    int N = data->N;
    double h = 1e-6; /* finite difference step */

    for (int i = 0; i < npar * npar; i++) H[i] = 0.0;

    /* Compute baseline residuals */
    double *eps0 = (double*)malloc((size_t)N * sizeof(double));
    if (!eps0) return;
    pem_residuals_bj(theta, nb, nc, nd, nf, nk, data->u, data->y, N, eps0);

    /* Compute psi_k(t) = -(eps(t,theta+h*e_k) - eps(t,theta))/h */
    double **psi_mat = (double**)malloc((size_t)npar * sizeof(double*));
    for (int k = 0; k < npar; k++)
        psi_mat[k] = (double*)malloc((size_t)N * sizeof(double));

    double *theta_pert = (double*)malloc((size_t)npar * sizeof(double));
    memcpy(theta_pert, theta, (size_t)npar * sizeof(double));

    for (int k = 0; k < npar; k++) {
        theta_pert[k] = theta[k] + h;
        double *eps_pert = (double*)malloc((size_t)N * sizeof(double));
        pem_residuals_bj(theta_pert, nb, nc, nd, nf, nk, data->u, data->y, N, eps_pert);
        for (int t = 0; t < N; t++)
            psi_mat[k][t] = -(eps_pert[t] - eps0[t]) / h;
        theta_pert[k] = theta[k]; /* restore */
        free(eps_pert);
    }
    free(theta_pert);

    /* H = (1/N) sum psi(t) * psi^T(t) */
    for (int t = 0; t < N; t++) {
        for (int i = 0; i < npar; i++)
            for (int j = 0; j < npar; j++)
                H[i * npar + j] += psi_mat[i][t] * psi_mat[j][t];
    }

    double invN = 1.0 / (double)N;
    for (int i = 0; i < npar * npar; i++) H[i] *= invN;

    for (int k = 0; k < npar; k++) free(psi_mat[k]);
    free(psi_mat); free(eps0);
}

void pem_gradient_bj(const double *theta, int nb, int nc, int nd, int nf, int nk,
                     const PEMData *data, double *g) {
    /* BJ gradient via finite differences.
     * g_k = dV/d(theta_k) ~ (V(theta+h*e_k) - V(theta))/h */
    int npar = nb + nc + nd + nf;
    double h = 1e-6;

    double V0 = pem_criterion_bj(theta, nb, nc, nd, nf, nk, data);
    double *theta_pert = (double*)malloc((size_t)npar * sizeof(double));
    memcpy(theta_pert, theta, (size_t)npar * sizeof(double));

    for (int k = 0; k < npar; k++) {
        theta_pert[k] = theta[k] + h;
        double V_pert = pem_criterion_bj(theta_pert, nb, nc, nd, nf, nk, data);
        g[k] = (V_pert - V0) / h;
        theta_pert[k] = theta[k];
    }
    free(theta_pert);
}

/* ================================================================
 * Regularization
 * ================================================================ */

void pem_regularize_hessian(double *H, int npar, double lambda) {
    /* H <- H + lambda * I
     * Tikhonov regularization ensures positive definiteness.
     * This converts Gauss-Newton to Levenberg-Marquardt step. */
    for (int i = 0; i < npar; i++)
        H[i * npar + i] += lambda;
}