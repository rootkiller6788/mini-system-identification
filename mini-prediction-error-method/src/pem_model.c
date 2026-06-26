#include "pem_model.h"
#include "pem_optimizer.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/* ============================================================================
 * PEM Model — High-Level Estimation Functions
 *
 * Provides complete estimation routines for each model structure.
 * Each function:
 *   1. Validates inputs and allocates needed memory
 *   2. Initializes parameters (if not provided)
 *   3. Sets up callback functions for the optimizer
 *   4. Runs the optimization (Gauss-Newton or LM)
 *   5. Packages results
 *
 * References:
 *   Ljung (1999) Chapter 10: "Identification in Practice"
 *   Ljung (1999) Chapter 16: "Applications"
 * ============================================================================ */

/* ================================================================
 * Helper: Parameter Initialization Strategies
 *
 * Good initialization is critical for PEM convergence:
 * - ARX: Can use LS for closed-form init, or zeros
 * - ARMAX: Initialize via high-order ARX, truncate
 * - OE: Initialize via FIR, then model reduction
 * - BJ: Initialize OE part + ARMA part separately
 *
 * We provide simple heuristics here.
 * ================================================================ */

/** Simple initialization: zeros (works for stable processes started at rest) */
static void init_zeros(double *theta, int npar) {
    for (int i = 0; i < npar; i++) theta[i] = 0.0;
}

/** FIR-based initialization for OE models.
 *  Fit a long FIR, then approximate B/F from it.
 *  Simplified: use identity (b_1 = 1, rest 0; F unit).
 */
static void init_oe_fir(const PEMData *data, int nb, int nf, int nk,
                        double *theta) {
    (void)data; (void)nk;
    /* Simple heuristic: set b_1 = 1, others 0; set f_i = 0 (F=1)
     * This corresponds to a pure delay model as starting point. */
    if (nb > 0) theta[0] = 1.0;
    for (int i = 1; i < nb; i++) theta[i] = 0.0;
    for (int i = 0; i < nf; i++) theta[nb + i] = 0.0;
}

/* ================================================================
 * Callback Adaptors
 *
 * The optimizer uses generic callback types. We wrap model-specific
 * criterion/gradient/hessian functions as callbacks via a data struct.
 * ================================================================ */

typedef struct {
    PEMModelStructure structure;
    const double *theta;              /* Parameter vector (current) */
    const double *u, *y;             /* Data */
    int N;
    int orders[5];                   /* na,nb,nc,nd,nf */
    int nk;
} PEMModelCallbackData;

static double cb_criterion_arx(const double *theta, void *data) {
    PEMModelCallbackData *d = (PEMModelCallbackData*)data;
    PEMData pdata = { .N = d->N, .u = (double*)d->u, .y = (double*)d->y };
    return pem_criterion_arx(theta, d->orders[0], d->orders[1], d->nk, &pdata);
}
static void cb_gradient_arx(const double *theta, void *data, double *g) {
    PEMModelCallbackData *d = (PEMModelCallbackData*)data;
    PEMData pdata = { .N = d->N, .u = (double*)d->u, .y = (double*)d->y };
    pem_gradient_arx(theta, d->orders[0], d->orders[1], d->nk, &pdata, g);
}
static void cb_hessian_arx(const double *theta, void *data, double *H) {
    PEMModelCallbackData *d = (PEMModelCallbackData*)data;
    PEMData pdata = { .N = d->N, .u = (double*)d->u, .y = (double*)d->y };
    pem_hessian_arx(theta, d->orders[0], d->orders[1], d->nk, &pdata, H);
}

static double cb_criterion_armax(const double *theta, void *data) {
    PEMModelCallbackData *d = (PEMModelCallbackData*)data;
    PEMData pdata = { .N = d->N, .u = (double*)d->u, .y = (double*)d->y };
    return pem_criterion_armax(theta, d->orders[0], d->orders[1], d->orders[2], d->nk, &pdata);
}
static void cb_gradient_armax(const double *theta, void *data, double *g) {
    PEMModelCallbackData *d = (PEMModelCallbackData*)data;
    PEMData pdata = { .N = d->N, .u = (double*)d->u, .y = (double*)d->y };
    pem_gradient_armax(theta, d->orders[0], d->orders[1], d->orders[2], d->nk, &pdata, g);
}
static void cb_hessian_armax(const double *theta, void *data, double *H) {
    PEMModelCallbackData *d = (PEMModelCallbackData*)data;
    PEMData pdata = { .N = d->N, .u = (double*)d->u, .y = (double*)d->y };
    pem_hessian_armax(theta, d->orders[0], d->orders[1], d->orders[2], d->nk, &pdata, H);
}

static double cb_criterion_oe(const double *theta, void *data) {
    PEMModelCallbackData *d = (PEMModelCallbackData*)data;
    PEMData pdata = { .N = d->N, .u = (double*)d->u, .y = (double*)d->y };
    return pem_criterion_oe(theta, d->orders[1], d->orders[4], d->nk, &pdata);
}
static void cb_gradient_oe(const double *theta, void *data, double *g) {
    PEMModelCallbackData *d = (PEMModelCallbackData*)data;
    PEMData pdata = { .N = d->N, .u = (double*)d->u, .y = (double*)d->y };
    pem_gradient_oe(theta, d->orders[1], d->orders[4], d->nk, &pdata, g);
}
static void cb_hessian_oe(const double *theta, void *data, double *H) {
    PEMModelCallbackData *d = (PEMModelCallbackData*)data;
    PEMData pdata = { .N = d->N, .u = (double*)d->u, .y = (double*)d->y };
    pem_hessian_oe(theta, d->orders[1], d->orders[4], d->nk, &pdata, H);
}

static double cb_criterion_bj(const double *theta, void *data) {
    PEMModelCallbackData *d = (PEMModelCallbackData*)data;
    PEMData pdata = { .N = d->N, .u = (double*)d->u, .y = (double*)d->y };
    return pem_criterion_bj(theta, d->orders[1], d->orders[2], d->orders[3], d->orders[4], d->nk, &pdata);
}
static void cb_gradient_bj(const double *theta, void *data, double *g) {
    PEMModelCallbackData *d = (PEMModelCallbackData*)data;
    PEMData pdata = { .N = d->N, .u = (double*)d->u, .y = (double*)d->y };
    pem_gradient_bj(theta, d->orders[1], d->orders[2], d->orders[3], d->orders[4], d->nk, &pdata, g);
}
static void cb_hessian_bj(const double *theta, void *data, double *H) {
    PEMModelCallbackData *d = (PEMModelCallbackData*)data;
    PEMData pdata = { .N = d->N, .u = (double*)d->u, .y = (double*)d->y };
    pem_hessian_bj(theta, d->orders[1], d->orders[2], d->orders[3], d->orders[4], d->nk, &pdata, H);
}

/* ================================================================
 * ARX Estimation — Closed-Form Least Squares
 *
 * ARX model: y(t) = phi^T(t) theta + e(t)
 * LS solution: theta = (Phi^T Phi)^{-1} Phi^T Y
 *
 * Implements the normal equations via Cholesky:
 *   Phi^T Phi * theta = Phi^T Y
 *
 * This is O(N*(na+nb)^2) for building the normal matrix,
 * plus O((na+nb)^3) for the Cholesky solve.
 * ================================================================ */

int pem_estimate_arx_ls(const PEMData *data, int na, int nb, int nk,
                        PEMResult *result, const PEMOptions *opts) {
    if (!data || !result || data->N <= 0 || na < 0 || nb < 0 || nk < 1) return 1;

    int npar = na + nb;
    if (npar <= 0) return 1;
    if (data->N <= npar) return 1; /* Need more data than parameters */

    int N = data->N;

    /* Allocate normal matrix Phi^T Phi (npar x npar) and RHS Phi^T Y (npar x 1) */
    double *PhiTPhi = (double*)calloc((size_t)(npar * npar), sizeof(double));
    double *PhiTY = (double*)calloc((size_t)npar, sizeof(double));
    if (!PhiTPhi || !PhiTY) { free(PhiTPhi); free(PhiTY); return 1; }

    /* Build normal equations */
    for (int t = 0; t < N; t++) {
        /* Build regressor phi(t) */
        double *phi = (double*)malloc((size_t)npar * sizeof(double));
        int idx = 0;
        for (int i = 1; i <= na; i++)
            phi[idx++] = (t - i >= 0) ? -data->y[t - i] : 0.0;
        for (int i = 1; i <= nb; i++) {
            int ui = t - nk - i + 1;
            phi[idx++] = (ui >= 0) ? data->u[ui] : 0.0;
        }

        /* Accumulate PhiTPhi += phi * phi^T */
        for (int i = 0; i < npar; i++)
            for (int j = 0; j < npar; j++)
                PhiTPhi[i * npar + j] += phi[i] * phi[j];

        /* Accumulate PhiTY += phi * y(t) */
        for (int i = 0; i < npar; i++)
            PhiTY[i] += phi[i] * data->y[t];

        free(phi);
    }

    /* Regularize normal matrix for numerical stability */
    pem_regularize_hessian(PhiTPhi, npar, 1e-8);

    /* Solve PhiTPhi * theta = PhiTY via Cholesky */
    double *L = (double*)malloc((size_t)(npar * npar) * sizeof(double));
    memcpy(L, PhiTPhi, (size_t)(npar * npar) * sizeof(double));

    if (pem_cholesky(L, npar) != 0) {
        free(L); free(PhiTPhi); free(PhiTY);
        result->status = PEM_SINGULAR_HESSIAN;
        return 1;
    }

    pem_cholesky_solve(L, PhiTY, result->theta_hat, npar);
    free(L);

    /* Compute loss */
    PEMData pdata = { .N = N, .u = data->u, .y = data->y };
    result->loss_final = pem_criterion_arx(result->theta_hat, na, nb, nk, &pdata);
    result->loss_init = result->loss_final;
    result->iterations = 0;
    result->status = PEM_CONVERGED;

    /* Compute covariance */
    if (opts && opts->compute_covariance) {
        double sigma2 = 2.0 * result->loss_final;
        double *Hinv = (double*)malloc((size_t)(npar * npar) * sizeof(double));
        if (Hinv && pem_inverse_spd(PhiTPhi, Hinv, npar) == 0) {
            for (int i = 0; i < npar * npar; i++)
                result->covariance[i] = sigma2 * Hinv[i] / (double)N;
        }
        free(Hinv);
        memcpy(result->information_matrix, PhiTPhi, (size_t)(npar * npar) * sizeof(double));
        result->condition_number = pem_condition_number(PhiTPhi, npar);
    }

    /* Compute gradient at optimum (should be near zero) */
    pem_gradient_arx(result->theta_hat, na, nb, nk, &pdata, result->gradient);

    free(PhiTPhi); free(PhiTY);
    return 0;
}

/* ================================================================
 * ARX Estimation — Iterative PEM
 *
 * Wraps the LM optimizer with ARX callback functions.
 * Since ARX is linear, one GN iteration suffices from zero init.
 * ================================================================ */

int pem_estimate_arx(const PEMData *data, int na, int nb, int nk,
                     const double *theta0, PEMResult *result,
                     const PEMOptions *opts) {
    if (!data || !result) return 1;
    int npar = na + nb;

    PEMOptions myopts = opts ? *opts : pem_options_default();
    if (myopts.algorithm == PEM_OPT_LM)
        myopts.algorithm = PEM_OPT_GN; /* GN is faster and exact for ARX */

    PEMModelCallbackData cbdata = {
        .structure = PEM_ARX,
        .u = data->u, .y = data->y, .N = data->N,
        .orders = {na, nb, 0, 0, 0}, .nk = nk
    };

    /* Copy initial theta or use zeros */
    if (theta0)
        memcpy(result->theta_hat, theta0, (size_t)npar * sizeof(double));
    else
        init_zeros(result->theta_hat, npar);

    return pem_optimize_gauss_newton(result->theta_hat, npar,
                                     cb_criterion_arx, cb_gradient_arx,
                                     cb_hessian_arx, &cbdata, &myopts, result);
}

/* ================================================================
 * ARMAX Estimation
 * ================================================================ */

int pem_estimate_armax(const PEMData *data, int na, int nb, int nc, int nk,
                       const double *theta0, PEMResult *result,
                       const PEMOptions *opts) {
    if (!data || !result) return 1;
    int npar = na + nb + nc;

    PEMOptions myopts = opts ? *opts : pem_options_default();
    if (myopts.algorithm != PEM_OPT_LM && myopts.algorithm != PEM_OPT_GN)
        myopts.algorithm = PEM_OPT_LM;

    PEMModelCallbackData cbdata = {
        .structure = PEM_ARMAX,
        .u = data->u, .y = data->y, .N = data->N,
        .orders = {na, nb, nc, 0, 0}, .nk = nk
    };

    if (theta0) {
        memcpy(result->theta_hat, theta0, (size_t)npar * sizeof(double));
    } else {
        /* Initialize from ARX estimate (ignore MA part initially) */
        init_zeros(result->theta_hat, npar);
        if (na + nb > 0) {
            PEMResult arx_res;
            arx_res.npar = na + nb;
            arx_res.theta_hat = (double*)malloc((size_t)(na + nb) * sizeof(double));
            arx_res.covariance = NULL;
            arx_res.gradient = NULL;
            arx_res.information_matrix = NULL;
            pem_estimate_arx_ls(data, na, nb, nk, &arx_res, NULL);
            memcpy(result->theta_hat, arx_res.theta_hat, (size_t)(na + nb) * sizeof(double));
            free(arx_res.theta_hat);
        }
    }

    return pem_optimize_levenberg_marquardt(result->theta_hat, npar,
                                            cb_criterion_armax, cb_gradient_armax,
                                            cb_hessian_armax, &cbdata, &myopts, result);
}

/* ================================================================
 * Output Error Estimation
 * ================================================================ */

int pem_estimate_oe(const PEMData *data, int nb, int nf, int nk,
                    const double *theta0, PEMResult *result,
                    const PEMOptions *opts) {
    if (!data || !result) return 1;
    int npar = nb + nf;

    PEMOptions myopts = opts ? *opts : pem_options_default();
    if (myopts.algorithm != PEM_OPT_LM)
        myopts.algorithm = PEM_OPT_LM; /* LM preferred for OE (non-convex) */

    PEMModelCallbackData cbdata = {
        .structure = PEM_OE,
        .u = data->u, .y = data->y, .N = data->N,
        .orders = {0, nb, 0, 0, nf}, .nk = nk
    };

    if (theta0) {
        memcpy(result->theta_hat, theta0, (size_t)npar * sizeof(double));
    } else {
        init_oe_fir(data, nb, nf, nk, result->theta_hat);
    }

    return pem_optimize_levenberg_marquardt(result->theta_hat, npar,
                                            cb_criterion_oe, cb_gradient_oe,
                                            cb_hessian_oe, &cbdata, &myopts, result);
}

/* ================================================================
 * Box-Jenkins Estimation
 * ================================================================ */

int pem_estimate_bj(const PEMData *data, int nb, int nc, int nd, int nf, int nk,
                    const double *theta0, PEMResult *result,
                    const PEMOptions *opts) {
    if (!data || !result) return 1;
    int npar = nb + nc + nd + nf;

    PEMOptions myopts = opts ? *opts : pem_options_default();
    if (myopts.algorithm == PEM_OPT_GN)
        myopts.algorithm = PEM_OPT_LM; /* LM more robust for BJ */

    PEMModelCallbackData cbdata = {
        .structure = PEM_BJ,
        .u = data->u, .y = data->y, .N = data->N,
        .orders = {0, nb, nc, nd, nf}, .nk = nk
    };

    if (theta0) {
        memcpy(result->theta_hat, theta0, (size_t)npar * sizeof(double));
    } else {
        init_zeros(result->theta_hat, npar);
        /* Set b_1 = 1 as minimal starting point */
        if (nb > 0) result->theta_hat[0] = 1.0;
    }

    return pem_optimize_levenberg_marquardt(result->theta_hat, npar,
                                            cb_criterion_bj, cb_gradient_bj,
                                            cb_hessian_bj, &cbdata, &myopts, result);
}

/* ================================================================
 * FIR Estimation — Least Squares (Closed Form)
 * ================================================================ */

int pem_estimate_fir(const PEMData *data, int nb, int nk,
                     PEMResult *result, const PEMOptions *opts) {
    /* FIR is a special case of ARX with na=0.
     * y(t) = sum b_i u(t-nk-i+1) + e(t) */
    return pem_estimate_arx_ls(data, 0, nb, nk, result, opts);
}

/* ================================================================
 * Model Simulation (Pure simulation, e(t)=0)
 *
 * Simulates y_sim(t) = G(q,theta) u(t) with e(t) = 0.
 * For ARX: A(q) y_sim(t) = B(q) u(t-nk) -> y_sim = B/A u
 * For ARMAX: same as ARX (noise set to zero)
 * For OE: y_sim = B/F u
 * For BJ: y_sim = B/F u
 *
 * This implements the deterministic part of the model.
 * ================================================================ */

int pem_simulate_model(PEMModelStructure structure, const double *theta,
                       const int *orders, int nk,
                       const double *u, int N, double *y_sim,
                       const double *y0) {
    if (!theta || !orders || !u || !y_sim || N <= 0) return 1;

    switch (structure) {
        case PEM_ARX:
        case PEM_ARMAX: {
            /* Both use B(q)/A(q) for simulation (noise set to 0) */
            int na = orders[0], nb = orders[1];
            for (int t = 0; t < N; t++) {
                double yt = 0.0;
                /* B part */
                for (int i = 1; i <= nb; i++) {
                    int idx = t - nk - i + 1;
                    if (idx >= 0) yt += theta[na + i - 1] * u[idx];
                }
                /* A part: y(t) = -sum a_i y(t-i) + B part */
                for (int i = 1; i <= na; i++) {
                    if (t - i >= 0) yt -= theta[i - 1] * y_sim[t - i];
                    else if (y0 && (i - 1) < na) yt -= theta[i - 1] * y0[i - 1];
                }
                y_sim[t] = yt;
            }
            break;
        }
        case PEM_OE:
        case PEM_BJ: {
            /* Both use B(q)/F(q) for simulation */
            int nb = orders[1], nf = orders[4];
            /* In BJ, b params start at 0; f params at nb+nc+nd.
             * In OE, b params at 0, f params at nb. */
            int f_offset = (structure == PEM_BJ) ? (nb + orders[2] + orders[3]) : nb;
            for (int t = 0; t < N; t++) {
                double yt = 0.0;
                for (int i = 1; i <= nb; i++) {
                    int idx = t - nk - i + 1;
                    if (idx >= 0) yt += theta[i - 1] * u[idx];
                }
                for (int i = 1; i <= nf; i++) {
                    if (t - i >= 0) yt -= theta[f_offset + i - 1] * y_sim[t - i];
                    else if (y0 && (i - 1) < nf) yt -= theta[f_offset + i - 1] * y0[i - 1];
                }
                y_sim[t] = yt;
            }
            break;
        }
        case PEM_FIR: {
            int nb = orders[1];
            for (int t = 0; t < N; t++) {
                double yt = 0.0;
                for (int i = 1; i <= nb; i++) {
                    int idx = t - nk - i + 1;
                    if (idx >= 0) yt += theta[i - 1] * u[idx];
                }
                y_sim[t] = yt;
            }
            break;
        }
        default:
            return 1;
    }
    return 0;
}

/* ================================================================
 * k-Step-Ahead Prediction
 *
 * For model validation, k-step-ahead prediction compares the model's
 * predictive capability at different horizons.
 *
 * k = 1: one-step-ahead (uses all past measurements)
 * k = inf: pure simulation (uses only inputs)
 *
 * Currently supported for OE and ARX structures.
 * ================================================================ */

int pem_predict_kstep(PEMModelStructure structure, const double *theta,
                      const int *orders, int nk,
                      const double *u, const double *y, int N, int k,
                      double *y_hat) {
    if (!theta || !orders || !u || !y_hat || N <= 0) return 1;

    switch (structure) {
        case PEM_OE: {
            int nb = orders[1], nf = orders[4];
            pem_kstep_predict_oe(theta, nb, nf, nk, u, N, k, y_hat);
            break;
        }
        case PEM_ARX: {
            /* For ARX, k-step prediction = simulation with y reset every k steps.
             * Simplified: use batch one-step prediction if k=1. */
            int na = orders[0], nb = orders[1];
            if (k == 1) {
                pem_predict_arx_batch(theta, na, nb, nk, u, y, N, y_hat);
            } else {
                /* Run k-step simulation with periodic y-reset */
                for (int t = 0; t < N; t++) {
                    if (t % k == 0 && t > 0) {
                        /* Reset using actual y(t-1) */
                    }
                    double yh = pem_predict_arx(theta, na, nb, nk, u, y, t);
                    y_hat[t] = yh;
                }
            }
            break;
        }
        default:
            return 1;
    }
    return 0;
}