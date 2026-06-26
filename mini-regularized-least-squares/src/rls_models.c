#include "rls_core.h"
#include "rls_models.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================================
 * Regressor Construction (L3: Mathematical Structures)
 * ============================================================================ */

void rls_build_fir_regressor(RLSMatrix *Phi, RLSVector *y_vec,
                              const RLSData *data, int nb) {
    if (!Phi || !y_vec || !data) return;
    int n_eff = data->N - nb;
    if (n_eff <= 0) return;
    for (int i = 0; i < n_eff; i++) {
        int t = i + nb;
        y_vec->data[i] = data->y[t];
        for (int j = 0; j < nb; j++)
            Phi->data[j * n_eff + i] = data->u[t - 1 - j];
    }
}

void rls_build_arx_regressor(RLSMatrix *Phi, RLSVector *y_vec,
                              const RLSData *data, const RLSModelOrder *order) {
    if (!Phi || !y_vec || !data || !order) return;
    int na = order->na, nb = order->nb, nk = order->nk;
    int max_delay = (na > (nb + nk - 1)) ? na : (nb + nk - 1);
    int n_eff = data->N - max_delay;
    if (n_eff <= 0) return;
    for (int i = 0; i < n_eff; i++) {
        int t = i + max_delay;
        y_vec->data[i] = data->y[t];
        /* AR part: -y(t-1), ..., -y(t-na) */
        for (int j = 0; j < na; j++)
            Phi->data[j * n_eff + i] = -data->y[t - 1 - j];
        /* X part: u(t-nk), ..., u(t-nk-nb+1) */
        for (int j = 0; j < nb; j++)
            Phi->data[(na + j) * n_eff + i] = data->u[t - nk - j];
    }
}

void rls_build_oe_regressor(RLSMatrix *Phi, RLSVector *y_vec,
                             const RLSData *data, const RLSModelOrder *order,
                             const RLSVector *theta_init) {
    if (!Phi || !y_vec || !data || !order) return;
    int nb = order->nb, nf = order->nf, nk = order->nk;
    int np = nb + nf;
    int max_delay = (nf > (nb + nk - 1)) ? nf : (nb + nk - 1);
    int n_eff = data->N - max_delay;
    if (n_eff <= 0) return;
    /* Simulate noise-free output for pseudo-linear regression.
       Simplified single-iteration: use past simulated outputs as regressors.
       For full OE, this is iterated (several passes). */
    RLSVector *w_sim = rls_vector_alloc(data->N);
    /* Initialize simulated output with measured output */
    for (int t = 0; t < data->N; t++) w_sim->data[t] = data->y[t];
    /* If theta_init provided, simulate once to refine regressor */
    if (theta_init) {
        for (int t = max_delay; t < data->N; t++) {
            double yh = 0.0;
            /* B part */
            for (int j = 0; j < nb; j++)
                yh += theta_init->data[j] * data->u[t - nk - j];
            /* F part (recursive) */
            for (int j = 0; j < nf; j++)
                yh -= theta_init->data[nb + j] * w_sim->data[t - 1 - j];
            w_sim->data[t] = yh;
        }
    }
    /* Build regressor from simulated output */
    for (int i = 0; i < n_eff; i++) {
        int t = i + max_delay;
        y_vec->data[i] = data->y[t];
        for (int j = 0; j < nb; j++)
            Phi->data[j * n_eff + i] = data->u[t - nk - j];
        for (int j = 0; j < nf; j++)
            Phi->data[(nb + j) * n_eff + i] = -w_sim->data[t - 1 - j];
    }
    rls_vector_free(w_sim);
}

void rls_build_armax_regressor(RLSMatrix *Phi, RLSVector *y_vec,
                                const RLSData *data, const RLSModelOrder *order,
                                const RLSVector *theta_init) {
    if (!Phi || !y_vec || !data || !order) return;
    int na = order->na, nb = order->nb, nc = order->nc, nk = order->nk;
    int max_delay = (na > (nb + nk - 1)) ? na : (nb + nk - 1);
    if (nc > max_delay) max_delay = nc;
    int n_eff = data->N - max_delay;
    if (n_eff <= 0) return;
    int np = na + nb + nc;
    /* Compute residuals using initial parameters */
    RLSVector *resid = rls_vector_alloc(data->N);
    if (theta_init) {
        for (int t = max_delay; t < data->N; t++) {
            double yh = 0.0;
            for (int j = 0; j < na; j++) yh -= theta_init->data[j] * data->y[t-1-j];
            for (int j = 0; j < nb; j++) yh += theta_init->data[na+j] * data->u[t-nk-j];
            for (int j = 0; j < nc; j++) yh += theta_init->data[na+nb+j] * resid->data[t-1-j];
            resid->data[t] = data->y[t] - yh;
        }
    }
    for (int i = 0; i < n_eff; i++) {
        int t = i + max_delay;
        y_vec->data[i] = data->y[t];
        for (int j = 0; j < na; j++)
            Phi->data[j * n_eff + i] = -data->y[t-1-j];
        for (int j = 0; j < nb; j++)
            Phi->data[(na+j) * n_eff + i] = data->u[t-nk-j];
        for (int j = 0; j < nc; j++)
            Phi->data[(na+nb+j) * n_eff + i] = resid->data[t-1-j];
    }
    rls_vector_free(resid);
}

void rls_build_ss_regressor(RLSMatrix *Phi, RLSVector *y_vec,
                             const RLSData *data, const RLSModelOrder *order) {
    if (!Phi || !y_vec || !data || !order) return;
    int nx = order->nx;
    int horizon = 10; /* past horizon for subspace identification */
    int n_eff = data->N - horizon;
    if (n_eff <= 0) return;
    /* Simplified: use stacked I/O as regressors */
    for (int i = 0; i < n_eff; i++) {
        int t = i + horizon;
        y_vec->data[i] = data->y[t];
        for (int j = 0; j < horizon; j++) {
            Phi->data[(2*j) * n_eff + i] = data->y[t-horizon+j];
            Phi->data[(2*j+1) * n_eff + i] = data->u[t-horizon+j];
        }
    }
}

void rls_build_narx_regressor(RLSMatrix *Phi, RLSVector *y_vec,
                               const RLSData *data, const RLSModelOrder *order,
                               int poly_degree) {
    if (!Phi || !y_vec || !data || !order) return;
    int na = order->na, nb = order->nb, nk = order->nk;
    int max_delay = (na > (nb+nk-1)) ? na : (nb+nk-1);
    int n_eff = data->N - max_delay;
    if (n_eff <= 0) return;
    int n_linear = na + nb;
    /* Build linear terms first, then add monomials up to poly_degree */
    RLSVector *lin = rls_vector_alloc(n_linear);
    for (int i = 0; i < n_eff; i++) {
        int t = i + max_delay;
        y_vec->data[i] = data->y[t];
        /* Fill linear terms */
        for (int j = 0; j < na; j++) lin->data[j] = -data->y[t-1-j];
        for (int j = 0; j < nb; j++) lin->data[na+j] = data->u[t-nk-j];
        /* Fill regressor columns */
        int col = 0;
        for (int j = 0; j < n_linear; j++) {
            Phi->data[col * n_eff + i] = lin->data[j]; col++;
        }
        /* Quadratic terms */
        if (poly_degree >= 2) {
            for (int j = 0; j < n_linear; j++)
                for (int k = j; k < n_linear; k++) {
                    Phi->data[col * n_eff + i] = lin->data[j] * lin->data[k]; col++;
                }
        }
        /* Cubic terms (selective to avoid explosion) */
        if (poly_degree >= 3) {
            for (int j = 0; j < n_linear; j++) {
                Phi->data[col * n_eff + i] = lin->data[j] * lin->data[j] * lin->data[j];
                col++;
            }
        }
    }
    rls_vector_free(lin);
}

/* ============================================================================
 * Model Simulation (L5: Methods)
 * ============================================================================ */

void rls_simulate_fir(RLSVector *y_sim, const RLSData *data,
                       const RLSVector *theta, int nb) {
    if (!y_sim || !data || !theta) return;
    for (int t = 0; t < data->N; t++) {
        if (t < nb) { y_sim->data[t] = 0.0; continue; }
        double yh = 0.0;
        for (int j = 0; j < nb; j++)
            yh += theta->data[j] * data->u[t - 1 - j];
        y_sim->data[t] = yh;
    }
}

void rls_simulate_arx(RLSVector *y_sim, const RLSData *data,
                       const RLSVector *theta, const RLSModelOrder *order) {
    if (!y_sim || !data || !theta || !order) return;
    int na = order->na, nb = order->nb, nk = order->nk;
    int max_delay = (na > (nb+nk-1)) ? na : (nb+nk-1);
    for (int t = 0; t < data->N; t++) {
        if (t < max_delay) { y_sim->data[t] = data->y[t]; continue; }
        double yh = 0.0;
        for (int j = 0; j < na; j++)
            yh -= theta->data[j] * y_sim->data[t-1-j];
        for (int j = 0; j < nb; j++)
            yh += theta->data[na+j] * data->u[t-nk-j];
        y_sim->data[t] = yh;
    }
}

void rls_simulate_oe(RLSVector *y_sim, const RLSData *data,
                      const RLSVector *theta, const RLSModelOrder *order) {
    if (!y_sim || !data || !theta || !order) return;
    int nb = order->nb, nf = order->nf, nk = order->nk;
    int max_delay = (nf > (nb+nk-1)) ? nf : (nb+nk-1);
    for (int t = 0; t < data->N; t++) {
        if (t < max_delay) { y_sim->data[t] = 0.0; continue; }
        double yh = 0.0;
        for (int j = 0; j < nb; j++)
            yh += theta->data[j] * data->u[t-nk-j];
        for (int j = 0; j < nf; j++)
            yh -= theta->data[nb+j] * y_sim->data[t-1-j];
        y_sim->data[t] = yh;
    }
}

void rls_simulate_ss(RLSVector *y_sim, const RLSData *data,
                      const RLSVector *theta, const RLSModelOrder *order) {
    if (!y_sim || !data || !theta || !order) return;
    int nx = order->nx;
    double *x = (double *)calloc(nx, sizeof(double));
    /* theta packs A (nx*nx), B (nx), C (nx) row-major */
    for (int t = 0; t < data->N; t++) {
        /* Output: y = C * x */
        double yh = 0.0;
        for (int j = 0; j < nx; j++)
            yh += theta->data[nx*nx + nx + j] * x[j];
        y_sim->data[t] = yh;
        /* State update: x_next = A * x + B * u */
        double *x_next = (double *)calloc(nx, sizeof(double));
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < nx; j++)
                x_next[i] += theta->data[j*nx + i] * x[j];
            x_next[i] += theta->data[nx*nx + i] * data->u[t];
        }
        memcpy(x, x_next, nx * sizeof(double));
        free(x_next);
    }
    free(x);
}

/* ============================================================================
 * Model Validation Metrics (L6: Canonical Problems)
 * ============================================================================ */

double rls_fit_percent(const RLSVector *y, const RLSVector *y_hat) {
    if (!y || !y_hat) return -INFINITY;
    int n = y->dim;
    double y_mean = 0.0;
    for (int i = 0; i < n; i++) y_mean += y->data[i];
    y_mean /= n;
    double num = 0.0, den = 0.0;
    for (int i = 0; i < n; i++) {
        double d1 = y->data[i] - y_hat->data[i];
        double d2 = y->data[i] - y_mean;
        num += d1 * d1;
        den += d2 * d2;
    }
    if (den < 1e-15) return 100.0;
    return 100.0 * (1.0 - sqrt(num / den));
}

double rls_mse(const RLSVector *y, const RLSVector *y_hat) {
    if (!y || !y_hat) return INFINITY;
    double s = 0.0;
    for (int i = 0; i < y->dim; i++) {
        double d = y->data[i] - y_hat->data[i];
        s += d * d;
    }
    return s / y->dim;
}

double rls_whiteness_test(const RLSVector *residuals, int max_lag) {
    if (!residuals || max_lag < 1) return 0.0;
    int n = residuals->dim;
    if (n <= max_lag) return 1.0;
    /* Compute autocorrelation and Ljung-Box Q statistic */
    double mean_r = 0.0;
    for (int i = 0; i < n; i++) mean_r += residuals->data[i];
    mean_r /= n;
    double var_r = 0.0;
    for (int i = 0; i < n; i++) {
        double d = residuals->data[i] - mean_r;
        var_r += d * d;
    }
    if (var_r < 1e-15) return 1.0;
    double Q = 0.0;
    for (int lag = 1; lag <= max_lag; lag++) {
        double acf = 0.0;
        for (int i = lag; i < n; i++)
            acf += (residuals->data[i] - mean_r) * (residuals->data[i-lag] - mean_r);
        acf /= var_r;
        Q += acf * acf / (n - lag);
    }
    Q *= n * (n + 2);
    /* Chi-squared approximation: Q ~ chi^2(max_lag).
       Return approximate p-value using normal approximation for large df. */
    double df = (double)max_lag;
    double z = (Q - df) / sqrt(2.0 * df);
    /* p-value from normal tail (very approximate) */
    double p = 0.5 * erfc(z / sqrt(2.0));
    return p;
}

double rls_independence_test(const RLSVector *residuals, const RLSVector *u,
                              int max_lag) {
    if (!residuals || !u || max_lag < 1) return 0.0;
    int n = residuals->dim;
    if (n <= max_lag) return 1.0;
    double mean_r = 0.0, mean_u = 0.0;
    for (int i = 0; i < n; i++) { mean_r += residuals->data[i]; mean_u += u->data[i]; }
    mean_r /= n; mean_u /= n;
    double var_r = 0.0, var_u = 0.0;
    for (int i = 0; i < n; i++) {
        var_r += (residuals->data[i]-mean_r)*(residuals->data[i]-mean_r);
        var_u += (u->data[i]-mean_u)*(u->data[i]-mean_u);
    }
    if (var_r < 1e-15 || var_u < 1e-15) return 1.0;
    double Q = 0.0;
    for (int lag = 1; lag <= max_lag; lag++) {
        double xcf = 0.0;
        for (int i = lag; i < n; i++)
            xcf += (residuals->data[i]-mean_r) * (u->data[i-lag]-mean_u);
        xcf /= sqrt(var_r * var_u);
        Q += xcf * xcf / (n - lag);
    }
    Q *= n * (n + 2);
    double df = (double)max_lag;
    double z = (Q - df) / sqrt(2.0 * df);
    return 0.5 * erfc(z / sqrt(2.0));
}

/* ============================================================================
 * Model Utilities
 * ============================================================================ */

int rls_model_num_params(const RLSModelOrder *order) {
    if (!order) return 0;
    switch (order->type) {
        case RLS_MODEL_FIR:   return order->nb;
        case RLS_MODEL_ARX:   return order->na + order->nb;
        case RLS_MODEL_OE:    return order->nb + order->nf;
        case RLS_MODEL_ARMAX: return order->na + order->nb + order->nc;
        case RLS_MODEL_BJ:    return order->nb + order->nc + order->nd + order->nf;
        case RLS_MODEL_SS:    return order->nx * order->nx + 2 * order->nx;
        case RLS_MODEL_NARX:  return order->na + order->nb; /* base, actual depends on poly_degree */
        default: return 0;
    }
}

int rls_model_effective_samples(const RLSData *data, const RLSModelOrder *order) {
    if (!data || !order) return 0;
    int max_delay = 0;
    switch (order->type) {
        case RLS_MODEL_FIR:   max_delay = order->nb; break;
        case RLS_MODEL_ARX:   max_delay = (order->na > order->nb+order->nk-1) ? order->na : order->nb+order->nk-1; break;
        case RLS_MODEL_OE:    max_delay = (order->nf > order->nb+order->nk-1) ? order->nf : order->nb+order->nk-1; break;
        case RLS_MODEL_ARMAX: max_delay = order->na;
                               if (order->nc > max_delay) max_delay = order->nc;
                               if (order->nb+order->nk-1 > max_delay) max_delay = order->nb+order->nk-1; break;
        case RLS_MODEL_BJ:    max_delay = (order->nf > order->nd) ? order->nf : order->nd;
                               if (order->nb+order->nk-1 > max_delay) max_delay = order->nb+order->nk-1; break;
        default: max_delay = 10; break;
    }
    int n_eff = data->N - max_delay;
    return (n_eff > 0) ? n_eff : 0;
}

void rls_model_order_print(const RLSModelOrder *order) {
    if (!order) { printf("NULL model order\n"); return; }
    printf("Model type=%d", order->type);
    switch (order->type) {
        case RLS_MODEL_FIR:   printf(" (FIR: nb=%d)", order->nb); break;
        case RLS_MODEL_ARX:   printf(" (ARX: na=%d nb=%d nk=%d)", order->na, order->nb, order->nk); break;
        case RLS_MODEL_OE:    printf(" (OE: nb=%d nf=%d nk=%d)", order->nb, order->nf, order->nk); break;
        case RLS_MODEL_ARMAX: printf(" (ARMAX: na=%d nb=%d nc=%d nk=%d)", order->na, order->nb, order->nc, order->nk); break;
        case RLS_MODEL_BJ:    printf(" (BJ: nb=%d nc=%d nd=%d nf=%d nk=%d)", order->nb, order->nc, order->nd, order->nf, order->nk); break;
        case RLS_MODEL_SS:    printf(" (SS: nx=%d)", order->nx); break;
        case RLS_MODEL_NARX:  printf(" (NARX: na=%d nb=%d)", order->na, order->nb); break;
    }
    printf(" ts=%.4f\n", order->ts);
}
