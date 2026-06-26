#include "freqid_param.h"
#include "freqid_identify.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ===== ARX Model ===== */
int freqid_arx_create(freqid_arx_model *arx, size_t na, size_t nb, size_t nk) {
    if (!arx || na == 0) return -1;
    arx->na = na; arx->nb = nb; arx->nk = nk; arx->fitted = 0;
    arx->A = (double *)calloc(na + 1, sizeof(double));
    arx->B = (double *)calloc((nb > nk ? nb - nk + 1 : 1), sizeof(double));
    if (!arx->A || !arx->B) { free(arx->A); free(arx->B); return -1; }
    arx->A[0] = 1.0;
    return 0;
}
void freqid_arx_free(freqid_arx_model *arx) { if (arx) { free(arx->A); free(arx->B); } }

freqid_complex freqid_arx_eval(const freqid_arx_model *arx, double omega, double Ts) {
    freqid_complex z = cexp(I * omega * Ts);
    freqid_complex z_inv = 1.0 / z;
    freqid_complex A_val = 0.0, B_val = 0.0;
    freqid_complex zp = 1.0;
    for (size_t i = 0; i <= arx->na; i++) { A_val += arx->A[i] * zp; zp *= z_inv; }
    zp = 1.0;
    for (size_t i = 0; i <= arx->nb; i++) { B_val += arx->B[i] * zp; zp *= z_inv; }
    double Am2 = creal(A_val)*creal(A_val) + cimag(A_val)*cimag(A_val);
    return (Am2 > 1e-30) ? (B_val / A_val) : 0.0;
}

freqid_transfer_function *freqid_arx_to_tf(const freqid_arx_model *arx, double Ts) {
    if (!arx) return NULL;
    freqid_transfer_function *tf = (freqid_transfer_function *)calloc(1, sizeof(freqid_transfer_function));
    if (!tf) return NULL;
    tf->num_order = arx->nb; tf->den_order = arx->na; tf->is_discrete = 1;
    tf->num = (double *)malloc((arx->nb + 1) * sizeof(double));
    tf->den = (double *)malloc((arx->na + 1) * sizeof(double));
    if (!tf->num || !tf->den) { freqid_tf_free(tf); return NULL; }
    memcpy(tf->num, arx->B, (arx->nb + 1) * sizeof(double));
    memcpy(tf->den, arx->A, (arx->na + 1) * sizeof(double));
    return tf;
}

/* ===== OE Model (Output Error) ===== */
int freqid_oe_create(freqid_oe_model *oe, size_t nb, size_t nf, size_t nk) {
    if (!oe || nf == 0) return -1;
    oe->nb = nb; oe->nf = nf; oe->nk = nk;
    oe->B = (double *)calloc(nb + 1, sizeof(double));
    oe->F = (double *)calloc(nf + 1, sizeof(double));
    if (!oe->B || !oe->F) { free(oe->B); free(oe->F); return -1; }
    oe->F[0] = 1.0;
    return 0;
}
void freqid_oe_free(freqid_oe_model *oe) { if (oe) { free(oe->B); free(oe->F); } }

freqid_complex freqid_oe_eval(const freqid_oe_model *oe, double omega, double Ts) {
    freqid_complex z = cexp(I * omega * Ts);
    freqid_complex z_inv = 1.0 / z;
    freqid_complex B_val = 0.0, F_val = 0.0;
    freqid_complex zp = 1.0;
    for (size_t i = 0; i <= oe->nb; i++) { B_val += oe->B[i] * zp; zp *= z_inv; }
    zp = 1.0;
    for (size_t i = 0; i <= oe->nf; i++) { F_val += oe->F[i] * zp; zp *= z_inv; }
    double Fm2 = creal(F_val)*creal(F_val) + cimag(F_val)*cimag(F_val);
    return (Fm2 > 1e-30) ? (B_val / F_val) : 0.0;
}

/* ===== ARMAX Model ===== */
int freqid_armax_create(freqid_armax_model *armax, size_t na, size_t nb, size_t nc, size_t nk) {
    if (!armax || na == 0) return -1;
    armax->na = na; armax->nb = nb; armax->nc = nc; armax->nk = nk;
    armax->A = (double *)calloc(na + 1, sizeof(double));
    armax->B = (double *)calloc(nb + 1, sizeof(double));
    armax->C = (double *)calloc(nc + 1, sizeof(double));
    if (!armax->A || !armax->B || !armax->C) { free(armax->A); free(armax->B); free(armax->C); return -1; }
    armax->A[0] = 1.0; armax->C[0] = 1.0;
    return 0;
}
void freqid_armax_free(freqid_armax_model *armax) { if (armax) { free(armax->A); free(armax->B); free(armax->C); } }

/* ===== Box-Jenkins Model ===== */
int freqid_bj_create(freqid_bj_model *bj, size_t nb, size_t nc, size_t nd, size_t nf, size_t nk) {
    if (!bj || nf == 0 || nd == 0) return -1;
    bj->nb = nb; bj->nc = nc; bj->nd = nd; bj->nf = nf; bj->nk = nk;
    bj->B = (double *)calloc(nb + 1, sizeof(double));
    bj->C = (double *)calloc(nc + 1, sizeof(double));
    bj->D = (double *)calloc(nd + 1, sizeof(double));
    bj->F = (double *)calloc(nf + 1, sizeof(double));
    if (!bj->B || !bj->C || !bj->D || !bj->F) { freqid_bj_free(bj); return -1; }
    bj->D[0] = 1.0; bj->F[0] = 1.0; bj->C[0] = 1.0;
    return 0;
}
void freqid_bj_free(freqid_bj_model *bj) { if (bj) { free(bj->B); free(bj->C); free(bj->D); free(bj->F); } }

/* ===== Model Order Selection (L5) ===== */

/* Evaluate model with orders (m,n) and compute AIC */
static double _eval_aic(const freqid_frf *frf, size_t m, size_t n, int discrete) {
    freqid_transfer_function *tf = freqid_ls_fit(frf, NULL, m, n, discrete, 30, 1e-4);
    if (!tf) return 1e308;
    freqid_frf *frf_model = freqid_tf_eval_frf(tf, &frf->freq);
    if (!frf_model) { freqid_tf_free(tf); return 1e308; }
    size_t n_par = m + n + 1;
    double aic = freqid_aic(frf, frf_model, n_par, frf->freq.n);
    freqid_tf_free(tf); freqid_frf_free(frf_model);
    return aic;
}

int freqid_order_select_aic(const freqid_frf *frf, size_t n_max, int discrete,
                             size_t *best_m, size_t *best_n) {
    if (!frf || n_max == 0 || !best_m || !best_n) return -1;
    double best_aic = 1e308;
    *best_m = 0; *best_n = 1;
    for (size_t n = 1; n <= n_max; n++) {
        for (size_t m = 0; m <= n; m++) {
            double aic = _eval_aic(frf, m, n, discrete);
            if (aic < best_aic) { best_aic = aic; *best_m = m; *best_n = n; }
        }
    }
    return 0;
}

/* BIC: penalizes model complexity more heavily than AIC */
static double _eval_bic(const freqid_frf *frf, size_t m, size_t n, int discrete) {
    freqid_transfer_function *tf = freqid_ls_fit(frf, NULL, m, n, discrete, 30, 1e-4);
    if (!tf) return 1e308;
    freqid_frf *frf_model = freqid_tf_eval_frf(tf, &frf->freq);
    if (!frf_model) { freqid_tf_free(tf); return 1e308; }
    size_t n_par = m + n + 1;
    double wsse = freqid_wsse(frf, frf_model, NULL);
    size_t n_data = frf->freq.n;
    double bic = (n_data > 0 && wsse > 0)
        ? (double)n_data * log(wsse / (double)n_data) + (double)n_par * log((double)n_data)
        : 1e308;
    freqid_tf_free(tf); freqid_frf_free(frf_model);
    return bic;
}

int freqid_order_select_bic(const freqid_frf *frf, size_t n_max, int discrete,
                             size_t *best_m, size_t *best_n) {
    if (!frf || n_max == 0 || !best_m || !best_n) return -1;
    double best_bic = 1e308;
    *best_m = 0; *best_n = 1;
    for (size_t n = 1; n <= n_max; n++) {
        for (size_t m = 0; m <= n; m++) {
            double bic = _eval_bic(frf, m, n, discrete);
            if (bic < best_bic) { best_bic = bic; *best_m = m; *best_n = n; }
        }
    }
    return 0;
}

/* Cross-Validation based order selection */
int freqid_order_select_cv(const freqid_frf *frf_train, const freqid_frf *frf_valid,
                            size_t n_max, int discrete, size_t *best_m, size_t *best_n) {
    if (!frf_train || !frf_valid || n_max == 0 || !best_m || !best_n) return -1;
    double best_err = 1e308;
    *best_m = 0; *best_n = 1;
    for (size_t n = 1; n <= n_max; n++) {
        for (size_t m = 0; m <= n; m++) {
            freqid_transfer_function *tf = freqid_ls_fit(frf_train, NULL, m, n, discrete, 30, 1e-4);
            if (!tf) continue;
            freqid_frf *frf_model = freqid_tf_eval_frf(tf, &frf_valid->freq);
            if (!frf_model) { freqid_tf_free(tf); continue; }
            double err = freqid_wsse(frf_valid, frf_model, NULL);
            if (err > 0 && err < best_err) { best_err = err; *best_m = m; *best_n = n; }
            freqid_tf_free(tf); freqid_frf_free(frf_model);
        }
    }
    return (best_err < 1e308) ? 0 : -1;
}

/* ===== Continuous-to-Discrete conversion =====
 * Full implementations in freqid_convert.c:
 *   - freqid_c2d_tustin (bilinear/Tustin)
 *   - freqid_c2d_zoh (zero-order hold)
 *   - freqid_d2c_tustin (inverse Tustin)
 */


/* ================================================================
 * Continuous-time model structure estimation from FRF (L5)
 * ================================================================
 * Identifies standard model structures from FRF data by
 * analyzing the asymptotic slope (low/high frequency behavior).
 */

/* Estimate relative degree (pole excess) from high-frequency slope */
int freqid_estimate_pole_excess(const freqid_frf *frf, int *n_excess) {
    if (!frf || !frf->points || frf->freq.n < 10 || !n_excess) return -1;
    /* Use last 10% of frequency range */
    size_t start = frf->freq.n * 9 / 10;
    if (start >= frf->freq.n - 2) start = frf->freq.n - 3;
    double sum_log_w = 0.0, sum_log_mag = 0.0, sum_log_w_log_mag = 0.0, sum_log_w2 = 0.0;
    size_t count = 0;
    for (size_t i = start; i < frf->freq.n; i++) {
        double log_w = log10(frf->freq.w[i] + 1e-15);
        double mag_db = frf->points[i].db;
        if (mag_db < -200.0) continue;
        sum_log_w += log_w; sum_log_mag += mag_db;
        sum_log_w_log_mag += log_w * mag_db; sum_log_w2 += log_w * log_w;
        count++;
    }
    if (count < 3) return -1;
    double denom = (double)count*sum_log_w2 - sum_log_w*sum_log_w;
    if (fabs(denom) < 1e-15) return -1;
    double slope = ((double)count*sum_log_w_log_mag - sum_log_w*sum_log_mag) / denom;
    /* Slope in dB/decade -> pole excess: -20*(n-m) dB/decade */
    *n_excess = (int)(-slope / 20.0 + 0.5);
    if (*n_excess < 0) *n_excess = 0;
    return 0;
}

/* Estimate DC gain from low-frequency FRF magnitude */
double freqid_estimate_dc_gain(const freqid_frf *frf) {
    if (!frf || !frf->points || frf->freq.n == 0) return 0.0;
    double dc_mag = frf->points[0].magnitude;
    /* Average first few points for robustness */
    size_t n_avg = frf->freq.n > 5 ? 5 : frf->freq.n;
    double sum = 0.0;
    for (size_t i = 0; i < n_avg; i++) sum += frf->points[i].magnitude;
    return sum / (double)n_avg;
}

/* Detect integrator (pole at origin) from low-frequency -20 dB/dec slope */
int freqid_detect_integrator(const freqid_frf *frf, double *has_integrator) {
    if (!frf || !frf->points || frf->freq.n < 10 || !has_integrator) return -1;
    size_t n_low = frf->freq.n / 10;
    if (n_low < 3) n_low = 3;
    double sum_log_w = 0.0, sum_log_mag = 0.0, sum_prod = 0.0, sum_log_w2 = 0.0;
    for (size_t i = 0; i < n_low; i++) {
        double log_w = log10(frf->freq.w[i] + 1e-15);
        double mag_db = frf->points[i].db;
        if (mag_db < -200.0) continue;
        sum_log_w += log_w; sum_log_mag += mag_db;
        sum_prod += log_w*mag_db; sum_log_w2 += log_w*log_w;
    }
    double denom = (double)n_low*sum_log_w2 - sum_log_w*sum_log_w;
    if (fabs(denom) < 1e-15) { *has_integrator = 0.0; return 0; }
    double slope = ((double)n_low*sum_prod - sum_log_w*sum_log_mag) / denom;
    *has_integrator = (slope < -15.0) ? 1.0 : 0.0; /* -20 dB/dec -> integrator */
    return 0;
}

/* ================================================================
 * Nonlinear ARX (NARX) model for frequency-domain interpretation
 * ================================================================
 * NARX extends ARX with nonlinear regressors (polynomial terms).
 * The frequency response is input-amplitude-dependent, characterized
 * by the Best Linear Approximation (BLA) framework.
 * Ref: Pintelon & Schoukens (2012), Ch. 4.
 */

typedef struct { size_t na, nb, nk, degree; double *A, *B; } freqid_narx_model;

int freqid_narx_create(freqid_narx_model *narx, size_t na, size_t nb,
                        size_t nk, size_t degree) {
    if (!narx || na == 0 || degree < 1 || degree > 5) return -1;
    narx->na = na; narx->nb = nb; narx->nk = nk; narx->degree = degree;
    narx->A = (double *)calloc(na + 1, sizeof(double));
    narx->B = (double *)calloc((nb + 1) * degree, sizeof(double));
    if (!narx->A || !narx->B) { free(narx->A); free(narx->B); return -1; }
    narx->A[0] = 1.0;
    return 0;
}

void freqid_narx_free(freqid_narx_model *narx) {
    if (narx) { free(narx->A); free(narx->B); }
}

/* BLA computation: average FRF over multiple excitation amplitudes */
freqid_frf *freqid_bla_estimate(const freqid_frf **frf_array,
                                 size_t n_amplitudes) {
    if (!frf_array || n_amplitudes == 0) return NULL;
    size_t n_freq = frf_array[0]->freq.n;
    freqid_frf *bla = (freqid_frf *)calloc(1, sizeof(freqid_frf));
    if (!bla) return NULL;
    bla->freq.n = n_freq;
    bla->freq.w = (double *)malloc(n_freq * sizeof(double));
    bla->points = (freqid_frf_point *)calloc(n_freq, sizeof(freqid_frf_point));
    if (!bla->freq.w || !bla->points) { freqid_frf_free(bla); return NULL; }
    bla->freq.w_min = frf_array[0]->freq.w_min;
    bla->freq.w_max = frf_array[0]->freq.w_max;
    for (size_t k = 0; k < n_freq; k++) {
        bla->freq.w[k] = frf_array[0]->freq.w[k];
        freqid_complex sum = 0.0;
        for (size_t a = 0; a < n_amplitudes; a++)
            sum += frf_array[a]->points[k].value;
        bla->points[k].value = sum / (double)n_amplitudes;
        bla->points[k].magnitude = cabs(bla->points[k].value);
        bla->points[k].phase_deg = carg(bla->points[k].value) * 180.0/M_PI;
        double m = bla->points[k].magnitude;
        bla->points[k].db = (m > 1e-15) ? 20.0*log10(m) : -300.0;
    }
    return bla;
}
