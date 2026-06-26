#include "freqid_defs.h"
#include "freqid_spectrum.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

freqid_complex freqid_complex_make(double real, double imag) { return real + imag * I; }
double freqid_complex_real(freqid_complex z) { return creal(z); }
double freqid_complex_imag(freqid_complex z) { return cimag(z); }
freqid_complex freqid_complex_conj(freqid_complex z) { return conj(z); }
double freqid_complex_mag(freqid_complex z) { return cabs(z); }
double freqid_complex_phase_rad(freqid_complex z) { return carg(z); }
double freqid_complex_phase_deg(freqid_complex z) { return carg(z) * 180.0 / M_PI; }
double freqid_complex_db(freqid_complex z) {
    double mag = cabs(z);
    if (mag < 1e-15) return -300.0;
    return 20.0 * log10(mag);
}
freqid_complex freqid_complex_add(freqid_complex a, freqid_complex b) { return a + b; }
freqid_complex freqid_complex_sub(freqid_complex a, freqid_complex b) { return a - b; }
freqid_complex freqid_complex_mul(freqid_complex a, freqid_complex b) { return a * b; }
freqid_complex freqid_complex_div(freqid_complex a, freqid_complex b) {
    double denom = creal(b)*creal(b) + cimag(b)*cimag(b);
    if (denom < 1e-30) return 0.0;
    return a / b;
}

/* Frequency Vector */
int freqid_freq_vector_linear(freqid_freq_vector *fv, double w_min, double w_max, size_t n) {
    if (!fv || n < 2 || w_min >= w_max) return -1;
    fv->w_min = w_min; fv->w_max = w_max; fv->n = n;
    fv->w = (double *)malloc(n * sizeof(double));
    if (!fv->w) return -1;
    double dw = (w_max - w_min) / (double)(n - 1);
    for (size_t i = 0; i < n; i++) fv->w[i] = w_min + dw * (double)i;
    return 0;
}
int freqid_freq_vector_log(freqid_freq_vector *fv, double w_min, double w_max, size_t n) {
    if (!fv || n < 2 || w_min <= 0.0 || w_max <= w_min) return -1;
    fv->w_min = w_min; fv->w_max = w_max; fv->n = n;
    fv->w = (double *)malloc(n * sizeof(double));
    if (!fv->w) return -1;
    double log_min = log10(w_min), log_max = log10(w_max);
    double dlog = (log_max - log_min) / (double)(n - 1);
    for (size_t i = 0; i < n; i++) fv->w[i] = pow(10.0, log_min + dlog * (double)i);
    return 0;
}
void freqid_freq_vector_free(freqid_freq_vector *fv) {
    if (fv && fv->w) { free(fv->w); fv->w = NULL; }
}
/* FRF from DFT */
freqid_frf *freqid_frf_from_dft(const freqid_complex *U_fft, const freqid_complex *Y_fft,
                                 const double *freq_hz, size_t n_freq) {
    if (!U_fft || !Y_fft || n_freq == 0) return NULL;
    freqid_frf *frf = (freqid_frf *)calloc(1, sizeof(freqid_frf));
    if (!frf) return NULL;
    frf->freq.n = n_freq;
    frf->freq.w = (double *)malloc(n_freq * sizeof(double));
    frf->points = (freqid_frf_point *)calloc(n_freq, sizeof(freqid_frf_point));
    if (!frf->freq.w || !frf->points) { freqid_frf_free(frf); return NULL; }
    for (size_t i = 0; i < n_freq; i++) {
        double U_mag2 = creal(U_fft[i])*creal(U_fft[i]) + cimag(U_fft[i])*cimag(U_fft[i]);
        frf->freq.w[i] = freq_hz ? (2.0 * M_PI * freq_hz[i]) : (double)i;
        frf->points[i].value = (U_mag2 > 1e-30) ? (Y_fft[i] / U_fft[i]) : 0.0;
        frf->points[i].magnitude = cabs(frf->points[i].value);
        frf->points[i].phase_deg = carg(frf->points[i].value) * 180.0 / M_PI;
        double m = frf->points[i].magnitude;
        frf->points[i].db = (m > 1e-15) ? 20.0 * log10(m) : -300.0;
    }
    frf->freq.w_min = frf->freq.w[0];
    frf->freq.w_max = frf->freq.w[n_freq - 1];
    return frf;
}
void freqid_frf_free(freqid_frf *frf) {
    if (frf) { freqid_freq_vector_free(&frf->freq); free(frf->points); free(frf); }
}

/* Transfer Function */
int freqid_tf_create(freqid_transfer_function *tf,
                      const double *num, size_t num_order,
                      const double *den, size_t den_order, int is_discrete) {
    if (!tf || !num || !den || den_order == 0 || den[0] == 0.0 || num_order > den_order) return -1;
    tf->num_order = num_order; tf->den_order = den_order; tf->is_discrete = is_discrete;
    tf->num = (double *)malloc((num_order + 1) * sizeof(double));
    tf->den = (double *)malloc((den_order + 1) * sizeof(double));
    if (!tf->num || !tf->den) { free(tf->num); free(tf->den); return -1; }
    memcpy(tf->num, num, (num_order + 1) * sizeof(double));
    memcpy(tf->den, den, (den_order + 1) * sizeof(double));
    return 0;
}
static freqid_complex poly_eval(const double *coeff, size_t order, freqid_complex s) {
    freqid_complex r = 0.0, sp = 1.0;
    for (size_t k = 0; k <= order; k++) { r += coeff[k] * sp; sp *= s; }
    return r;
}
freqid_complex freqid_tf_eval(const freqid_transfer_function *tf, freqid_complex s) {
    if (!tf || !tf->num || !tf->den) return 0.0;
    freqid_complex num_v = poly_eval(tf->num, tf->num_order, s);
    freqid_complex den_v = poly_eval(tf->den, tf->den_order, s);
    double dm2 = creal(den_v)*creal(den_v) + cimag(den_v)*cimag(den_v);
    return (dm2 < 1e-30) ? 0.0 : (num_v / den_v);
}
freqid_frf *freqid_tf_eval_frf(const freqid_transfer_function *tf, const freqid_freq_vector *fv) {
    if (!tf || !fv || !fv->w || fv->n == 0) return NULL;
    freqid_frf *frf = (freqid_frf *)calloc(1, sizeof(freqid_frf));
    if (!frf) return NULL;
    frf->freq.n = fv->n; frf->freq.w_min = fv->w_min; frf->freq.w_max = fv->w_max;
    frf->freq.w = (double *)malloc(fv->n * sizeof(double));
    frf->points = (freqid_frf_point *)calloc(fv->n, sizeof(freqid_frf_point));
    if (!frf->freq.w || !frf->points) { freqid_frf_free(frf); return NULL; }
    for (size_t i = 0; i < fv->n; i++) {
        frf->freq.w[i] = fv->w[i];
        freqid_complex s = tf->is_discrete ? cexp(I * fv->w[i]) : (I * fv->w[i]);
        frf->points[i].value = freqid_tf_eval(tf, s);
        frf->points[i].magnitude = cabs(frf->points[i].value);
        frf->points[i].phase_deg = carg(frf->points[i].value) * 180.0 / M_PI;
        double m = frf->points[i].magnitude;
        frf->points[i].db = (m > 1e-15) ? 20.0 * log10(m) : -300.0;
    }
    return frf;
}
void freqid_tf_free(freqid_transfer_function *tf) {
    if (tf) { free(tf->num); free(tf->den); tf->num = NULL; tf->den = NULL; }
}

/* Bode Data */
freqid_bode_data *freqid_bode_from_frf(const freqid_frf *frf) {
    if (!frf || !frf->points || frf->freq.n == 0) return NULL;
    freqid_bode_data *bd = (freqid_bode_data *)calloc(1, sizeof(freqid_bode_data));
    if (!bd) return NULL;
    bd->freq.n = frf->freq.n; bd->freq.w_min = frf->freq.w_min; bd->freq.w_max = frf->freq.w_max;
    bd->freq.w = (double *)malloc(frf->freq.n * sizeof(double));
    bd->mag_db = (double *)malloc(frf->freq.n * sizeof(double));
    bd->phase_deg = (double *)malloc(frf->freq.n * sizeof(double));
    if (!bd->freq.w || !bd->mag_db || !bd->phase_deg) { freqid_bode_free(bd); return NULL; }
    for (size_t i = 0; i < frf->freq.n; i++) {
        bd->freq.w[i] = frf->freq.w[i];
        bd->mag_db[i] = frf->points[i].db;
        bd->phase_deg[i] = frf->points[i].phase_deg;
    }
    return bd;
}
freqid_bode_data *freqid_bode_from_tf(const freqid_transfer_function *tf, const freqid_freq_vector *fv) {
    freqid_frf *frf = freqid_tf_eval_frf(tf, fv);
    if (!frf) return NULL;
    freqid_bode_data *bd = freqid_bode_from_frf(frf);
    freqid_frf_free(frf);
    return bd;
}
void freqid_bode_free(freqid_bode_data *bd) {
    if (bd) { freqid_freq_vector_free(&bd->freq); free(bd->mag_db); free(bd->phase_deg); free(bd); }
}
/* Nyquist Data */
freqid_nyquist_data *freqid_nyquist_from_frf(const freqid_frf *frf) {
    if (!frf || !frf->points || frf->freq.n == 0) return NULL;
    freqid_nyquist_data *nd = (freqid_nyquist_data *)calloc(1, sizeof(freqid_nyquist_data));
    if (!nd) return NULL;
    nd->n = frf->freq.n;
    nd->real_part = (double *)malloc(nd->n * sizeof(double));
    nd->imag_part = (double *)malloc(nd->n * sizeof(double));
    if (!nd->real_part || !nd->imag_part) { freqid_nyquist_free(nd); return NULL; }
    for (size_t i = 0; i < nd->n; i++) {
        nd->real_part[i] = creal(frf->points[i].value);
        nd->imag_part[i] = cimag(frf->points[i].value);
    }
    return nd;
}
void freqid_nyquist_free(freqid_nyquist_data *nd) { if (nd) { free(nd->real_part); free(nd->imag_part); free(nd); } }
/* Nichols Data */
freqid_nichols_data *freqid_nichols_from_frf(const freqid_frf *frf) {
    if (!frf || !frf->points || frf->freq.n == 0) return NULL;
    freqid_nichols_data *nd = (freqid_nichols_data *)calloc(1, sizeof(freqid_nichols_data));
    if (!nd) return NULL;
    nd->n = frf->freq.n;
    nd->mag_db = (double *)malloc(nd->n * sizeof(double));
    nd->phase_deg = (double *)malloc(nd->n * sizeof(double));
    if (!nd->mag_db || !nd->phase_deg) { freqid_nichols_free(nd); return NULL; }
    for (size_t i = 0; i < nd->n; i++) {
        nd->mag_db[i] = frf->points[i].db;
        nd->phase_deg[i] = frf->points[i].phase_deg;
    }
    return nd;
}
void freqid_nichols_free(freqid_nichols_data *nd) { if (nd) { free(nd->mag_db); free(nd->phase_deg); free(nd); } }

/* --- H1 Estimator: H1 = S_uy / S_uu --- */
freqid_frf *freqid_frf_h1_estimator(const double *u, const double *y,
                                     size_t n_data, double fs,
                                     size_t n_fft, double overlap) {
    if (!u || !y || n_data < n_fft || n_fft == 0 || fs <= 0.0) return NULL;
    double *freq_out = NULL, *psd_uu = NULL;
    freqid_complex *cpsd_uy = NULL;
    size_t n_freq = 0;
    if (freqid_psd_welch(u, n_data, fs, n_fft, overlap, FREQID_WIN_HANN,
                          &freq_out, &psd_uu, &n_freq) != 0) return NULL;
    if (freqid_cpsd_welch(u, y, n_data, fs, n_fft, overlap, FREQID_WIN_HANN,
                           NULL, &cpsd_uy, NULL) != 0) {
        free(freq_out); free(psd_uu); return NULL;
    }
    freqid_frf *frf = (freqid_frf *)calloc(1, sizeof(freqid_frf));
    if (!frf) { free(freq_out); free(psd_uu); free(cpsd_uy); return NULL; }
    frf->freq.n = n_freq; frf->freq.w_min = 2.0*M_PI*freq_out[0]; frf->freq.w_max = 2.0*M_PI*freq_out[n_freq-1];
    frf->freq.w = (double *)malloc(n_freq * sizeof(double));
    frf->points = (freqid_frf_point *)calloc(n_freq, sizeof(freqid_frf_point));
    if (!frf->freq.w || !frf->points) { freqid_frf_free(frf); free(freq_out); free(psd_uu); free(cpsd_uy); return NULL; }
    for (size_t i = 0; i < n_freq; i++) {
        frf->freq.w[i] = 2.0 * M_PI * freq_out[i];
        if (psd_uu[i] > 1e-30) frf->points[i].value = cpsd_uy[i] / psd_uu[i];
        else frf->points[i].value = 0.0;
        frf->points[i].magnitude = cabs(frf->points[i].value);
        frf->points[i].phase_deg = carg(frf->points[i].value) * 180.0 / M_PI;
        double m = frf->points[i].magnitude;
        frf->points[i].db = (m > 1e-15) ? 20.0 * log10(m) : -300.0;
    }
    free(freq_out); free(psd_uu); free(cpsd_uy);
    return frf;
}
/* --- H2 Estimator: H2 = S_yy / S_yu --- */
freqid_frf *freqid_frf_h2_estimator(const double *u, const double *y,
                                     size_t n_data, double fs,
                                     size_t n_fft, double overlap) {
    if (!u || !y || n_data < n_fft || n_fft == 0 || fs <= 0.0) return NULL;
    double *freq_out = NULL, *psd_yy = NULL;
    freqid_complex *cpsd_yu = NULL;
    size_t n_freq = 0;
    if (freqid_psd_welch(y, n_data, fs, n_fft, overlap, FREQID_WIN_HANN,
                          &freq_out, &psd_yy, &n_freq) != 0) return NULL;
    if (freqid_cpsd_welch(y, u, n_data, fs, n_fft, overlap, FREQID_WIN_HANN,
                           NULL, &cpsd_yu, NULL) != 0) {
        free(freq_out); free(psd_yy); return NULL;
    }
    freqid_frf *frf = (freqid_frf *)calloc(1, sizeof(freqid_frf));
    if (!frf) { free(freq_out); free(psd_yy); free(cpsd_yu); return NULL; }
    frf->freq.n = n_freq; frf->freq.w_min = 2.0*M_PI*freq_out[0]; frf->freq.w_max = 2.0*M_PI*freq_out[n_freq-1];
    frf->freq.w = (double *)malloc(n_freq * sizeof(double));
    frf->points = (freqid_frf_point *)calloc(n_freq, sizeof(freqid_frf_point));
    if (!frf->freq.w || !frf->points) { freqid_frf_free(frf); free(freq_out); free(psd_yy); free(cpsd_yu); return NULL; }
    for (size_t i = 0; i < n_freq; i++) {
        frf->freq.w[i] = 2.0 * M_PI * freq_out[i];
        double cpsd_mag2 = creal(cpsd_yu[i])*creal(cpsd_yu[i]) + cimag(cpsd_yu[i])*cimag(cpsd_yu[i]);
        if (cpsd_mag2 > 1e-30) frf->points[i].value = psd_yy[i] / conj(cpsd_yu[i]);
        else frf->points[i].value = 0.0;
        frf->points[i].magnitude = cabs(frf->points[i].value);
        frf->points[i].phase_deg = carg(frf->points[i].value) * 180.0 / M_PI;
        double m = frf->points[i].magnitude;
        frf->points[i].db = (m > 1e-15) ? 20.0 * log10(m) : -300.0;
    }
    free(freq_out); free(psd_yy); free(cpsd_yu);
    return frf;
}

/* --- Time-Domain Response via Controllable Canonical Form --- */
typedef struct { double *A, *B, *C, D; size_t n; double *x; } _ss_repr;
static int _tf2ss(const freqid_transfer_function *tf, _ss_repr *ss) {
    if (!tf || tf->den_order == 0) return -1;
    size_t n = tf->den_order; ss->n = n;
    ss->x = (double *)calloc(n, sizeof(double));
    ss->A = (double *)calloc(n * n, sizeof(double));
    ss->B = (double *)calloc(n, sizeof(double));
    ss->C = (double *)calloc(n, sizeof(double));
    if (!ss->x || !ss->A || !ss->B || !ss->C) return -1;
    for (size_t i = 0; i < n - 1; i++) ss->A[i * n + (i + 1)] = 1.0;
    double a0 = tf->den[0];
    for (size_t i = 0; i < n; i++) ss->A[(n - 1) * n + i] = -tf->den[n - i] / a0;
    ss->B[n - 1] = 1.0 / a0;
    for (size_t i = 0; i <= tf->num_order && i < n; i++) ss->C[i] = tf->num[i];
    ss->D = (tf->num_order >= tf->den_order) ? tf->num[tf->den_order] / a0 : 0.0;
    return 0;
}
static void _ss_free(_ss_repr *ss) { if (ss) { free(ss->x); free(ss->A); free(ss->B); free(ss->C); } }

freqid_time_response *freqid_impulse_response(const freqid_transfer_function *tf,
                                                double t_end, size_t n_pts) {
    if (!tf || n_pts < 2 || t_end <= 0.0) return NULL;
    _ss_repr ss = {0};
    if (_tf2ss(tf, &ss) != 0) return NULL;
    freqid_time_response *tr = (freqid_time_response *)calloc(1, sizeof(freqid_time_response));
    if (!tr) { _ss_free(&ss); return NULL; }
    tr->n = n_pts;
    tr->t = (double *)malloc(n_pts * sizeof(double));
    tr->y = (double *)malloc(n_pts * sizeof(double));
    if (!tr->t || !tr->y) { freqid_time_response_free(tr); _ss_free(&ss); return NULL; }
    double dt = t_end / (double)(n_pts - 1);
    for (size_t i = 0; i < ss.n; i++) ss.x[i] = ss.B[i] / dt; /* Dirac delta approx */
    for (size_t k = 0; k < n_pts; k++) {
        tr->t[k] = dt * (double)k;
        tr->y[k] = ss.D / dt;
        for (size_t i = 0; i < ss.n; i++) tr->y[k] += ss.C[i] * ss.x[i];
        double *dx = (double *)calloc(ss.n, sizeof(double));
        for (size_t i = 0; i < ss.n; i++)
            for (size_t j = 0; j < ss.n; j++)
                dx[i] += ss.A[i * ss.n + j] * ss.x[j];
        for (size_t i = 0; i < ss.n; i++) ss.x[i] += dt * dx[i];
        free(dx);
    }
    _ss_free(&ss); return tr;
}

freqid_time_response *freqid_step_response(const freqid_transfer_function *tf,
                                             double t_end, size_t n_pts) {
    if (!tf || n_pts < 2 || t_end <= 0.0) return NULL;
    _ss_repr ss = {0};
    if (_tf2ss(tf, &ss) != 0) return NULL;
    freqid_time_response *tr = (freqid_time_response *)calloc(1, sizeof(freqid_time_response));
    if (!tr) { _ss_free(&ss); return NULL; }
    tr->n = n_pts;
    tr->t = (double *)malloc(n_pts * sizeof(double));
    tr->y = (double *)malloc(n_pts * sizeof(double));
    if (!tr->t || !tr->y) { freqid_time_response_free(tr); _ss_free(&ss); return NULL; }
    double dt = t_end / (double)(n_pts - 1);
    for (size_t k = 0; k < n_pts; k++) {
        tr->t[k] = dt * (double)k;
        tr->y[k] = ss.D;
        for (size_t i = 0; i < ss.n; i++) tr->y[k] += ss.C[i] * ss.x[i];
        double *dx = (double *)calloc(ss.n, sizeof(double));
        for (size_t i = 0; i < ss.n; i++) {
            for (size_t j = 0; j < ss.n; j++) dx[i] += ss.A[i * ss.n + j] * ss.x[j];
            dx[i] += ss.B[i] * 1.0; /* unit step */
        }
        for (size_t i = 0; i < ss.n; i++) ss.x[i] += dt * dx[i];
        free(dx);
    }
    _ss_free(&ss); return tr;
}
void freqid_time_response_free(freqid_time_response *tr) {
    if (tr) { free(tr->t); free(tr->y); free(tr); }
}
