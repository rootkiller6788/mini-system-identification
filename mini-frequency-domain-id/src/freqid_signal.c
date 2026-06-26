#include "freqid_defs.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Signal generation utilities for system identification experiments */

int freqid_signal_sine(double freq_hz, double fs, double duration, double amp,
                        double phase_deg, double **t_out, double **y_out, size_t *n_out) {
    if (freq_hz <= 0.0 || fs <= 0.0 || duration <= 0.0 || !n_out) return -1;
    size_t N = (size_t)(duration * fs);
    if (N < 2) return -1;
    double *t = (double *)malloc(N*sizeof(double));
    double *y = (double *)malloc(N*sizeof(double));
    if (!t || !y) { free(t); free(y); return -1; }
    double Ts = 1.0/fs, phase_rad = phase_deg * M_PI / 180.0;
    for (size_t i = 0; i < N; i++) {
        t[i] = (double)i * Ts;
        y[i] = amp * sin(2.0*M_PI*freq_hz*t[i] + phase_rad);
    }
    *t_out = t; *y_out = y; *n_out = N;
    return 0;
}

int freqid_signal_chirp(double f0, double f1, double fs, double duration, double amp,
                          double **t_out, double **y_out, size_t *n_out) {
    if (f0 <= 0.0 || f1 <= f0 || fs <= 0.0 || duration <= 0.0 || !n_out) return -1;
    size_t N = (size_t)(duration * fs);
    if (N < 2) return -1;
    double *t = (double *)malloc(N*sizeof(double));
    double *y = (double *)malloc(N*sizeof(double));
    if (!t || !y) { free(t); free(y); return -1; }
    double Ts = 1.0/fs;
    double k = (f1 - f0) / duration; /* chirp rate */
    for (size_t i = 0; i < N; i++) {
        t[i] = (double)i * Ts;
        y[i] = amp * sin(2.0*M_PI*(f0*t[i] + 0.5*k*t[i]*t[i]));
    }
    *t_out = t; *y_out = y; *n_out = N;
    return 0;
}

int freqid_signal_prbs(size_t n_stages, size_t N, double amp,
                        double **y_out) {
    if (n_stages < 2 || n_stages > 32 || N == 0 || !y_out) return -1;
    double *y = (double *)malloc(N*sizeof(double));
    if (!y) return -1;
    /* PRBS generator using LFSR */
    unsigned int lfsr = 0xACE1u; /* non-zero seed */
    for (size_t i = 0; i < N; i++) {
        unsigned int bit = ((lfsr >> (n_stages-1)) ^ (lfsr >> (n_stages-2))) & 1u;
        lfsr = ((lfsr << 1) | bit) & ((1u << n_stages) - 1u);
        y[i] = (lfsr & 1u) ? amp : -amp;
    }
    *y_out = y;
    return 0;
}

/* Butterworth low-pass filter (second-order, bilinear transform design) */
typedef struct { double b0, b1, b2, a1, a2; double x1, x2, y1, y2; } freqid_butter_lp;

int freqid_butter_lp_design(double fc, double fs, freqid_butter_lp *filt) {
    if (fc <= 0.0 || fs <= 0.0 || fc >= fs/2.0 || !filt) return -1;
    double w0 = 2.0 * M_PI * fc / fs;
    double cos_w0 = cos(w0);
    double alpha = sin(w0) / sqrt(2.0); /* Q = 1/sqrt(2) for Butterworth */
    double b0 = (1.0 - cos_w0) / 2.0;
    double b1 = 1.0 - cos_w0;
    double b2 = (1.0 - cos_w0) / 2.0;
    double a0 = 1.0 + alpha;
    double a1 = -2.0 * cos_w0;
    double a2 = 1.0 - alpha;
    filt->b0 = b0 / a0; filt->b1 = b1 / a0; filt->b2 = b2 / a0;
    filt->a1 = a1 / a0; filt->a2 = a2 / a0;
    filt->x1 = 0.0; filt->x2 = 0.0; filt->y1 = 0.0; filt->y2 = 0.0;
    return 0;
}

double freqid_butter_lp_step(freqid_butter_lp *filt, double x) {
    if (!filt) return 0.0;
    double y = filt->b0*x + filt->b1*filt->x1 + filt->b2*filt->x2
             - filt->a1*filt->y1 - filt->a2*filt->y2;
    filt->x2 = filt->x1; filt->x1 = x;
    filt->y2 = filt->y1; filt->y1 = y;
    return y;
}

void freqid_butter_lp_filter(double *x, size_t N, double fc, double fs, double *y) {
    if (!x || !y || N == 0) return;
    freqid_butter_lp filt;
    if (freqid_butter_lp_design(fc, fs, &filt) != 0) return;
    for (size_t i = 0; i < N; i++) y[i] = freqid_butter_lp_step(&filt, x[i]);
}

/* Data detrending: remove linear trend */
void freqid_signal_detrend(double *x, size_t N) {
    if (!x || N < 2) return;
    double sum_x = 0.0, sum_t = 0.0, sum_xt = 0.0, sum_tt = 0.0;
    for (size_t i = 0; i < N; i++) {
        double t = (double)i;
        sum_x += x[i]; sum_t += t; sum_xt += x[i]*t; sum_tt += t*t;
    }
    double denom = (double)N*sum_tt - sum_t*sum_t;
    if (fabs(denom) < 1e-15) return;
    double a = ((double)N*sum_xt - sum_t*sum_x) / denom;
    double b = (sum_x - a*sum_t) / (double)N;
    for (size_t i = 0; i < N; i++) x[i] -= (a*(double)i + b);
}

/* Data normalization: zero mean, unit variance */
void freqid_signal_normalize(double *x, size_t N) {
    if (!x || N < 2) return;
    double mean = 0.0, var = 0.0;
    for (size_t i = 0; i < N; i++) mean += x[i];
    mean /= (double)N;
    for (size_t i = 0; i < N; i++) {
        double d = x[i] - mean; var += d*d;
    }
    var /= (double)N;
    double std = sqrt(var);
    if (std < 1e-15) return;
    for (size_t i = 0; i < N; i++) x[i] = (x[i] - mean) / std;
}


/* ================================================================
 * Cross-correlation function (L2)
 * ================================================================
 * R_xy[tau] = (1/N) * sum x[n]*y[n+tau]
 * Used for time-delay estimation and system identification.
 */
int freqid_signal_crosscorr(const double *x, const double *y, size_t N,
                             size_t max_lag, double **r_out) {
    if (!x || !y || N == 0 || max_lag >= N || !r_out) return -1;
    double *r = (double *)malloc((2*max_lag + 1) * sizeof(double));
    if (!r) return -1;
    for (size_t k = 0; k <= max_lag; k++) {
        double sum = 0.0;
        for (size_t n = 0; n < N - k; n++) sum += x[n] * y[n + k];
        r[max_lag + k] = sum / (double)(N - k);
        if (k > 0) {
            sum = 0.0;
            for (size_t n = 0; n < N - k; n++) sum += y[n] * x[n + k];
            r[max_lag - k] = sum / (double)(N - k);
        }
    }
    *r_out = r;
    return 0;
}

/* ================================================================
 * Peak-to-peak and RMS amplitude (L2)
 * ================================================================ */
double freqid_signal_rms(const double *x, size_t N) {
    if (!x || N == 0) return 0.0;
    double sum2 = 0.0;
    for (size_t i = 0; i < N; i++) sum2 += x[i]*x[i];
    return sqrt(sum2 / (double)N);
}

double freqid_signal_peak_to_peak(const double *x, size_t N) {
    if (!x || N == 0) return 0.0;
    double xmin = x[0], xmax = x[0];
    for (size_t i = 1; i < N; i++) {
        if (x[i] < xmin) xmin = x[i];
        if (x[i] > xmax) xmax = x[i];
    }
    return xmax - xmin;
}
