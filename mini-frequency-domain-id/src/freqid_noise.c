#include "freqid_defs.h"
#include "freqid_spectrum.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int freqid_frf_noise_variance(const freqid_frf *frf, const double *coherence,
                               size_t n_avg, double **var_out) {
    if (!frf || !frf->points || frf->freq.n == 0 || n_avg < 2 || !var_out) return -1;
    size_t n = frf->freq.n;
    double *var = (double *)malloc(n * sizeof(double));
    if (!var) return -1;
    for (size_t i = 0; i < n; i++) {
        double gamma2 = coherence ? coherence[i] : 0.9;
        if (gamma2 > 1.0) gamma2 = 1.0;
        if (gamma2 < 0.0) gamma2 = 0.0;
        double G_mag2 = frf->points[i].magnitude * frf->points[i].magnitude;
        var[i] = (1.0 - gamma2) / (2.0 * (double)n_avg) * G_mag2;
        if (gamma2 > 0.999) var[i] *= 0.1;
    }
    *var_out = var;
    return 0;
}

int freqid_frf_confidence_bounds(const freqid_frf *frf, const double *noise_var,
                                  double confidence, size_t n_avg,
                                  double **upper_db, double **lower_db) {
    if (!frf || !frf->points || frf->freq.n == 0 || !noise_var || !upper_db || !lower_db) return -1;
    size_t n = frf->freq.n;
    double *up = (double *)malloc(n * sizeof(double));
    double *lo = (double *)malloc(n * sizeof(double));
    if (!up || !lo) { free(up); free(lo); return -1; }
    double z = 1.96;
    if (confidence > 0.95) z = 2.576;
    else if (confidence < 0.90) z = 1.645;
    for (size_t i = 0; i < n; i++) {
        double G_mag = frf->points[i].magnitude;
        double std_dev = sqrt((noise_var[i] > 0) ? noise_var[i] : 1e-30);
        double delta_db = 20.0 * log10(1.0 + z * std_dev / (G_mag + 1e-15));
        up[i] = frf->points[i].db + delta_db;
        lo[i] = frf->points[i].db - delta_db;
    }
    *upper_db = up; *lower_db = lo;
    return 0;
}

int freqid_frf_snr(const freqid_frf *frf, const double *noise_var, double **snr_db_out) {
    if (!frf || !frf->points || frf->freq.n == 0 || !noise_var || !snr_db_out) return -1;
    size_t n = frf->freq.n;
    double *snr_db = (double *)malloc(n * sizeof(double));
    if (!snr_db) return -1;
    for (size_t i = 0; i < n; i++) {
        double G2 = frf->points[i].magnitude * frf->points[i].magnitude;
        double nv = noise_var[i] > 1e-30 ? noise_var[i] : 1e-30;
        snr_db[i] = 10.0 * log10(G2 / nv);
    }
    *snr_db_out = snr_db;
    return 0;
}

int freqid_residual_spectrum(const double *y_measured, const double *y_model,
                              size_t N, double fs, size_t n_fft, double overlap,
                              double **freq_out, double **psd_out, size_t *n_freq_out) {
    if (!y_measured || !y_model || N < n_fft || fs <= 0.0) return -1;
    double *residual = (double *)malloc(N * sizeof(double));
    if (!residual) return -1;
    for (size_t i = 0; i < N; i++) residual[i] = y_measured[i] - y_model[i];
    int ret = freqid_psd_welch(residual, N, fs, n_fft, overlap,
                                FREQID_WIN_HANN, freq_out, psd_out, n_freq_out);
    free(residual);
    return ret;
}

double freqid_ljung_box_test(const double *residual, size_t N, size_t max_lag) {
    if (!residual || N == 0 || max_lag == 0) return -1.0;
    double *r = NULL;
    if (freqid_autocorr(residual, N, max_lag, &r) != 0) return -1.0;
    double r0 = r[0];
    if (fabs(r0) < 1e-15) { free(r); return 0.0; }
    double Q = 0.0;
    for (size_t k = 1; k <= max_lag; k++) {
        double rho_k = r[k] / r0;
        Q += rho_k * rho_k / (double)(N - k);
    }
    Q *= (double)N * (double)(N + 2);
    free(r);
    return Q;
}

typedef struct { size_t p, q; double *ar, *ma; } freqid_arma_noise;

int freqid_arma_noise_fit(const double *residual, size_t N, size_t p, size_t q,
                           freqid_arma_noise *model) {
    if (!residual || !model || N < p+q+1) return -1;
    model->p = p; model->q = q;
    model->ar = (double *)calloc(p + 1, sizeof(double));
    model->ma = (double *)calloc(q + 1, sizeof(double));
    if (!model->ar || !model->ma) { free(model->ar); free(model->ma); return -1; }
    model->ar[0] = 1.0; model->ma[0] = 1.0;
    double *r = NULL;
    if (freqid_autocorr(residual, N, p + q, &r) != 0) {
        free(model->ar); free(model->ma); return -1;
    }
    if (p == 0) {
        for (size_t i = 0; i <= q; i++) model->ma[i] = (i == 0) ? 1.0 : 0.0;
    } else if (p == 1) {
        model->ar[1] = (fabs(r[0]) > 1e-15) ? (-r[1] / r[0]) : 0.0;
    } else {
        double *a = (double *)calloc(p + 1, sizeof(double));
        double *a_prev = (double *)calloc(p + 1, sizeof(double));
        double sigma2 = r[0];
        a[0] = 1.0;
        for (size_t k = 1; k <= p; k++) {
            double sum = 0.0;
            for (size_t j = 1; j < k; j++) sum += a_prev[j] * r[k - j];
            double kk = (r[k] - sum) / sigma2;
            a[k] = kk;
            for (size_t j = 1; j < k; j++) a[j] = a_prev[j] - kk * a_prev[k - j];
            sigma2 *= (1.0 - kk * kk);
            memcpy(a_prev, a, (p + 1) * sizeof(double));
        }
        model->ar[0] = 1.0;
        for (size_t i = 1; i <= p; i++) model->ar[i] = a[i];
        free(a); free(a_prev);
    }
    free(r);
    return 0;
}

void freqid_arma_noise_free(freqid_arma_noise *model) {
    if (model) { free(model->ar); free(model->ma); }
}

int freqid_generate_white_noise(size_t N, double std_dev, double **noise_out) {
    if (N == 0 || std_dev < 0.0 || !noise_out) return -1;
    double *nz = (double *)malloc(N * sizeof(double));
    if (!nz) return -1;
    unsigned int seed = 12345;
    for (size_t i = 0; i < N; i += 2) {
        double u1 = ((double)((seed = seed * 1103515245 + 12345) & 0x7fffffff)) / 0x7fffffff;
        double u2 = ((double)((seed = seed * 1103515245 + 12345) & 0x7fffffff)) / 0x7fffffff;
        double rr = sqrt(-2.0 * log(u1 + 1e-15));
        double theta = 2.0 * M_PI * u2;
        nz[i] = rr * cos(theta) * std_dev;
        if (i + 1 < N) nz[i + 1] = rr * sin(theta) * std_dev;
    }
    *noise_out = nz;
    return 0;
}
