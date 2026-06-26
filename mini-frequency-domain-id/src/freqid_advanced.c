#include "freqid_defs.h"
#include "freqid_spectrum.h"
#include "freqid_identify.h"
#include "freqid_app.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

freqid_frf *freqid_advanced_hv_estimator(const double *u, const double *y,
                                          size_t n_data, double fs,
                                          size_t n_fft, double overlap) {
    if (!u || !y || n_data < n_fft || n_fft == 0 || fs <= 0.0) return NULL;
    freqid_frf *H1 = freqid_frf_h1_estimator(u, y, n_data, fs, n_fft, overlap);
    freqid_frf *H2 = freqid_frf_h2_estimator(u, y, n_data, fs, n_fft, overlap);
    if (!H1 || !H2) { freqid_frf_free(H1); freqid_frf_free(H2); return NULL; }
    size_t n = H1->freq.n < H2->freq.n ? H1->freq.n : H2->freq.n;
    freqid_frf *Hv = (freqid_frf *)calloc(1, sizeof(freqid_frf));
    if (!Hv) { freqid_frf_free(H1); freqid_frf_free(H2); return NULL; }
    Hv->freq.n = n; Hv->freq.w_min = H1->freq.w_min; Hv->freq.w_max = H1->freq.w_max;
    Hv->freq.w = (double *)malloc(n * sizeof(double));
    Hv->points = (freqid_frf_point *)calloc(n, sizeof(freqid_frf_point));
    if (!Hv->freq.w || !Hv->points) { freqid_frf_free(Hv); freqid_frf_free(H1); freqid_frf_free(H2); return NULL; }
    for (size_t i = 0; i < n; i++) {
        Hv->freq.w[i] = H1->freq.w[i];
        double mag_H1 = H1->points[i].magnitude, mag_H2 = H2->points[i].magnitude;
        double phase_H1 = carg(H1->points[i].value);
        double mag_Hv = sqrt(mag_H1 * mag_H2);
        Hv->points[i].value = mag_Hv * (cos(phase_H1) + I * sin(phase_H1));
        Hv->points[i].magnitude = mag_Hv;
        Hv->points[i].phase_deg = phase_H1 * 180.0 / M_PI;
        Hv->points[i].db = (mag_Hv > 1e-15) ? 20.0*log10(mag_Hv) : -300.0;
    }
    freqid_frf_free(H1); freqid_frf_free(H2);
    return Hv;
}

freqid_state_space *freqid_advanced_freq_subspace(const freqid_frf *frf, size_t n_order) {
    if (!frf || !frf->points || frf->freq.n < n_order*2 || n_order == 0) return NULL;
    freqid_state_space *ss = (freqid_state_space *)calloc(1, sizeof(freqid_state_space));
    if (!ss) return NULL;
    ss->n_states = n_order*2; ss->n_inputs = 1; ss->n_outputs = 1;
    ss->A = (double *)calloc(ss->n_states*ss->n_states, sizeof(double));
    ss->B = (double *)calloc(ss->n_states, sizeof(double));
    ss->C = (double *)calloc(ss->n_states, sizeof(double));
    ss->D = (double *)calloc(1, sizeof(double));
    if (!ss->A || !ss->B || !ss->C || !ss->D) { freqid_ss_free(ss); return NULL; }
    for (size_t k = 0; k < n_order; k++) {
        size_t freq_idx = k * frf->freq.n / n_order;
        if (freq_idx >= frf->freq.n) freq_idx = frf->freq.n - 1;
        double wn = frf->freq.w[freq_idx];
        size_t i2 = 2*k; size_t ns = ss->n_states;
        ss->A[i2*ns + i2] = 0.0;
        if (i2+1 < ns) ss->A[i2*ns + i2 + 1] = 1.0;
        if (i2+1 < ns) {
            ss->A[(i2+1)*ns + i2] = -wn*wn;
            ss->A[(i2+1)*ns + i2 + 1] = -2.0*0.05*wn;
        }
        if (i2 < ns) ss->B[i2] = 0.0;
        if (i2+1 < ns) ss->B[i2+1] = 1.0;
        double mag = frf->points[freq_idx].magnitude;
        if (i2 < ns) ss->C[i2] = (mag > 1e-15) ? mag : 1.0;
    }
    ss->D[0] = 0.0;
    return ss;
}

int freqid_advanced_stft_spectrogram(const double *x, size_t N, double fs,
                                      size_t n_fft, size_t hop,
                                      freqid_window_type win_type,
                                      double ***spec_out, size_t *n_times,
                                      size_t *n_freqs) {
    if (!x || N < n_fft || n_fft < 2 || hop == 0 || fs <= 0.0 || !spec_out || !n_times || !n_freqs) return -1;
    size_t n_seg = (N - n_fft) / hop + 1;
    if (n_seg == 0) return -1;
    size_t nf = n_fft / 2 + 1;
    double **spec = (double **)malloc(n_seg * sizeof(double *));
    if (!spec) return -1;
    double *window = freqid_window_generate(win_type, n_fft, 3.0);
    if (!window) { free(spec); return -1; }
    double *seg = (double *)malloc(n_fft * sizeof(double));
    freqid_complex *X = (freqid_complex *)malloc(n_fft * sizeof(freqid_complex));
    if (!seg || !X) { free(window); free(seg); free(X); free(spec); return -1; }
    for (size_t s = 0; s < n_seg; s++) {
        spec[s] = (double *)malloc(nf * sizeof(double));
        if (!spec[s]) { for (size_t j=0;j<s;j++) free(spec[j]); free(spec); free(window); free(seg); free(X); return -1; }
        size_t start = s * hop;
        for (size_t i = 0; i < n_fft; i++) seg[i] = ((start+i) < N) ? x[start+i] * window[i] : 0.0;
        for (size_t i = 0; i < n_fft; i++) X[i] = seg[i] + 0.0*I;
        freqid_fft_radix2(X, n_fft, 0);
        for (size_t k = 0; k < nf; k++) spec[s][k] = 20.0*log10(cabs(X[k]) + 1e-15);
    }
    free(window); free(seg); free(X);
    *spec_out = spec; *n_times = n_seg; *n_freqs = nf;
    return 0;
}

void freqid_advanced_stft_free(double **spec, size_t n_times) {
    if (spec) { for (size_t i=0;i<n_times;i++) free(spec[i]); free(spec); }
}

/* Krylov subspace model reduction via Arnoldi iteration */
typedef struct { double *V; size_t n, k; } freqid_krylov_basis;

int freqid_krylov_arnoldi(const double *A, size_t n, const double *b,
                           size_t k, freqid_krylov_basis *basis) {
    if (!A || n == 0 || !b || k == 0 || k > n || !basis) return -1;
    double *V = (double *)calloc(n*(k+1), sizeof(double));
    double *H = (double *)calloc((k+1)*k, sizeof(double));
    if (!V || !H) { free(V); free(H); return -1; }
    double b_norm = 0.0;
    for (size_t i=0;i<n;i++) b_norm += b[i]*b[i];
    b_norm = sqrt(b_norm);
    if (b_norm < 1e-15) { free(V); free(H); return -1; }
    for (size_t i=0;i<n;i++) V[i] = b[i]/b_norm;
    for (size_t j=0;j<k;j++) {
        double *w = (double *)calloc(n, sizeof(double));
        for (size_t i=0;i<n;i++) for (size_t l=0;l<n;l++) w[i] += A[i*n+l] * V[j*n+l];
        for (size_t i=0;i<=j;i++) {
            double hij = 0.0;
            for (size_t l=0;l<n;l++) hij += V[i*n+l]*w[l];
            H[i*k+j] = hij;
            for (size_t l=0;l<n;l++) w[l] -= hij*V[i*n+l];
        }
        double w_norm = 0.0;
        for (size_t i=0;i<n;i++) w_norm += w[i]*w[i];
        w_norm = sqrt(w_norm);
        if (w_norm > 1e-15) { H[(j+1)*k+j] = w_norm; for (size_t i=0;i<n;i++) V[(j+1)*n+i] = w[i]/w_norm; }
        free(w);
    }
    basis->V = V; basis->n = n; basis->k = k;
    free(H);
    return 0;
}

void freqid_krylov_free(freqid_krylov_basis *basis) { if (basis) { free(basis->V); } }

/* Reduced-order model via Krylov projection */
freqid_state_space *freqid_krylov_reduce(const freqid_state_space *ss, size_t k) {
    if (!ss || k == 0 || k >= ss->n_states) return NULL;
    freqid_krylov_basis basis;
    if (freqid_krylov_arnoldi(ss->A, ss->n_states, ss->B, k, &basis) != 0) return NULL;
    freqid_state_space *red = (freqid_state_space *)calloc(1, sizeof(freqid_state_space));
    if (!red) { freqid_krylov_free(&basis); return NULL; }
    red->n_states = k; red->n_inputs = 1; red->n_outputs = 1;
    red->A = (double *)calloc(k*k, sizeof(double));
    red->B = (double *)calloc(k, sizeof(double));
    red->C = (double *)calloc(k, sizeof(double));
    red->D = (double *)calloc(1, sizeof(double));
    if (!red->A || !red->B || !red->C || !red->D) { freqid_ss_free(red); freqid_krylov_free(&basis); return NULL; }
    for (size_t i=0;i<k;i++) for (size_t j=0;j<k;j++)
        for (size_t l=0;l<ss->n_states;l++) for (size_t m=0;m<ss->n_states;m++)
            red->A[i*k+j] += basis.V[i*basis.n+l] * ss->A[l*ss->n_states+m] * basis.V[j*basis.n+m];
    for (size_t i=0;i<k;i++) for (size_t l=0;l<ss->n_states;l++)
        red->B[i] += basis.V[i*basis.n+l] * ss->B[l];
    for (size_t i=0;i<k;i++) for (size_t l=0;l<ss->n_states;l++)
        red->C[i] += ss->C[l] * basis.V[i*basis.n+l];
    red->D[0] = ss->D[0];
    freqid_krylov_free(&basis);
    return red;
}
