#include "freqid_spectrum.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ================================================================
 * Window Functions ！ Spectral Leakage Control (L2)
 * ================================================================ */

double *freqid_window_generate(freqid_window_type win_type, size_t N, double beta) {
    if (N == 0) return NULL;
    double *w = (double *)malloc(N * sizeof(double));
    if (!w) return NULL;
    double Nm1 = (double)(N - 1);
    switch (win_type) {
    case FREQID_WIN_RECTANGLE:
        for (size_t i = 0; i < N; i++) w[i] = 1.0;
        break;
    case FREQID_WIN_HANN:
        for (size_t i = 0; i < N; i++) w[i] = 0.5 * (1.0 - cos(2.0 * M_PI * (double)i / Nm1));
        break;
    case FREQID_WIN_HAMMING:
        for (size_t i = 0; i < N; i++) w[i] = 0.54 - 0.46 * cos(2.0 * M_PI * (double)i / Nm1);
        break;
    case FREQID_WIN_BLACKMAN:
        for (size_t i = 0; i < N; i++)
            w[i] = 0.42 - 0.5 * cos(2.0 * M_PI * (double)i / Nm1)
                       + 0.08 * cos(4.0 * M_PI * (double)i / Nm1);
        break;
    case FREQID_WIN_BARTLETT:
        for (size_t i = 0; i < N; i++) w[i] = 1.0 - fabs(2.0 * (double)i / Nm1 - 1.0);
        break;
    case FREQID_WIN_KAISER: {
        /* Kaiser-Bessel window using I0 Bessel function series approximation */
        double alpha = beta / 2.0;
        double denom = 1.0, term = 1.0;
        for (int k = 1; k <= 20; k++) { term *= (alpha/k)*(alpha/k); denom += term; }
        for (size_t i = 0; i < N; i++) {
            double arg = beta * sqrt(1.0 - pow(2.0*(double)i/Nm1 - 1.0, 2.0));
            double val = 1.0, tk = 1.0;
            for (int k = 1; k <= 20; k++) { tk *= (arg/(2.0*k))*(arg/(2.0*k)); val += tk; }
            w[i] = val / denom;
        }
        break;
    }
    default: free(w); return NULL;
    }
    return w;
}

void freqid_window_apply(double *x, const double *w, size_t N) {
    if (!x || !w) return;
    for (size_t i = 0; i < N; i++) x[i] *= w[i];
}

double freqid_window_coherent_gain(const double *w, size_t N) {
    if (!w || N == 0) return 1.0;
    double sum = 0.0;
    for (size_t i = 0; i < N; i++) sum += w[i];
    return sum / (double)N;
}

/* ================================================================
 * DFT ！ Direct O(N^2) Implementation (L3)
 * ================================================================
 * X[k] = sum_{n=0}^{N-1} x[n] * exp(-j*2*pi*k*n/N)
 */

int freqid_dft_real(const double *x, size_t N, freqid_complex **X_out) {
    if (!x || !X_out || N == 0) return -1;
    freqid_complex *X = (freqid_complex *)malloc(N * sizeof(freqid_complex));
    if (!X) return -1;
    for (size_t k = 0; k < N; k++) {
        X[k] = 0.0;
        for (size_t n = 0; n < N; n++) {
            double angle = -2.0 * M_PI * (double)k * (double)n / (double)N;
            X[k] += x[n] * (cos(angle) + I * sin(angle));
        }
    }
    *X_out = X;
    return 0;
}

/* Inverse DFT */
int freqid_idft(const freqid_complex *X, size_t N, double **x_out) {
    if (!X || !x_out || N == 0) return -1;
    double *x = (double *)malloc(N * sizeof(double));
    if (!x) return -1;
    for (size_t n = 0; n < N; n++) {
        freqid_complex sum = 0.0;
        for (size_t k = 0; k < N; k++) {
            double angle = 2.0 * M_PI * (double)k * (double)n / (double)N;
            sum += X[k] * (cos(angle) + I * sin(angle));
        }
        x[n] = creal(sum) / (double)N;
    }
    *x_out = x;
    return 0;
}

/* ================================================================
 * Radix-2 Cooley-Tukey FFT ！ O(N log N) (L5)
 * ================================================================ */

static size_t bit_reverse(size_t x, size_t log2n) {
    size_t n = 0;
    for (size_t i = 0; i < log2n; i++) {
        n = (n << 1) | (x & 1);
        x >>= 1;
    }
    return n;
}

void freqid_fft_radix2(freqid_complex *x, size_t N, int inv) {
    if (!x || N < 2) return;
    /* Compute log2(N) */
    size_t log2n = 0, temp = N;
    while (temp >>= 1) log2n++;
    /* Bit-reversal permutation */
    for (size_t i = 0; i < N; i++) {
        size_t j = bit_reverse(i, log2n);
        if (j > i) { freqid_complex t = x[i]; x[i] = x[j]; x[j] = t; }
    }
    /* Danielson-Lanczos: butterfly operations */
    for (size_t len = 2; len <= N; len <<= 1) {
        double angle = 2.0 * M_PI / (double)len * (inv ? 1.0 : -1.0);
        freqid_complex wlen = cos(angle) + I * sin(angle);
        for (size_t i = 0; i < N; i += len) {
            freqid_complex w = 1.0;
            size_t half = len / 2;
            for (size_t j = 0; j < half; j++) {
                freqid_complex u = x[i + j];
                freqid_complex v = x[i + j + half] * w;
                x[i + j] = u + v;
                x[i + j + half] = u - v;
                w *= wlen;
            }
        }
    }
    /* Scale for inverse FFT */
    if (inv) {
        for (size_t i = 0; i < N; i++) x[i] /= (double)N;
    }
}

void freqid_magnitude_spectrum(const freqid_complex *X, size_t N, double *mag_out) {
    if (!X || !mag_out) return;
    for (size_t i = 0; i < N; i++) mag_out[i] = cabs(X[i]);
}


/* ================================================================
 * Welch PSD Estimation (L5)
 * ================================================================
 * Algorithm:
 *   1. Partition signal into K overlapping segments of length N_fft
 *   2. Window each segment
 *   3. Compute periodogram: P_i(f) = |DFT(x_i)|^2 / (N*U)
 *   4. Average: P_welch(f) = (1/K)*sum P_i(f)
 */

int freqid_psd_welch(const double *x, size_t N, double fs,
                     size_t n_fft, double overlap,
                     freqid_window_type win_type,
                     double **freq_out, double **psd_out,
                     size_t *n_freq_out) {
    if (!x || N < n_fft || n_fft < 2 || fs <= 0.0 || !freq_out || !psd_out || !n_freq_out)
        return -1;
    if (overlap < 0.0 || overlap >= 1.0) overlap = 0.5;

    /* Generate window */
    double *window = freqid_window_generate(win_type, n_fft, 3.0);
    if (!window) return -1;

    /* Window power correction: U = (1/M) * sum(w[n]^2) */
    double U = 0.0;
    for (size_t i = 0; i < n_fft; i++) U += window[i] * window[i];
    U /= (double)n_fft;
    if (U < 1e-15) { free(window); return -1; }

    /* Step size between segments */
    size_t step = (size_t)((double)n_fft * (1.0 - overlap));
    if (step == 0) step = 1;
    size_t n_seg = (N - n_fft) / step + 1;
    if (n_seg == 0) { free(window); return -1; }

    size_t n_freq = n_fft / 2 + 1;
    double *psd_accum = (double *)calloc(n_freq, sizeof(double));
    if (!psd_accum) { free(window); return -1; }

    /* Process each segment */
    double *seg = (double *)malloc(n_fft * sizeof(double));
    freqid_complex *X = (freqid_complex *)malloc(n_fft * sizeof(freqid_complex));
    if (!seg || !X) { free(psd_accum); free(window); free(seg); free(X); return -1; }

    for (size_t s = 0; s < n_seg; s++) {
        size_t start = s * step;
        for (size_t i = 0; i < n_fft; i++) {
            seg[i] = (start + i < N) ? x[start + i] : 0.0;
            seg[i] *= window[i]; /* apply window */
        }
        /* Zero-pad to n_fft and compute FFT */
        for (size_t i = 0; i < n_fft; i++) X[i] = seg[i] + 0.0 * I;
        freqid_fft_radix2(X, n_fft, 0);
        /* Accumulate periodogram (one-sided) */
        double norm = 1.0 / ((double)n_fft * U * fs);
        psd_accum[0] += (creal(X[0])*creal(X[0]) + cimag(X[0])*cimag(X[0])) * norm;
        for (size_t k = 1; k < n_freq - 1; k++)
            psd_accum[k] += 2.0 * (creal(X[k])*creal(X[k]) + cimag(X[k])*cimag(X[k])) * norm;
        if (n_fft % 2 == 0)
            psd_accum[n_freq-1] += (creal(X[n_freq-1])*creal(X[n_freq-1]) + cimag(X[n_freq-1])*cimag(X[n_freq-1])) * norm;
    }

    /* Average and output frequency axis */
    double *freq = (double *)malloc(n_freq * sizeof(double));
    double *psd  = (double *)malloc(n_freq * sizeof(double));
    if (!freq || !psd) { free(psd_accum); free(window); free(seg); free(X); free(freq); free(psd); return -1; }
    for (size_t k = 0; k < n_freq; k++) {
        freq[k] = (double)k * fs / (double)n_fft;
        psd[k] = psd_accum[k] / (double)n_seg;
    }

    free(psd_accum); free(window); free(seg); free(X);
    *freq_out = freq; *psd_out = psd; *n_freq_out = n_freq;
    return 0;
}

/* ================================================================
 * Cross Power Spectral Density ！ Welch Method (L2/L3)
 * ================================================================
 * S_xy(f) = (1/K) * sum X_i*(f) * Y_i(f) / (fs * N * U)
 */

int freqid_cpsd_welch(const double *x, const double *y, size_t N, double fs,
                      size_t n_fft, double overlap,
                      freqid_window_type win_type,
                      double **freq_out, freqid_complex **cpsd_out,
                      size_t *n_freq_out) {
    if (!x || !y || N < n_fft || n_fft < 2 || fs <= 0.0 || !cpsd_out)
        return -1;
    if (overlap < 0.0 || overlap >= 1.0) overlap = 0.5;

    double *window = freqid_window_generate(win_type, n_fft, 3.0);
    if (!window) return -1;

    double U = 0.0;
    for (size_t i = 0; i < n_fft; i++) U += window[i] * window[i];
    U /= (double)n_fft;
    if (U < 1e-15) { free(window); return -1; }

    size_t step = (size_t)((double)n_fft * (1.0 - overlap));
    if (step == 0) step = 1;
    size_t n_seg = (N - n_fft) / step + 1;
    size_t n_freq = n_fft / 2 + 1;

    freqid_complex *cpsd_accum = (freqid_complex *)calloc(n_freq, sizeof(freqid_complex));
    double *seg_x = (double *)malloc(n_fft * sizeof(double));
    double *seg_y = (double *)malloc(n_fft * sizeof(double));
    freqid_complex *X = (freqid_complex *)malloc(n_fft * sizeof(freqid_complex));
    freqid_complex *Y = (freqid_complex *)malloc(n_fft * sizeof(freqid_complex));
    if (!cpsd_accum || !seg_x || !seg_y || !X || !Y) {
        free(cpsd_accum); free(seg_x); free(seg_y); free(X); free(Y);
        free(window); return -1;
    }

    for (size_t s = 0; s < n_seg; s++) {
        size_t start = s * step;
        for (size_t i = 0; i < n_fft; i++) {
            seg_x[i] = (start + i < N) ? x[start + i] * window[i] : 0.0;
            seg_y[i] = (start + i < N) ? y[start + i] * window[i] : 0.0;
        }
        for (size_t i = 0; i < n_fft; i++) { X[i] = seg_x[i] + 0.0*I; Y[i] = seg_y[i] + 0.0*I; }
        freqid_fft_radix2(X, n_fft, 0);
        freqid_fft_radix2(Y, n_fft, 0);
        double norm = 1.0 / ((double)n_fft * U * fs);
        for (size_t k = 0; k < n_freq; k++)
            cpsd_accum[k] += conj(X[k]) * Y[k] * norm;
        /* Double the one-sided spectrum except DC and Nyquist */
        if (n_fft % 2 == 0 && n_freq > 1) {
            for (size_t k = 1; k < n_freq - 1; k++) cpsd_accum[k] *= 2.0;
        }
    }

    double *freq = (freq_out ? (double *)malloc(n_freq * sizeof(double)) : NULL);
    freqid_complex *cpsd = (freqid_complex *)malloc(n_freq * sizeof(freqid_complex));
    if (!cpsd || (freq_out && !freq)) {
        free(freq); free(cpsd); free(cpsd_accum); free(seg_x); free(seg_y);
        free(X); free(Y); free(window); return -1;
    }
    for (size_t k = 0; k < n_freq; k++) {
        if (freq) freq[k] = (double)k * fs / (double)n_fft;
        cpsd[k] = cpsd_accum[k] / (double)n_seg;
    }

    free(cpsd_accum); free(seg_x); free(seg_y); free(X); free(Y); free(window);
    if (freq_out) *freq_out = freq;
    *cpsd_out = cpsd;
    if (n_freq_out) *n_freq_out = n_freq;
    return 0;
}


/* ================================================================
 * Magnitude-Squared Coherence (L2)
 * ================================================================
 * gamma^2(f) = |S_xy(f)|^2 / (S_xx(f) * S_yy(f))
 * 
 * Coherence measures the fraction of output power linearly related
 * to the input at each frequency.  Values near 1 indicate high-quality
 * FRF estimates; values near 0 indicate noise or nonlinear distortion.
 */

int freqid_coherence(const double *x, const double *y, size_t N, double fs,
                     size_t n_fft, double overlap,
                     freqid_window_type win_type,
                     double **freq_out, double **coh_out,
                     size_t *n_freq_out) {
    if (!x || !y || N < n_fft || n_fft < 2 || fs <= 0.0 || !coh_out) return -1;

    /* Compute S_xx */
    double *f_xx = NULL, *psd_xx = NULL; size_t nf_xx = 0;
    if (freqid_psd_welch(x, N, fs, n_fft, overlap, win_type, &f_xx, &psd_xx, &nf_xx) != 0) return -1;

    /* Compute S_yy */
    double *f_yy = NULL, *psd_yy = NULL; size_t nf_yy = 0;
    if (freqid_psd_welch(y, N, fs, n_fft, overlap, win_type, &f_yy, &psd_yy, &nf_yy) != 0) {
        free(f_xx); free(psd_xx); return -1;
    }

    /* Compute S_xy */
    freqid_complex *cpsd_xy = NULL; size_t nf_xy = 0;
    if (freqid_cpsd_welch(x, y, N, fs, n_fft, overlap, win_type, NULL, &cpsd_xy, &nf_xy) != 0) {
        free(f_xx); free(psd_xx); free(f_yy); free(psd_yy); return -1;
    }

    size_t n_freq = nf_xx < nf_yy ? nf_xx : nf_yy;
    if (n_freq > nf_xy) n_freq = nf_xy;

    double *freq = (double *)malloc(n_freq * sizeof(double));
    double *coh  = (double *)malloc(n_freq * sizeof(double));
    if (!freq || !coh) { free(f_xx); free(psd_xx); free(f_yy); free(psd_yy); free(cpsd_xy); free(freq); free(coh); return -1; }

    for (size_t k = 0; k < n_freq; k++) {
        freq[k] = f_xx[k];
        double cpsd_mag2 = creal(cpsd_xy[k])*creal(cpsd_xy[k]) + cimag(cpsd_xy[k])*cimag(cpsd_xy[k]);
        double denom = psd_xx[k] * psd_yy[k];
        coh[k] = (denom > 1e-30) ? (cpsd_mag2 / denom) : 0.0;
        /* Clamp to [0,1] to handle numerical errors */
        if (coh[k] < 0.0) coh[k] = 0.0;
        if (coh[k] > 1.0) coh[k] = 1.0;
    }

    free(f_xx); free(psd_xx); free(f_yy); free(psd_yy); free(cpsd_xy);
    if (freq_out) *freq_out = freq; else free(freq);
    *coh_out = coh;
    if (n_freq_out) *n_freq_out = n_freq;
    return 0;
}

/* ================================================================
 * Autocorrelation Estimation (L2)
 * ================================================================
 * r_xx[tau] = (1/N) * sum_{n=0}^{N-1-tau} x[n] * x[n+tau]
 * Unbiased: divide by (N-tau); Biased: divide by N (used here for positive-definiteness)
 */

int freqid_autocorr(const double *x, size_t N, size_t max_lag, double **r_out) {
    if (!x || N == 0 || max_lag >= N || !r_out) return -1;
    double *r = (double *)malloc((max_lag + 1) * sizeof(double));
    if (!r) return -1;
    double mean = 0.0;
    for (size_t i = 0; i < N; i++) mean += x[i];
    mean /= (double)N;
    for (size_t tau = 0; tau <= max_lag; tau++) {
        double sum = 0.0;
        for (size_t n = 0; n < N - tau; n++)
            sum += (x[n] - mean) * (x[n + tau] - mean);
        r[tau] = sum / (double)N;  /* biased estimate */
    }
    *r_out = r;
    return 0;
}

/* ================================================================
 * Blackman-Tukey Correlogram PSD (L5)
 * ================================================================
 * 1. Estimate autocorrelation r_xx[tau]
 * 2. Window autocorrelation with triangular (Bartlett) lag window
 * 3. DFT of windowed autocorrelation = PSD
 *
 * Ref: Blackman & Tukey (1958), "The Measurement of Power Spectra"
 */

int freqid_psd_blackman_tukey(const double *x, size_t N,
                              size_t max_lag, double fs,
                              double **freq_out, double **psd_out,
                              size_t *n_freq_out) {
    if (!x || N == 0 || max_lag >= N || max_lag == 0 || fs <= 0.0 || !psd_out) return -1;

    /* Step 1: autocorrelation */
    double *r = NULL;
    if (freqid_autocorr(x, N, max_lag, &r) != 0) return -1;

    /* Step 2: form even-length symmetric sequence for DFT
       r_sym = [r[0], r[1], ..., r[max_lag-1], r[max_lag], r[max_lag-1], ..., r[1]]
       Length = 2 * max_lag
       This ensures the PSD is real and non-negative. */
    size_t M = 2 * max_lag;
    freqid_complex *R = (freqid_complex *)calloc(M, sizeof(freqid_complex));
    if (!R) { free(r); return -1; }
    /* triangular (Bartlett) lag window: w[tau] = 1 - tau/max_lag */
    R[0] = r[0] * 1.0 + 0.0 * I;
    for (size_t tau = 1; tau <= max_lag; tau++) {
        double w_lag = 1.0 - (double)tau / (double)max_lag; /* Bartlett window */
        freqid_complex val = r[tau] * w_lag + 0.0 * I;
        R[tau] = val;
        R[M - tau] = val; /* even symmetry */
    }

    /* Step 3: DFT of windowed autocorrelation */
    freqid_fft_radix2(R, M, 0);

    /* Output one-sided PSD */
    size_t n_freq = M / 2 + 1;
    double *freq = (double *)malloc(n_freq * sizeof(double));
    double *psd  = (double *)malloc(n_freq * sizeof(double));
    if (!freq || !psd) { free(r); free(R); free(freq); free(psd); return -1; }
    for (size_t k = 0; k < n_freq; k++) {
        freq[k] = (double)k * fs / (double)M;
        psd[k] = creal(R[k]) / fs;
    }

    free(r); free(R);
    *freq_out = freq; *psd_out = psd;
    if (n_freq_out) *n_freq_out = n_freq;
    return 0;
}
