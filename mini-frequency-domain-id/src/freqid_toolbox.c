#include "freqid_defs.h"
#include "freqid_spectrum.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ================================================================
 * Frequency Response from Impulse Response (L4: Duality)
 * ================================================================
 * H(jw) = FFT{h(t)} ¡ª the FRF is the Fourier transform of the
 * impulse response.  This is the time-frequency duality cornerstone.
 */

freqid_frf *freqid_frf_from_impulse(const double *h, size_t N, double Ts) {
    if (!h || N == 0 || Ts <= 0.0) return NULL;
    freqid_complex *H_dft = NULL;
    if (freqid_dft_real(h, N, &H_dft) != 0) return NULL;
    freqid_frf *frf = (freqid_frf *)calloc(1, sizeof(freqid_frf));
    if (!frf) { free(H_dft); return NULL; }
    size_t n_freq = N/2 + 1;
    frf->freq.n = n_freq;
    frf->freq.w = (double *)malloc(n_freq * sizeof(double));
    frf->points = (freqid_frf_point *)calloc(n_freq, sizeof(freqid_frf_point));
    if (!frf->freq.w || !frf->points) { freqid_frf_free(frf); free(H_dft); return NULL; }
    double fs = 1.0/Ts;
    for (size_t k = 0; k < n_freq; k++) {
        double f_hz = (double)k * fs / (double)N;
        frf->freq.w[k] = 2.0*M_PI*f_hz;
        frf->points[k].value = H_dft[k] * Ts;
        frf->points[k].magnitude = cabs(frf->points[k].value);
        frf->points[k].phase_deg = carg(frf->points[k].value) * 180.0/M_PI;
        double m = frf->points[k].magnitude;
        frf->points[k].db = (m > 1e-15) ? 20.0*log10(m) : -300.0;
    }
    frf->freq.w_min = frf->freq.w[0];
    frf->freq.w_max = frf->freq.w[n_freq-1];
    free(H_dft);
    return frf;
}

/* ================================================================
 * Coherence-Based FRF Quality Mask
 * ================================================================
 * Returns 1 for reliable frequency points (coherence > threshold),
 * 0 for unreliable points. Used to select data for parametric fitting.
 */

int freqid_quality_mask(const double *coherence, size_t n, double threshold,
                         int **mask_out) {
    if (!coherence || n == 0 || !mask_out) return -1;
    int *mask = (int *)malloc(n * sizeof(int));
    if (!mask) return -1;
    for (size_t i = 0; i < n; i++) mask[i] = (coherence[i] >= threshold) ? 1 : 0;
    *mask_out = mask;
    return 0;
}

/* ================================================================
 * Interpolate FRF to New Frequency Grid (L5)
 * ================================================================
 * Uses linear interpolation of real and imaginary parts.
 */

freqid_frf *freqid_interpolate_frf(const freqid_frf *frf,
                                    const double *freq_new_hz, size_t n_new) {
    if (!frf || !frf->points || frf->freq.n < 2 || !freq_new_hz || n_new == 0) return NULL;
    freqid_frf *out = (freqid_frf *)calloc(1, sizeof(freqid_frf));
    if (!out) return NULL;
    out->freq.n = n_new;
    out->freq.w = (double *)malloc(n_new * sizeof(double));
    out->points = (freqid_frf_point *)calloc(n_new, sizeof(freqid_frf_point));
    if (!out->freq.w || !out->points) { freqid_frf_free(out); return NULL; }
    size_t j = 0;
    for (size_t i = 0; i < n_new; i++) {
        double f_target = freq_new_hz[i];
        double w_target = 2.0*M_PI*f_target;
        out->freq.w[i] = w_target;
        /* Find bounding points in original FRF */
        while (j < frf->freq.n - 1 && frf->freq.w[j+1] < w_target) j++;
        if (j >= frf->freq.n - 1) {
            out->points[i].value = frf->points[frf->freq.n - 1].value;
        } else {
            double w1 = frf->freq.w[j], w2 = frf->freq.w[j+1];
            double alpha = (w2 > w1) ? (w_target - w1)/(w2 - w1) : 0.0;
            if (alpha < 0.0) alpha = 0.0; if (alpha > 1.0) alpha = 1.0;
            freqid_complex v1 = frf->points[j].value;
            freqid_complex v2 = frf->points[j+1].value;
            out->points[i].value = v1 + alpha*(v2 - v1);
        }
        out->points[i].magnitude = cabs(out->points[i].value);
        out->points[i].phase_deg = carg(out->points[i].value) * 180.0/M_PI;
        double m = out->points[i].magnitude;
        out->points[i].db = (m > 1e-15) ? 20.0*log10(m) : -300.0;
    }
    out->freq.w_min = out->freq.w[0];
    out->freq.w_max = out->freq.w[n_new-1];
    return out;
}

/* ================================================================
 * Bode Plot Utility: Find Gain Crossover Frequency
 * ================================================================
 * Returns the frequency where |G(jw)| = 1 (0 dB).
 */

double freqid_find_gain_crossover(const freqid_frf *frf) {
    if (!frf || !frf->points || frf->freq.n < 2) return 0.0;
    for (size_t i = 1; i < frf->freq.n; i++) {
        if (frf->points[i].db <= 0.0 && frf->points[i-1].db >= 0.0) {
            double f1 = frf->freq.w[i-1]/(2.0*M_PI), f2 = frf->freq.w[i]/(2.0*M_PI);
            double m1 = frf->points[i-1].db, m2 = frf->points[i].db;
            if (fabs(m2-m1) > 1e-10) return f1 + (f2-f1)*(0.0-m1)/(m2-m1);
            return f2;
        }
    }
    return 0.0;
}

/* ================================================================
 * Compute system delay (time delay) from FRF phase slope
 * ================================================================
 * tau = -d(phase)/dw, estimated via linear regression on unwrapped phase
 */

double freqid_estimate_delay(const freqid_frf *frf, double freq_low, double freq_high) {
    if (!frf || !frf->points || frf->freq.n < 3) return 0.0;
    /* Unwrap phase and do linear regression */
    double sum_w = 0.0, sum_p = 0.0, sum_wp = 0.0, sum_ww = 0.0;
    size_t count = 0;
    double phase_prev = frf->points[0].phase_deg;
    double unwrap_offset = 0.0;
    for (size_t i = 0; i < frf->freq.n; i++) {
        double f = frf->freq.w[i] / (2.0*M_PI);
        if (f < freq_low || f > freq_high) continue;
        double phase = frf->points[i].phase_deg;
        /* Unwrap: detect jumps > 180 degrees */
        while (phase - phase_prev > 180.0) { unwrap_offset -= 360.0; phase_prev += 360.0; }
        while (phase_prev - phase > 180.0) { unwrap_offset += 360.0; phase_prev -= 360.0; }
        double phase_unwrapped = phase + unwrap_offset;
        phase_prev = phase;
        double w = frf->freq.w[i];
        sum_w += w; sum_p += phase_unwrapped; sum_wp += w*phase_unwrapped; sum_ww += w*w;
        count++;
    }
    if (count < 3) return 0.0;
    double denom = (double)count*sum_ww - sum_w*sum_w;
    if (fabs(denom) < 1e-15) return 0.0;
    double slope = ((double)count*sum_wp - sum_w*sum_p) / denom;
    return -slope * M_PI / 180.0; /* convert deg/rad/s to seconds */
}
