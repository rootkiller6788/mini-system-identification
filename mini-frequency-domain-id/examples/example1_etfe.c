#include "freqid_defs.h"
#include "freqid_identify.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/*
 * Example 1: Empirical Transfer Function Estimate (ETFE)
 *
 * Demonstrates non-parametric frequency-domain identification
 * of a known first-order system from input-output data.
 *
 * System: G(s) = 5 / (s + 10)  ¡ú  bandwidth = 10 rad/s
 * Input: white noise (broadband excitation)
 * Output: simulated via Euler integration
 */

int main(void) {
    printf("=== Example 1: ETFE of First-Order System ===\n\n");

    double fs = 100.0;           /* sampling frequency [Hz] */
    double duration = 10.0;      /* [seconds] */
    size_t N = (size_t)(fs * duration);
    double dt = 1.0 / fs;

    /* Generate white noise input */
    double *u = (double *)malloc(N * sizeof(double));
    double *y = (double *)malloc(N * sizeof(double));
    if (!u || !y) { printf("Allocation failed\n"); return 1; }

    unsigned int seed = 42;
    for (size_t i = 0; i < N; i++) {
        seed = seed * 1103515245 + 12345;
        u[i] = ((double)(seed & 0x7fffffff)) / 0x7fffffff - 0.5;
    }

    /* Simulate first-order system: dy/dt = -10*y + 5*u */
    double y_state = 0.0;
    double a = 10.0, b = 5.0;
    for (size_t i = 0; i < N; i++) {
        y[i] = y_state;
        double dy = -a * y_state + b * u[i];
        y_state += dy * dt;
    }

    /* Compute ETFE */
    freqid_frf *frf = freqid_etfe(u, y, N, fs);
    if (!frf) { printf("ETFE failed\n"); free(u); free(y); return 1; }

    /* Display results at key frequencies */
    printf("Freq [Hz]  |  Mag [dB]  |  Phase [deg]\n");
    printf("-----------+------------+-------------\n");
    double key_freqs[] = {0.1, 0.5, 1.0, 2.0, 5.0, 10.0};
    for (int k = 0; k < 6; k++) {
        double w_target = 2.0 * M_PI * key_freqs[k];
        size_t best_idx = 0;
        double best_diff = 1e308;
        for (size_t i = 0; i < frf->freq.n; i++) {
            double diff = fabs(frf->freq.w[i] - w_target);
            if (diff < best_diff) { best_diff = diff; best_idx = i; }
        }
        printf("%9.2f  | %9.2f  | %10.2f\n",
               key_freqs[k],
               frf->points[best_idx].db,
               frf->points[best_idx].phase_deg);
    }

    /* The true system: G(s)=5/(s+10), at these freqs:
     *  0.1 Hz: |G| = 5/sqrt(0.6283^2+100) ¡Ö 0.5 = -6.0 dB, phase ¡Ö -3.6¡ã
     *  1.0 Hz: |G| = 5/sqrt(6.283^2+100) ¡Ö 0.425 = -7.4 dB, phase ¡Ö -32.1¡ã
     *  10 Hz:  |G| = 5/sqrt(62.83^2+100) ¡Ö 0.079 = -22.0 dB, phase ¡Ö -81¡ã
     */

    printf("\nTrue system G(s) = 5/(s+10):\n");
    printf("DC gain = %.2f (-14.0 dB), Bandwidth = 10 rad/s (1.59 Hz)\n",
           5.0/10.0);

    freqid_frf_free(frf);
    free(u); free(y);
    printf("\nExample 1 complete.\n");
    return 0;
}
