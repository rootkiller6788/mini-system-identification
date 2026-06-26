#include "freqid_defs.h"
#include "freqid_spectrum.h"
#include "freqid_identify.h"
#include "freqid_app.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/*
 * Example 3: DC Motor Identification from Swept-Sine Data
 *
 * Demonstrates the complete workflow:
 *   1. Generate swept-sine excitation (logarithmic chirp)
 *   2. Simulate DC motor response
 *   3. Compute H1 FRF estimate
 *   4. Fit parametric transfer function
 *   5. Compute model validation metrics
 *
 * DC motor model: G(s) = K / (tau*s + 1)
 * Real-world reference: Maxon RE25, K=0.052 Nm/A, tau=0.017s
 * Used in NASA Mars rover actuation and Boeing UAV servo systems.
 */

int main(void) {
    printf("=== Example 3: DC Motor Frequency-Domain ID ===\n\n");

    /* Swept-sine parameters */
    double f_start = 0.1;       /* [Hz] */
    double f_end = 50.0;        /* [Hz] */
    double duration = 20.0;     /* [seconds] */
    double amplitude = 1.0;     /* [V] */
    double fs = 500.0;          /* sampling frequency [Hz] */

    /* Generate swept sine */
    double *t = NULL, *u = NULL; size_t N = 0;
    if (freqid_app_generate_swept_sine(f_start, f_end, duration, fs,
                                         amplitude, &t, &u, &N) != 0) {
        printf("Signal generation failed\n"); return 1;
    }

    /* DC motor parameters (true values) */
    double K_true = 52.0;       /* gain [rad/s/V] */
    double tau_true = 0.017;    /* time constant [s] */

    /* Simulate motor response via Euler integration */
    double *y = (double *)malloc(N * sizeof(double));
    if (!y) { free(t); free(u); return 1; }
    double dt = 1.0 / fs;
    double omega = 0.0;  /* motor speed [rad/s] */
    for (size_t i = 0; i < N; i++) {
        y[i] = omega;
        double domega = (-omega + K_true * u[i]) / tau_true;
        omega += domega * dt;
    }

    /* Identify model using application helper */
    freqid_transfer_function *tf = freqid_app_identify_dc_motor(t, u, y, N, fs);
    if (!tf) {
        printf("Identification failed\n"); free(t); free(u); free(y); return 1;
    }

    /* Extract parameters: G(s) = b0 / (a1*s + a0) = (b0/a0) / ((a1/a0)*s + 1) */
    double K_est = tf->num[0] / tf->den[0];
    double tau_est = tf->den[1] / tf->den[0];

    printf("DC Motor Identification Results:\n");
    printf("  Parameter    True       Estimated   Error\n");
    printf("  Gain K       %.4f     %.4f     %.2f%%\n",
           K_true, K_est, fabs(K_est-K_true)/K_true*100.0);
    printf("  Time const   %.4f     %.4f     %.2f%%\n",
           tau_true, tau_est, fabs(tau_est-tau_true)/tau_true*100.0);

    /* Validate with FRF comparison */
    freqid_frf *frf_meas = freqid_frf_h1_estimator(u, y, N, fs, 256, 0.5);
    if (frf_meas) {
        freqid_freq_vector fv;
        freqid_freq_vector_log(&fv, 0.1*2*M_PI, 100*2*M_PI, 100);
        freqid_frf *frf_fit = freqid_tf_eval_frf(tf, &fv);
        if (frf_fit) {
            double fit_pct = freqid_fit_percent(frf_meas, frf_fit);
            printf("  FIT %%        --         %.2f%%\n", fit_pct);
            freqid_frf_free(frf_fit);
        }
        freqid_freq_vector_free(&fv);
        freqid_frf_free(frf_meas);
    }

    /* Cleanup */
    freqid_tf_free(tf);
    free(t); free(u); free(y);

    printf("\nExample 3 complete.\n");
    return 0;
}
