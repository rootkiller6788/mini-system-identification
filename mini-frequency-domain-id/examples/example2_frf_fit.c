#include "freqid_defs.h"
#include "freqid_identify.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/*
 * Example 2: Parametric Transfer Function Fitting
 *
 * Identifies a second-order mass-spring-damper system from
 * frequency-domain data using nonlinear least-squares.
 *
 * True system: G(s) = 1 / (s^2 + 0.5s + 25)
 *   ¡ú natural frequency ¦Ø_n = 5 rad/s (0.796 Hz)
 *   ¡ú damping ratio ¦Æ = 0.05 (lightly damped)
 *   ¡ú DC gain = 1/25 = 0.04
 */

int main(void) {
    printf("=== Example 2: Parametric TF Fitting to FRF ===\n\n");

    /* Generate "measured" FRF from known system */
    freqid_freq_vector fv;
    freqid_freq_vector_log(&fv, 0.1, 100.0, 200);

    freqid_transfer_function true_tf;
    double num_true[] = {1.0};
    double den_true[] = {25.0, 0.5, 1.0};  /* s^2 + 0.5s + 25 */
    freqid_tf_create(&true_tf, num_true, 0, den_true, 2, 0);

    freqid_frf *frf_true = freqid_tf_eval_frf(&true_tf, &fv);

    /* Fit second-order model using LS */
    freqid_transfer_function *tf_fit = freqid_ls_fit(frf_true, NULL, 0, 2, 0, 100, 1e-6);

    if (!tf_fit) {
        printf("LS fitting failed\n");
        freqid_frf_free(frf_true);
        freqid_tf_free(&true_tf);
        freqid_freq_vector_free(&fv);
        return 1;
    }

    /* Display identified model */
    printf("True model:  G(s) = 1 / (s^2 + 0.5s + 25)\n");
    printf("Identified:  G(s) = %.4f / (%.4f*s^2 + %.4f*s + %.4f)\n\n",
           tf_fit->num[0],
           tf_fit->den[2], tf_fit->den[1], tf_fit->den[0]);

    /* Compute FIT percentage */
    freqid_frf *frf_fit = freqid_tf_eval_frf(tf_fit, &fv);
    double fit = freqid_fit_percent(frf_true, frf_fit);
    printf("FIT percentage: %.2f%%\n", fit);

    /* Compare key characteristics */
    double wn_true = sqrt(25.0);  /* = 5.0 */
    double wn_fit = sqrt(tf_fit->den[0] / tf_fit->den[2]);
    printf("\nNatural frequency: true=%.3f rad/s, fit=%.3f rad/s\n",
           wn_true, wn_fit);

    double zeta_true = 0.5 / (2.0 * 5.0);  /* = 0.05 */
    double zeta_fit = tf_fit->den[1] / (2.0 * sqrt(tf_fit->den[0] * tf_fit->den[2]));
    printf("Damping ratio:    true=%.4f, fit=%.4f\n", zeta_true, zeta_fit);

    /* Cleanup */
    freqid_frf_free(frf_true);
    freqid_frf_free(frf_fit);
    freqid_tf_free(&true_tf);
    freqid_tf_free(tf_fit);
    freqid_freq_vector_free(&fv);

    printf("\nExample 2 complete.\n");
    return 0;
}
