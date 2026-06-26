/**
 * example_wh_benchmark.c ? Wiener-Hammerstein Benchmark Problem
 *
 * Recreates the classic WH benchmark from Schoukens et al. (2015):
 *   L1: 2nd-order Chebyshev low-pass filter (cutoff 0.5*fs)
 *   N:  f(x) = x + 0.1*x^3 (mild cubic nonlinearity)
 *   L2: 2nd-order Butterworth low-pass filter
 *
 * Demonstrates identification using the iterative method and
 * reports FIT metrics on training and test data.
 *
 * Reference:
 *   Schoukens, J. et al. (2015). "Identification of Wiener-Hammerstein
 *   Systems." Automatica, 52, 1-11.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "wh_model.h"
#include "wh_linear.h"
#include "wh_nonlinear.h"
#include "wh_identification.h"
#include "wh_simulation.h"
#include "wh_signal.h"

int main(void) {
    printf("????????????????????????????????????????????????????????????\n");
    printf("?  Wiener-Hammerstein Benchmark Example                   ?\n");
    printf("?  Schoukens et al. (2015) ? Automatica Benchmark         ?\n");
    printf("????????????????????????????????????????????????????????????\n\n");

    /* ??? Step 1: Create "true" system ???????????????????????????????? */
    WH_Model* true_sys = wh_model_create();

    /* L1: 5-tap FIR: [1.0, 0.8, 0.4, 0.2, 0.1] ? normalized low-pass */
    double b1[] = {1.0, 0.8, 0.4, 0.2, 0.1};
    wh_linear_init_fir(&true_sys->L1, b1, 5, 1.0);

    /* N: f(x) = 0.1*x + 0.9*x^3 (strong cubic nonlinearity) */
    double nl_coeffs[] = {0.0, 0.1, 0.0, 0.9};
    wh_nl_init_polynomial(&true_sys->N, nl_coeffs, 3);

    /* L2: 3-tap FIR: [0.5, 0.3, 0.2] */
    double b2[] = {0.5, 0.3, 0.2};
    wh_linear_init_fir(&true_sys->L2, b2, 3, 1.0);

    printf("True system created:\n");
    printf("  L1: 5-tap FIR low-pass\n");
    printf("  N:  f(x) = 0.1*x + 0.9*x^3\n");
    printf("  L2: 3-tap FIR\n\n");

    /* ??? Step 2: Generate excitation and response ???????????????????? */
    int N = 2000;
    int N_test = 500;
    double* u_train = (double*)malloc(N * sizeof(double));
    double* y_train = (double*)malloc(N * sizeof(double));
    double* u_test = (double*)malloc(N_test * sizeof(double));
    double* y_test = (double*)malloc(N_test * sizeof(double));

    if (!u_train || !y_train || !u_test || !y_test) {
        printf("Memory allocation failed.\n");
        return 1;
    }

    /* Generate PRBS excitation for training */
    printf("Generating PRBS excitation (N=%d)...\n", N);
    wh_signal_prbs(u_train, N, 1.0, 10, 42);

    /* Simulate true system */
    printf("Simulating true system...\n");
    wh_model_reset(true_sys);
    for (int i = 0; i < N; i++) {
        y_train[i] = wh_model_evaluate(true_sys, u_train[i]);
    }

    /* Generate Gaussian excitation for testing */
    wh_signal_gaussian(u_test, N_test, 0.0, 1.0, 123);

    wh_model_reset(true_sys);
    for (int i = 0; i < N_test; i++) {
        y_test[i] = wh_model_evaluate(true_sys, u_test[i]);
    }

    /* ??? Step 3: Identify WH model ?????????????????????????????????? */
    printf("\n--- Identification ---\n");

    WH_IdentConfig cfg = wh_ident_config_default();
    cfg.method = WH_ID_ITERATIVE;
    cfg.order_L1 = 5;
    cfg.order_L2 = 3;
    cfg.nl_degree = 3;
    cfg.max_iterations = 30;
    cfg.tolerance = 1e-6;
    cfg.verbosity = 1;

    WH_IdentResult ident_result;
    memset(&ident_result, 0, sizeof(WH_IdentResult));

    int ret = wh_ident_iterative(u_train, y_train, N, &cfg, &ident_result);

    if (ret == 0 && ident_result.model) {
        printf("\nIdentification results:\n");
        printf("  Iterations: %d\n", ident_result.iterations);
        printf("  Converged: %s\n", ident_result.converged ? "Yes" : "No");
        printf("  Training FIT: %.2f%%\n", ident_result.fit_percent);
        printf("  Training MSE: %.6f\n", ident_result.final_loss);
        printf("  AIC: %.3f, BIC: %.3f\n", ident_result.aic, ident_result.bic);
        printf("  N params: %d\n", ident_result.n_parameters);
        printf("  Stability: %s\n",
               wh_model_is_stable(ident_result.model) ? "Stable" : "Unstable");

        /* Print identified model */
        printf("\n--- Identified Model ---\n");
        wh_model_print(ident_result.model);

        /* ??? Step 4: Validate on test data ????????????????????????? */
        printf("\n--- Validation on Test Data ---\n");

        WH_SimConfig sim_cfg = wh_sim_config_default();
        sim_cfg.n_transient = 50;

        WH_SimOutput sim_out;
        memset(&sim_out, 0, sizeof(WH_SimOutput));

        if (wh_sim_run_with_reference(ident_result.model,
                                       u_test, y_test, N_test,
                                       &sim_cfg, &sim_out) == 0) {
            printf("  Test FIT: %.2f%%\n", sim_out.fit_percent);
            printf("  Test MSE: %.6f\n", sim_out.mse);
            printf("  Test RMSE: %.6f\n", wh_sim_compute_rmse(y_test, sim_out.y, sim_out.n_samples));

            if (sim_out.fit_percent > 70.0) {
                printf("  VERDICT: Good identification ?\n");
            } else if (sim_out.fit_percent > 40.0) {
                printf("  VERDICT: Moderate identification ??\n");
            } else {
                printf("  VERDICT: Poor identification ?\n");
            }
            wh_sim_output_free(&sim_out);
        }

        wh_ident_result_free(&ident_result);
    } else {
        printf("Identification failed (ret=%d).\n", ret);
    }

    /* Cleanup */
    wh_model_free(true_sys);
    free(u_train); free(y_train); free(u_test); free(y_test);

    printf("\nDone.\n");
    return 0;
}
