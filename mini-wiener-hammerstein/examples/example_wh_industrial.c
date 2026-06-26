/**
 * example_wh_industrial.c ? Industrial Actuator with Saturation
 *
 * Models a typical industrial actuator system: an electric motor
 * (L1: 2nd-order dynamics), a saturation nonlinearity (voltage limit),
 * and a mechanical load (L2: 1st-order dynamics).
 *
 * Demonstrates WH identification on a system with hard saturation.
 * This is a common scenario in industrial control where actuators
 * have physical limits.
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
#include "wh_validation.h"

int main(void) {
    printf("????????????????????????????????????????????????????????????\n");
    printf("?  Industrial Actuator with Saturation ? WH Model         ?\n");
    printf("????????????????????????????????????????????????????????????\n\n");

    /* ??? System description ????????????????????????????????????????? */
    printf("System: Motor (2nd-order) ? Saturation ?10V ? Load (1st-order)\n\n");

    /* ??? Build true system ?????????????????????????????????????????? */
    WH_Model* sys = wh_model_create();

    /* L1: Motor dynamics ? 2nd-order IIR: H(z) = 0.1/(1 - 1.5z?? + 0.7z??) */
    double L1_b[] = {0.1};
    double L1_a[] = {1.0, -1.5, 0.7};
    wh_linear_init_iir(&sys->L1, L1_b, 1, L1_a, 3, 1.0);

    /* N: Saturation at ?10V */
    wh_nl_init_saturation(&sys->N, 1.0, 10.0);

    /* L2: Load ? 1st-order: H(z) = 0.8/(1 - 0.2z??) */
    double L2_b[] = {0.8};
    double L2_a[] = {1.0, -0.2};
    wh_linear_init_iir(&sys->L2, L2_b, 1, L2_a, 2, 1.0);

    /* ??? Generate data ?????????????????????????????????????????????? */
    int N_train = 3000;
    int N_val = 600;
    double* u_train = (double*)malloc(N_train * sizeof(double));
    double* y_train = (double*)malloc(N_train * sizeof(double));
    double* u_val = (double*)malloc(N_val * sizeof(double));
    double* y_val = (double*)malloc(N_val * sizeof(double));

    /* Training: multisine with varying amplitude to excite saturation */
    wh_signal_multisine(u_train, N_train, 1.0, 0.01, 0.4, 20, 15.0, 42);
    wh_model_reset(sys);
    for (int i = 0; i < N_train; i++) {
        y_train[i] = wh_model_evaluate(sys, u_train[i]);
    }

    /* Validation: Gaussian noise */
    wh_signal_gaussian(u_val, N_val, 0.0, 5.0, 777);
    wh_model_reset(sys);
    for (int i = 0; i < N_val; i++) {
        y_val[i] = wh_model_evaluate(sys, u_val[i]);
    }

    /* ??? Identify ??????????????????????????????????????????????????? */
    printf("Identifying WH model with iterative method...\n");

    WH_IdentConfig cfg = wh_ident_config_default();
    cfg.method = WH_ID_ITERATIVE;
    cfg.order_L1 = 2;
    cfg.order_L2 = 1;
    cfg.nl_degree = 1; /* Start with linear N, iterative will refine */
    cfg.nl_type = WH_NL_SATURATION;
    cfg.max_iterations = 50;
    cfg.tolerance = 1e-5;

    WH_IdentResult res;
    memset(&res, 0, sizeof(WH_IdentResult));

    if (wh_ident_iterative(u_train, y_train, N_train, &cfg, &res) == 0 && res.model) {
        printf("Iterations: %d, Converged: %s\n",
               res.iterations, res.converged ? "Yes" : "No");
        printf("Training FIT: %.2f%%\n", res.fit_percent);

        /* Validate */
        double val_fit = wh_validate_fit(res.model, u_val, y_val, N_val, NULL);
        printf("Validation FIT: %.2f%%\n", val_fit);

        /* Residual analysis */
        WH_ResidualAnalysis ra;
        memset(&ra, 0, sizeof(WH_ResidualAnalysis));
        if (wh_validate_residuals(res.model, u_val, y_val, N_val, 20, NULL, &ra) == 0) {
            printf("Residual whiteness: %s\n", ra.is_white_95 ? "PASS" : "FAIL");
            printf("Input independence: %s\n", ra.is_independent_95 ? "PASS" : "FAIL");
            wh_validate_residuals_free(&ra);
        }

        /* Comprehensive validation */
        WH_ValidationReport report;
        memset(&report, 0, sizeof(WH_ValidationReport));
        if (wh_validate_comprehensive(res.model, u_val, y_val, N_val, 5, &report) == 0) {
            wh_validate_report_print(&report);
        }

        wh_ident_result_free(&res);
    } else {
        printf("Identification failed.\n");
    }

    wh_model_free(sys);
    free(u_train); free(y_train); free(u_val); free(y_val);

    printf("\nDone.\n");
    return 0;
}
