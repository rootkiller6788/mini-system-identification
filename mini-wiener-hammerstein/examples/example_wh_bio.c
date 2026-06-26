/**
 * example_wh_bio.c ? Biological Dose-Response System
 *
 * Models a pharmacological dose-response system:
 *   L1: Absorption dynamics (drug delivery ? bloodstream)
 *   N:  Sigmoid dose-response (Hill equation)
 *   L2: Elimination dynamics (bloodstream ? effect compartment)
 *
 * The Wiener-Hammerstein structure naturally captures the cascade:
 * pharmacokinetics (L1) ? pharmacodynamics (N) ? effect kinetics (L2).
 *
 * This is an important application area because biological systems
 * frequently exhibit block-structured nonlinear dynamics.
 *
 * Reference: Gabrielsson, J. & Weiner, D. (2006).
 *   Pharmacokinetic and Pharmacodynamic Data Analysis. 4th ed.
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
    printf("?  Biological Dose-Response ? Wiener-Hammerstein Model    ?\n");
    printf("????????????????????????????????????????????????????????????\n\n");

    /* ??? System: PK ? PD ? Effect ??????????????????????????????????? */
    printf("System: Absorption(L1) ? Sigmoid PD(N) ? Elimination(L2)\n\n");

    WH_Model* bio_sys = wh_model_create();

    /* L1: Absorption ? 2nd-order: H(z) = 0.05 + 0.1z?? / (1 - 1.2z?? + 0.5z??) */
    double L1_b[] = {0.05, 0.1};
    double L1_a[] = {1.0, -1.2, 0.5};
    wh_linear_init_iir(&bio_sys->L1, L1_b, 2, L1_a, 3, 1.0);

    /* N: Sigmoid (Hill-type): f(x) = E_max * x^n / (EC50^n + x^n)
     *   Approximated with sigmoid: f(x) = 100 / (1 + exp(-0.5*(x-10))) */
    wh_nl_init_sigmoid(&bio_sys->N, 100.0, 0.5, 10.0);

    /* L2: Elimination ? 1st-order: H(z) = 0.7 / (1 - 0.3z??) */
    double L2_b[] = {0.7};
    double L2_a[] = {1.0, -0.3};
    wh_linear_init_iir(&bio_sys->L2, L2_b, 1, L2_a, 2, 1.0);

    /* ??? Generate data: stepped dosing ?????????????????????????????? */
    int N = 2000;
    double* u_dose = (double*)malloc(N * sizeof(double));
    double* y_effect = (double*)malloc(N * sizeof(double));

    /* Dose sequence: 5 levels for 400 samples each */
    double dose_levels[] = {2.0, 5.0, 10.0, 20.0, 3.0};
    for (int i = 0; i < N; i++) {
        u_dose[i] = dose_levels[i / 400];
    }

    wh_model_reset(bio_sys);
    for (int i = 0; i < N; i++) {
        y_effect[i] = wh_model_evaluate(bio_sys, u_dose[i]);
    }

    /* Print dose-response summary */
    printf("Dose-Response Data (first 600 samples):\n");
    printf("  t=0-400:  Dose=%.1f ? steady-state effect ? %.1f\n",
           dose_levels[0], y_effect[398]);
    printf("  t=400-800: Dose=%.1f ? steady-state effect ? %.1f\n",
           dose_levels[1], y_effect[798]);

    /* ??? Identify model ????????????????????????????????????????????? */
    printf("\nIdentifying PK/PD model...\n");

    WH_IdentConfig cfg = wh_ident_config_default();
    cfg.method = WH_ID_ITERATIVE;
    cfg.order_L1 = 4;
    cfg.order_L2 = 2;
    cfg.nl_degree = 0; /* Will use approach with NL init */
    cfg.nl_type = WH_NL_SIGMOID;
    cfg.max_iterations = 40;
    cfg.tolerance = 1e-5;

    WH_IdentResult res;
    memset(&res, 0, sizeof(WH_IdentResult));

    if (wh_ident_iterative(u_dose, y_effect, N, &cfg, &res) == 0 && res.model) {
        printf("Identification complete:\n");
        printf("  Iterations: %d\n", res.iterations);
        printf("  Converged: %s\n", res.converged ? "Yes" : "No");
        printf("  Training FIT: %.2f%%\n", res.fit_percent);

        /* Validate on full dataset */
        WH_ValidationReport report;
        memset(&report, 0, sizeof(WH_ValidationReport));
        if (wh_validate_comprehensive(res.model, u_dose, y_effect, N, 5, &report) == 0) {
            wh_validate_report_print(&report);
        }

        /* Print identified model components */
        printf("\nIdentified model:\n");
        wh_model_print(res.model);

        wh_ident_result_free(&res);
    } else {
        printf("Identification failed.\n");
    }

    wh_model_free(bio_sys);
    free(u_dose); free(y_effect);

    printf("\nDone.\n");
    return 0;
}
