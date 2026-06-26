#include "subspace_core.h"
#include "subspace_algorithms.h"
#include "subspace_validation.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* ============================================================================
 * Example: MOESP Subspace Identification
 *
 * Demonstrates the MOESP method on a 3rd-order system.
 * MOESP uses LQ decomposition to eliminate future input influence,
 * then SVD of the projected output to recover the extended observability
 * matrix Gamma_i which directly yields A and C matrices.
 * ============================================================================ */

int main(void) {
    printf("============================================================\n");
    printf("  MOESP Subspace Identification Example\n");
    printf("============================================================\n\n");

    int N = 800;
    SubspaceData *data = subspace_data_alloc(N, 1, 1);
    if (!data) { printf("Memory error\n"); return 1; }
    data->Ts = 0.05;

    /* Generate PRBS input */
    srand(67890);
    printf("Generating excitation signal...\n");
    for (int k = 0; k < N; k++)
        data->u[k] = ((rand() % 3) - 1);  /* -1, 0, 1 */

    /* True 3rd-order system (band-pass filter):
     * Poles at 0.8, 0.6+/-0.3j
     * Transfer function: H(z) = (z-0.5) / ((z-0.8)(z^2-1.2z+0.45)) */
    double A[9] = { 0.8, 0.0, 0.0,
                    0.0, 0.6, -0.3,
                    0.0, 0.3, 0.6 };
    double B[3] = { 0.2, 0.4, 0.1 };
    double C[3] = { 1.0, 0.8, 0.0 };
    double D = 0.05;
    double x[3] = {0, 0, 0};

    printf("True system: 3rd-order band-pass filter\n");
    printf("Poles: 0.8, 0.6+/-0.3j\n\n");
    printf("Simulating...\n");

    for (int k = 0; k < N; k++) {
        data->y[k] = C[0]*x[0] + C[1]*x[1] + C[2]*x[2] + D * data->u[k]
                     + 0.005 * ((double)rand()/RAND_MAX - 0.5);
        double xn[3];
        for (int i = 0; i < 3; i++) {
            xn[i] = 0.0;
            for (int j = 0; j < 3; j++)
                xn[i] += A[(size_t)i*3 + j] * x[j];
            xn[i] += B[i] * data->u[k];
        }
        x[0] = xn[0]; x[1] = xn[1]; x[2] = xn[2];
    }

    /* Configure MOESP */
    SubspaceOptions opts = subspace_options_default();
    opts.i = 12;
    opts.max_order = 8;
    opts.algorithm = SS_MOESP;
    opts.order_crit = SS_ORDER_SVC;
    opts.sv_threshold = 0.99;

    printf("Running MOESP identification...\n");
    SubspaceResult *result = subspace_result_alloc();
    int ret = subspace_moesp(data, &opts, result);

    if (ret == 0 && result->model) {
        printf("\n--- MOESP Results ---\n");
        subspace_print_result(result);

        if (result->singular_values && result->sv_count > 0) {
            subspace_order_selection_plot(result->singular_values,
                result->sv_count < 12 ? result->sv_count : 12);
        }

        printf("\n--- Validation ---\n");
        subspace_validation_report(result->model, data);

        printf("\nIdentified poles:\n");
        double *re = (double*)malloc((size_t)result->order_estimated * sizeof(double));
        double *im = (double*)malloc((size_t)result->order_estimated * sizeof(double));
        if (re && im) {
            subspace_model_poles(result->model, re, im);
            for (int i = 0; i < result->order_estimated; i++) {
                double mag = sqrt(re[i]*re[i] + im[i]*im[i]);
                printf("  lambda[%d] = % .4f %+.4fj  |lambda| = %.4f\n",
                       i, re[i], im[i], mag);
            }
        }
        free(re); free(im);
    } else {
        printf("MOESP failed (code %d)\n", ret);
    }

    subspace_result_free(result);
    subspace_data_free(data);
    printf("\nExample complete.\n");
    return 0;
}
