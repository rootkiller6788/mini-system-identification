#include "subspace_core.h"
#include "subspace_algorithms.h"
#include "subspace_validation.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* ============================================================================
 * Example: N4SID Subspace Identification
 *
 * Demonstrates the full N4SID pipeline:
 *   1. Generate data from a known 2nd-order system
 *   2. Apply N4SID to identify a state-space model
 *   3. Validate the identified model
 *
 * The true system is a damped oscillator:
 *   x(k+1) = [0.7  0.2] x(k) + [0.5] u(k)
 *            [-0.2 0.7]          [0.3]
 *   y(k)   = [1.0  0.5] x(k) + 0.1 u(k) + noise
 * ============================================================================ */

int main(void) {
    printf("============================================================\n");
    printf("  N4SID Subspace Identification Example\n");
    printf("============================================================\n\n");

    /* Parameters */
    int N = 1000;
    int n_inputs = 1, n_outputs = 1;
    SubspaceData *data = subspace_data_alloc(N, n_inputs, n_outputs);
    if (!data) { printf("Failed to allocate data\n"); return 1; }
    data->Ts = 0.1;
    data->name = "N4SID Example Data";

    /* Generate excitation input: pseudo-random binary sequence */
    srand(12345);
    printf("Generating PRBS excitation signal (%d samples)...\n", N);
    for (int k = 0; k < N; k++) {
        data->u[k] = ((rand() % 2) == 0) ? -1.0 : 1.0;
    }

    /* Simulate true 2nd-order system */
    double A[4] = {0.7, 0.2, -0.2, 0.7};
    double B[2] = {0.5, 0.3};
    double C[2] = {1.0, 0.5};
    double D = 0.1;
    double x[2] = {0.0, 0.0};

    printf("Simulating true 2nd-order system...\n");
    for (int k = 0; k < N; k++) {
        /* Output */
        data->y[k] = C[0] * x[0] + C[1] * x[1] + D * data->u[k]
                     + 0.01 * ((double)rand() / RAND_MAX - 0.5);
        /* State update */
        double xn0 = A[0] * x[0] + A[1] * x[1] + B[0] * data->u[k];
        double xn1 = A[2] * x[0] + A[3] * x[1] + B[1] * data->u[k];
        x[0] = xn0; x[1] = xn1;
    }
    printf("True eigenvalues: %.4f +/- %.4fj (|lambda| = %.4f)\n\n",
           0.7, 0.2, sqrt(0.7*0.7 + 0.2*0.2));

    /* Configure N4SID */
    SubspaceOptions opts = subspace_options_default();
    opts.i = 15;
    opts.max_order = 8;
    opts.algorithm = SS_N4SID;
    opts.weighting = SS_WGT_N4SID;
    opts.order_crit = SS_ORDER_SVD_GAP;
    opts.sv_threshold = 0.01;
    opts.verbose = true;

    /* Run identification */
    printf("Running N4SID identification...\n");
    SubspaceResult *result = subspace_result_alloc();
    int ret = subspace_n4sid(data, &opts, result);

    if (ret == 0 && result->model) {
        printf("\n--- Identification Results ---\n");
        subspace_print_result(result);

        if (result->singular_values && result->sv_count > 0) {
            subspace_order_selection_plot(result->singular_values,
                result->sv_count < 15 ? result->sv_count : 15);
        }

        /* Validate model */
        printf("\n--- Model Validation ---\n");
        subspace_validation_report(result->model, data);

        /* Compare identified model to true model */
        printf("\n--- Parameter Comparison ---\n");
        printf("Identified A matrix:\n");
        for (int i = 0; i < result->order_estimated; i++) {
            printf("  ");
            for (int j = 0; j < result->order_estimated; j++) {
                printf("% .4f ", result->model->A[(size_t)i *
                       (size_t)result->order_estimated + (size_t)j]);
            }
            printf("\n");
        }
    } else {
        printf("Identification failed with code %d\n", ret);
    }

    subspace_result_free(result);
    subspace_data_free(data);
    printf("\nExample complete.\n");
    return 0;
}
