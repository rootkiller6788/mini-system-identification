#include "subspace_core.h"
#include "subspace_algorithms.h"
#include "subspace_validation.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* ============================================================================
 * Example: CVA Subspace Identification
 *
 * Demonstrates Canonical Variate Analysis on a 4th-order MIMO system
 * (2 inputs, 2 outputs). CVA maximizes the correlation between past
 * and conditional future, which makes it particularly robust to noise
 * compared to N4SID and MOESP.
 * ============================================================================ */

int main(void) {
    printf("============================================================\n");
    printf("  CVA Subspace Identification Example (MIMO)\n");
    printf("============================================================\n\n");

    int N = 1200;
    int r = 2, m = 2;  /* 2 inputs, 2 outputs */
    SubspaceData *data = subspace_data_alloc(N, r, m);
    if (!data) { printf("Memory error\n"); return 1; }
    data->Ts = 0.1;

    /* True 4th-order MIMO system */
    double A[16] = {
        0.5, 0.3, -0.1, 0.0,
       -0.2, 0.6,  0.1, 0.0,
        0.0, 0.0,  0.7, 0.2,
        0.0, 0.0, -0.1, 0.8
    };
    double B[8] = {
        0.5, 0.1,
        0.0, 0.3,
        0.2, 0.4,
        0.1, 0.0
    };
    double C[8] = {
        1.0, 0.0, 0.5, 0.0,
        0.0, 1.0, 0.0, 0.3
    };
    double D[4] = {
        0.05, 0.0,
        0.0, 0.03
    };
    double x[4] = {0, 0, 0, 0};

    /* Generate excitation */
    srand(99999);
    printf("Generating MIMO excitation...\n");
    for (int k = 0; k < N; k++) {
        data->u[(size_t)k * 2 + 0] = 2.0 * ((double)rand()/RAND_MAX - 0.5);
        data->u[(size_t)k * 2 + 1] = sin(0.02 * (double)k);
    }

    printf("Simulating 4th-order MIMO system...\n");
    for (int k = 0; k < N; k++) {
        /* Output */
        for (int i = 0; i < m; i++) {
            double yi = 0.0;
            for (int j = 0; j < 4; j++)
                yi += C[(size_t)i * 4 + j] * x[j];
            for (int j = 0; j < r; j++)
                yi += D[(size_t)i * r + j] * data->u[(size_t)k * r + j];
            data->y[(size_t)k * m + i] = yi
                + 0.02 * ((double)rand()/RAND_MAX - 0.5);
        }
        /* State update */
        double xn[4] = {0, 0, 0, 0};
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++)
                xn[i] += A[(size_t)i * 4 + j] * x[j];
            for (int j = 0; j < r; j++)
                xn[i] += B[(size_t)i * r + j] * data->u[(size_t)k * r + j];
        }
        for (int i = 0; i < 4; i++) x[i] = xn[i];
    }

    /* Configure CVA */
    SubspaceOptions opts = subspace_options_default();
    opts.i = 15;
    opts.max_order = 10;
    opts.algorithm = SS_CVA;
    opts.weighting = SS_WGT_CVA;
    opts.order_crit = SS_ORDER_SVD_GAP;

    printf("Running CVA identification...\n");
    SubspaceResult *result = subspace_result_alloc();
    int ret = subspace_cva(data, &opts, result);

    if (ret == 0 && result->model) {
        printf("\n--- CVA Results ---\n");
        subspace_print_result(result);

        printf("\n--- Validation ---\n");
        subspace_validation_report(result->model, data);

        /* Compare step responses for first input-output pair */
        printf("\n--- Step Response (u1 -> y1) ---\n");
        double step_true[30], step_id[30];
        SubspaceModel *true_model = subspace_model_alloc(4, r, m);
        if (true_model) {
            memcpy(true_model->A, A, 16 * sizeof(double));
            memcpy(true_model->B, B, 8 * sizeof(double));
            memcpy(true_model->C, C, 8 * sizeof(double));
            memcpy(true_model->D, D, 4 * sizeof(double));

            subspace_model_step_response(true_model, step_true, 30);
            subspace_model_step_response(result->model, step_id, 30);

            printf("  k   True      Identified\n");
            for (int k = 0; k < 15; k++)
                printf("  %2d  % .4f   % .4f\n",
                       k, step_true[k], step_id[k]);

            subspace_model_free(true_model);
        }
    } else {
        printf("CVA failed (code %d)\n", ret);
    }

    subspace_result_free(result);
    subspace_data_free(data);
    printf("\nExample complete.\n");
    return 0;
}
