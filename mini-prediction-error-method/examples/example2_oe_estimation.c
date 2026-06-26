#include "pem_core.h"
#include "pem_model.h"
#include "pem_validation.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/* Example 2: Output Error Model Estimation
 *
 * OE models are suitable when measurement noise is white and uncorrelated
 * with the input. Unlike ARX, the OE predictor uses only past inputs
 * (not past outputs), making the criterion non-convex.
 *
 * We demonstrate OE estimation with Levenberg-Marquardt optimization.
 *
 * True system: w(t) = B(q)/F(q) u(t-1) + e(t)
 *   B(q) = 1.0
 *   F(q) = 1 - 0.6 q^{-1} + 0.3 q^{-2}
 *   Parameters: b_1=1.0, f_1=-0.6, f_2=0.3
 */

int main(void) {
    printf("=== Example 2: Output Error Model Estimation ===\n\n");

    int N = 400;
    double b1_true = 1.0, f1_true = -0.6, f2_true = 0.3;

    PEMData *data = pem_data_alloc(N);
    srand((unsigned)time(NULL));

    double w = 0.0, w_prev = 0.0;
    for (int t = 0; t < N; t++) {
        data->u[t] = (t % 25 < 13) ? 1.0 : -1.0;
        double u_prev = (t > 0) ? data->u[t-1] : 0.0;
        double w_prev2 = (t > 1) ? w_prev : 0.0;
        double noise = 0.05 * ((double)rand() / (double)RAND_MAX - 0.5);
        w_prev = w;
        w = b1_true * u_prev - f1_true * w_prev - f2_true * w_prev2 + noise;
        data->y[t] = w;
    }

    printf("Generated %d data points from OE(1,2,1)\n", N);
    printf("True: b1=%.2f f1=%.2f f2=%.2f\n\n", b1_true, f1_true, f2_true);

    PEMResult *result = pem_result_alloc(3);
    PEMOptions opts = pem_options_default();
    opts.verbose = true;
    opts.max_iterations = 30;

    printf("Estimating OE(1,2,1) via Levenberg-Marquardt...\n");
    int ret = pem_estimate_oe(data, 1, 2, 1, NULL, result, &opts);

    if (ret == 0) {
        pem_result_print(result);
        printf("\nTrue vs Estimated:\n");
        printf("  b1: %.4f vs %.4f\n", b1_true, result->theta_hat[0]);
        printf("  f1: %.4f vs %.4f\n", f1_true, result->theta_hat[1]);
        printf("  f2: %.4f vs %.4f\n", f2_true, result->theta_hat[2]);

        /* Validate */
        double *y_hat = (double*)malloc((size_t)N * sizeof(double));
        pem_predict_oe_batch(result->theta_hat, 1, 2, 1, data->u, N, y_hat);
        double fit = pem_nrmse_fit(data->y, y_hat, N);
        printf("\nModel fit: %.2f%%\n", fit);
        free(y_hat);
    } else {
        printf("Estimation may not have fully converged.\n");
        pem_result_print(result);
    }

    pem_result_free(result);
    pem_data_free(data);
    printf("\nDone.\n");
    return 0;
}