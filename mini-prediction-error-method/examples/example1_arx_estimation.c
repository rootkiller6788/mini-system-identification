#include "pem_core.h"
#include "pem_model.h"
#include "pem_validation.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/* Example 1: ARX Model Estimation
 *
 * Demonstrates the prediction error method for ARX model identification.
 * Generates data from a known ARX system, then estimates the parameters
 * using both closed-form least squares and iterative PEM.
 *
 * True system: y(t) = 0.8*y(t-1) - 0.3*y(t-2) + 0.5*u(t-1) + e(t)
 * ARX notation: A(q) = 1 + a_1 q^{-1} + a_2 q^{-2}
 *               y(t) = -a_1 y(t-1) - a_2 y(t-2) + b_1 u(t-1) + e(t)
 * So: a_1 = -0.8, a_2 = 0.3, b_1 = 0.5
 */

int main(void) {
    printf("=== Example 1: ARX Model Estimation ===\n\n");

    int N = 500;
    double a1_true = -0.8, a2_true = 0.3, b1_true = 0.5;

    /* Generate data */
    PEMData *data = pem_data_alloc(N);
    data->y[0] = 0.0;
    data->y[1] = 0.0;
    srand((unsigned)time(NULL));

    for (int t = 0; t < N; t++) {
        /* PRBS-like input */
        data->u[t] = (rand() % 2 == 0) ? 1.0 : -1.0;
        if (t >= 2) {
            double noise = 0.02 * ((double)rand() / (double)RAND_MAX - 0.5);
            data->y[t] = -a1_true * data->y[t-1] - a2_true * data->y[t-2]
                        + b1_true * data->u[t-1] + noise;
        }
    }

    printf("Generated %d data points from ARX(2,1,1)\n", N);
    printf("True parameters: a1=%.3f a2=%.3f b1=%.3f\n\n", a1_true, a2_true, b1_true);

    /* Estimate using closed-form LS */
    PEMResult *result = pem_result_alloc(3);
    PEMOptions opts = pem_options_default();
    opts.verbose = false;
    opts.compute_covariance = true;

    printf("Estimating ARX(2,1,1) via Least Squares...\n");
    int ret = pem_estimate_arx_ls(data, 2, 1, 1, result, &opts);

    if (ret == 0) {
        pem_result_print(result);
        printf("\nParameter errors:\n");
        printf("  a1: true=%.4f  est=%.4f  err=%.4f\n",
               a1_true, result->theta_hat[0], a1_true - result->theta_hat[0]);
        printf("  a2: true=%.4f  est=%.4f  err=%.4f\n",
               a2_true, result->theta_hat[1], a2_true - result->theta_hat[1]);
        printf("  b1: true=%.4f  est=%.4f  err=%.4f\n",
               b1_true, result->theta_hat[2], b1_true - result->theta_hat[2]);

        /* Validate model */
        double *y_hat = (double*)malloc((size_t)N * sizeof(double));
        double *eps = (double*)malloc((size_t)N * sizeof(double));
        pem_predict_arx_batch(result->theta_hat, 2, 1, 1,
                              data->u, data->y, N, y_hat);
        pem_residuals_arx(result->theta_hat, 2, 1, 1,
                          data->u, data->y, N, eps);

        PEMValidation *val = pem_validation_alloc();
        pem_validate_model(data->y, y_hat, eps, data->u, N, 3, val);
        printf("\n");
        pem_validation_print(val);

        free(y_hat); free(eps);
        pem_validation_free(val);
    } else {
        printf("Estimation failed!\n");
    }

    pem_result_free(result);
    pem_data_free(data);
    printf("\nDone.\n");
    return 0;
}