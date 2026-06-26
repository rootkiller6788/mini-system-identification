#include "pem_core.h"
#include "pem_model.h"
#include "pem_validation.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/* Example 3: ARMAX Model Estimation
 *
 * ARMAX extends ARX with a moving-average noise model C(q)e(t).
 * This captures correlated noise structures often found in practice.
 *
 * True system: A(q)y(t) = B(q)u(t-1) + C(q)e(t)
 *   A(q) = 1 - 0.5 q^{-1}   => a_1 = -0.5
 *   B(q) = 0.7              => b_1 = 0.7
 *   C(q) = 1 + 0.3 q^{-1}   => c_1 = 0.3
 */

int main(void) {
    printf("=== Example 3: ARMAX Model Estimation ===\n\n");

    int N = 500;
    double a1_true = -0.5, b1_true = 0.7, c1_true = 0.3;

    PEMData *data = pem_data_alloc(N);
    srand(42);  /* Fixed seed for reproducibility */

    double eps_prev = 0.0;
    for (int t = 0; t < N; t++) {
        data->u[t] = (t % 15 < 8) ? 1.0 : -1.0;
        double eps = 0.1 * ((double)rand() / (double)RAND_MAX - 0.5);
        double y_prev = (t > 0) ? data->y[t-1] : 0.0;
        double u_prev = (t > 0) ? data->u[t-1] : 0.0;
        data->y[t] = -a1_true * y_prev + b1_true * u_prev
                    + eps + c1_true * eps_prev;
        eps_prev = eps;
    }

    printf("Generated %d data points from ARMAX(1,1,1,1)\n", N);
    printf("True: a1=%.2f b1=%.2f c1=%.2f\n\n", a1_true, b1_true, c1_true);

    PEMResult *result = pem_result_alloc(3);
    PEMOptions opts = pem_options_default();
    opts.max_iterations = 50;
    opts.verbose = true;

    printf("Estimating ARMAX via Levenberg-Marquardt...\n");
    int ret = pem_estimate_armax(data, 1, 1, 1, 1, NULL, result, &opts);

    if (ret == 0) {
        pem_result_print(result);
        printf("\nParameter comparison:\n");
        printf("  a1: true=%.4f est=%.4f\n", a1_true, result->theta_hat[0]);
        printf("  b1: true=%.4f est=%.4f\n", b1_true, result->theta_hat[1]);
        printf("  c1: true=%.4f est=%.4f\n", c1_true, result->theta_hat[2]);
    }

    pem_result_free(result);
    pem_data_free(data);
    printf("\nDone.\n");
    return 0;
}