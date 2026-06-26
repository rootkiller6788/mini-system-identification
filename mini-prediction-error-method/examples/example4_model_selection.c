#include "pem_core.h"
#include "pem_model.h"
#include "pem_validation.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* Example 4: Model Order Selection via Information Criteria
 *
 * Demonstrates how to use AIC, AICc, BIC, and FPE to select
 * the optimal model order among candidate models.
 *
 * True system: ARX(2,2,1)
 * Candidate models: ARX(1,1,1), ARX(2,2,1), ARX(3,3,1)
 */

static void evaluate_model(PEMData *data, int na, int nb, int nk,
                           const char *label) {
    int npar = na + nb;
    PEMResult *result = pem_result_alloc(npar);
    PEMOptions opts = pem_options_default();

    pem_estimate_arx_ls(data, na, nb, nk, result, &opts);

    /* Compute validation statistics */
    int N = data->N;
    double *y_hat = (double*)malloc((size_t)N * sizeof(double));
    double *eps = (double*)malloc((size_t)N * sizeof(double));
    pem_predict_arx_batch(result->theta_hat, na, nb, nk,
                          data->u, data->y, N, y_hat);
    pem_residuals_arx(result->theta_hat, na, nb, nk,
                      data->u, data->y, N, eps);

    PEMValidation *val = pem_validation_alloc();
    pem_validate_model(data->y, y_hat, eps, data->u, N, npar, val);

    printf("\n--- %s (na=%d, nb=%d, npar=%d) ---\n", label, na, nb, npar);
    printf("  Loss:     %.6e\n", val->loss);
    printf("  Fit:      %.2f%%\n", val->fit_percent);
    printf("  AIC:      %.4f\n", val->aic);
    printf("  AICc:     %.4f\n", val->aicc);
    printf("  BIC:      %.4f\n", val->bic);
    printf("  FPE:      %.6f\n", val->fpe);

    pem_result_free(result);
    free(y_hat); free(eps);
    pem_validation_free(val);
}

int main(void) {
    printf("=== Example 4: Model Order Selection ===\n\n");

    int N = 300;
    PEMData *data = pem_data_alloc(N);

    /* Generate data from ARX(2,2,1): a1=-0.5, a2=0.2, b1=0.6, b2=0.3 */
    double a1 = -0.5, a2 = 0.2, b1 = 0.6, b2 = 0.3;
    data->y[0] = 0.0; data->y[1] = 0.0;
    for (int t = 0; t < N; t++) {
        data->u[t] = (t % 12 < 6) ? 1.0 : -1.0;
        if (t >= 2) {
            data->y[t] = -a1 * data->y[t-1] - a2 * data->y[t-2]
                        + b1 * data->u[t-1] + b2 * data->u[t-2];
        }
    }

    printf("Data generated from ARX(2,2,1)\n");
    printf("True: a1=%.1f a2=%.1f b1=%.1f b2=%.1f\n\n", a1, a2, b1, b2);

    /* Evaluate candidate models */
    evaluate_model(data, 1, 1, 1, "ARX(1,1,1) underfit");
    evaluate_model(data, 2, 2, 1, "ARX(2,2,1) true    ");
    evaluate_model(data, 3, 3, 1, "ARX(3,3,1) overfit ");

    printf("\nThe true model should have the best (lowest) AICc/BIC.\n");

    pem_data_free(data);
    return 0;
}