#include "pem_core.h"
#include "pem_predictor.h"
#include "pem_criterion.h"
#include "pem_model.h"
#include "pem_optimizer.h"
#include "pem_validation.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* Demo: Comprehensive overview of the PEM framework
 *
 * Demonstrates the complete PEM workflow:
 *   1. Generate identification data
 *   2. Estimate ARX and OE models
 *   3. Validate models using statistical tests
 *   4. Compare model structures using information criteria
 *   5. Simulate and predict
 */

int main(void) {
    printf("============================================\n");
    printf("  Prediction Error Method - Demo Overview\n");
    printf("============================================\n\n");

    /* --- Part 1: Create test data --- */
    printf("[1] Generating test data...\n");
    int N = 300;
    PEMData *data = pem_data_alloc(N);

    /* ARX(2,2,1): y(t) = 0.7*y(t-1) - 0.2*y(t-2) + 0.5*u(t-1) + 0.3*u(t-2) */
    /* ARX params: a1=-0.7, a2=0.2, b1=0.5, b2=0.3 */
    data->y[0] = 0.0; data->y[1] = 0.0;
    for (int t = 0; t < N; t++) {
        data->u[t] = (t % 20 < 10) ? 1.0 : -1.0;
        if (t >= 2)
            data->y[t] = 0.7*data->y[t-1] - 0.2*data->y[t-2]
                       + 0.5*data->u[t-1] + 0.3*data->u[t-2];
    }
    printf("  Created %d samples of input-output data\n", N);

    /* --- Part 2: ARX Estimation --- */
    printf("\n[2] ARX Estimation (Least Squares)...\n");
    PEMResult *arx_result = pem_result_alloc(4);
    PEMOptions opts = pem_options_default();
    pem_estimate_arx_ls(data, 2, 2, 1, arx_result, &opts);
    pem_result_print(arx_result);

    /* --- Part 3: OE Estimation --- */
    printf("\n[3] OE Estimation (Levenberg-Marquardt)...\n");
    PEMResult *oe_result = pem_result_alloc(4);
    opts.max_iterations = 30;
    pem_estimate_oe(data, 2, 2, 1, NULL, oe_result, &opts);
    pem_result_print(oe_result);

    /* --- Part 4: Model Validation --- */
    printf("\n[4] Model Validation...\n");
    double *arx_yhat = (double*)malloc((size_t)N * sizeof(double));
    double *arx_eps = (double*)malloc((size_t)N * sizeof(double));
    double *oe_yhat = (double*)malloc((size_t)N * sizeof(double));
    double *oe_eps = (double*)malloc((size_t)N * sizeof(double));

    pem_predict_arx_batch(arx_result->theta_hat, 2, 2, 1,
                          data->u, data->y, N, arx_yhat);
    pem_residuals_arx(arx_result->theta_hat, 2, 2, 1,
                      data->u, data->y, N, arx_eps);
    pem_predict_oe_batch(oe_result->theta_hat, 2, 2, 1, data->u, N, oe_yhat);
    pem_residuals_oe(oe_result->theta_hat, 2, 2, 1,
                     data->u, data->y, N, oe_eps);

    PEMValidation *arx_val = pem_validation_alloc();
    PEMValidation *oe_val = pem_validation_alloc();
    pem_validate_model(data->y, arx_yhat, arx_eps, data->u, N, 4, arx_val);
    pem_validate_model(data->y, oe_yhat, oe_eps, data->u, N, 4, oe_val);

    printf("\nARX Model:\n");
    pem_validation_print(arx_val);
    printf("\nOE Model:\n");
    pem_validation_print(oe_val);

    /* --- Part 5: Comparison --- */
    printf("\n[5] Model Comparison:\n");
    printf("  %-20s %12s %12s\n", "Metric", "ARX", "OE");
    printf("  %-20s %12.2f %12.2f\n", "Fit (%)",
           arx_val->fit_percent, oe_val->fit_percent);
    printf("  %-20s %12.4f %12.4f\n", "AIC",
           arx_val->aic, oe_val->aic);
    printf("  %-20s %12.4f %12.4f\n", "BIC",
           arx_val->bic, oe_val->bic);
    printf("  %-20s %12.4f %12.4f\n", "ResidWhiteness",
           arx_val->residual_whiteness, oe_val->residual_whiteness);

    /* Cleanup */
    pem_result_free(arx_result);
    pem_result_free(oe_result);
    pem_data_free(data);
    free(arx_yhat); free(arx_eps); free(oe_yhat); free(oe_eps);
    pem_validation_free(arx_val);
    pem_validation_free(oe_val);

    printf("\n============================================\n");
    printf("  Demo Complete\n");
    printf("============================================\n");
    return 0;
}