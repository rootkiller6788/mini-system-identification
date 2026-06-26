#include "uq_core.h"
#include "uq_validation.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* Example 1: Linear Regression with Full Uncertainty Quantification
 * Demonstrates OLS fitting, parameter confidence intervals,
 * prediction intervals, residual diagnostics, and ANOVA.
 */

int main(void) {
    printf("=== Uncertainty Quantification: Linear Regression ===\n\n");

    int n = 30, p = 2;
    /* Generate synthetic data: y = 2.0 + 3.5*x + noise ~ N(0, 0.5²) */
    UQMatrix* X = uq_matrix_create(n, p);
    UQVector* y = uq_vector_create(n);

    double true_beta0 = 2.0, true_beta1 = 3.5;
    printf("True model: y = %.1f + %.1f*x + ε, ε ~ N(0, 0.25)\n\n", true_beta0, true_beta1);

    UQDistribution* noise = uq_dist_create_normal(0.0, 0.5);
    for (int i = 0; i < n; i++) {
        double xi = (double)i / (double)(n - 1) * 10.0;
        uq_matrix_set(X, i, 0, 1.0);
        uq_matrix_set(X, i, 1, xi);
        y->components[i] = true_beta0 + true_beta1 * xi + uq_dist_sample(noise);
    }

    /* Fit OLS */
    UQLinearModel* lm = uq_lm_create(X, y);
    uq_lm_fit(lm);

    printf("OLS Results (n=%d, p=%d):\n", n, p);
    printf("  β0 = %.4f  (true: %.1f)\n", lm->coefficients->components[0], true_beta0);
    printf("  β1 = %.4f  (true: %.1f)\n", lm->coefficients->components[1], true_beta1);
    printf("  σ²  = %.4f\n", lm->sigma_squared);

    /* Parameter confidence intervals */
    double sb0 = sqrt(uq_matrix_get(lm->covariance_beta, 0, 0));
    double sb1 = sqrt(uq_matrix_get(lm->covariance_beta, 1, 1));
    double t_crit = uq_stats_student_t_quantile(0.975, n - p);
    printf("\n95%% Confidence Intervals:\n");
    printf("  β0: [%.4f, %.4f]\n",
           lm->coefficients->components[0] - t_crit * sb0,
           lm->coefficients->components[0] + t_crit * sb0);
    printf("  β1: [%.4f, %.4f]\n",
           lm->coefficients->components[1] - t_crit * sb1,
           lm->coefficients->components[1] + t_crit * sb1);

    /* Prediction at x_new = 5.0 */
    double x_new[] = {1.0, 5.0};
    double y_hat, se_fit, se_pred;
    uq_lm_predict(lm, x_new, &y_hat, &se_fit, &se_pred);
    printf("\nPrediction at x=5.0:\n");
    printf("  y_hat = %.4f ± %.4f (fit SE)\n", y_hat, se_fit);
    printf("  y_pred ∈ [%.4f, %.4f] (95%% PI)\n",
           y_hat - t_crit * se_pred, y_hat + t_crit * se_pred);

    /* ANOVA */
    double ssr, sse, sst, F, pF;
    uq_lm_anova(lm, &ssr, &sse, &sst, &F, &pF);
    printf("\nANOVA:\n");
    printf("  SSR = %.4f, SSE = %.4f, SST = %.4f\n", ssr, sse, sst);
    printf("  R²  = %.4f\n", ssr / sst);
    printf("  F-statistic = %.4f\n", F);

    /* PRESS statistic */
    double press = uq_lm_press(lm);
    printf("  PRESS = %.4f\n", press);

    /* Validation metrics */
    UQValidationResult rmse = uq_validate_rmse(y->components, lm->response->components, n);
    UQValidationResult r2 = uq_validate_r_squared(y->components, lm->response->components, n, p);
    printf("\nValidation:\n  RMSE = %.6f\n  R² = %.4f\n", rmse.value, r2.value);

    /* Cleanup */
    uq_dist_free(noise);
    uq_lm_free(lm);
    uq_matrix_free(X);
    uq_vector_free(y);

    printf("\n=== Example 1 Complete ===\n");
    return 0;
}
