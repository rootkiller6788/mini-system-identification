#include "uq_propagation.h"
#include "uq_sensitivity.h"
#include "uq_sampling.h"
#include "uq_validation.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* Example 3: Uncertainty Propagation through a Nonlinear ODE Model
 * Demonstrates FOSM, Monte Carlo propagation, Sobol' sensitivity analysis,
 * PCE (Polynomial Chaos Expansion), and GP surrogate modeling.
 *
 * Model: Duffing oscillator peak amplitude f(k, c) = 1/√((1-k²)² + (c)²)
 * (normalized response amplitude of forced oscillator)
 */

static double duffing_amplitude(double* params, void* data) {
    (void)data;
    double k = params[0]; /* Stiffness ratio */
    double c = params[1]; /* Damping ratio */
    double denom = sqrt((1.0 - k * k) * (1.0 - k * k) + c * c);
    return (denom > 1e-12) ? 1.0 / denom : 100.0;
}

int main(void) {
    printf("=== Uncertainty Propagation: Duffing Oscillator ===\n\n");

    printf("Model: Peak amplitude A(k,c) = 1/sqrt((1-k²)² + c²)\n");
    printf("Input distributions:\n");
    printf("  k ~ N(0.8, 0.1²) — stiffness uncertainty\n");
    printf("  c ~ N(0.3, 0.05²) — damping uncertainty\n\n");

    /* Input distributions */
    UQDistribution* dist_k = uq_dist_create_normal(0.8, 0.1);
    UQDistribution* dist_c = uq_dist_create_normal(0.3, 0.05);

    /* ================================================================
     * Method 1: Monte Carlo Propagation
     * ================================================================ */
    printf("--- Method 1: Monte Carlo (N=10000) ---\n");
    int n_mc = 10000;
    double* mc_outputs = (double*)malloc(n_mc * sizeof(double));
    UQDistribution* input_dists[] = {dist_k, dist_c};

    double mc_mean, mc_var, mc_skew, mc_kurt;
    uq_mc_propagate(n_mc, duffing_amplitude, NULL, input_dists, 2,
                    mc_outputs, &mc_mean, &mc_var, &mc_skew, &mc_kurt);
    printf("  E[A] = %.4f\n", mc_mean);
    printf("  Var(A) = %.6f (σ = %.4f)\n", mc_var, sqrt(mc_var));
    printf("  Skewness = %.4f\n", mc_skew);
    printf("  Kurtosis = %.4f\n", mc_kurt);
    double p_fail = uq_probability_of_failure(mc_outputs, n_mc, 5.0);
    printf("  P(A > 5.0) = %.4f\n", p_fail);

    /* ================================================================
     * Method 2: First-Order Second-Moment (FOSM)
     * ================================================================ */
    printf("\n--- Method 2: FOSM ---\n");
    double x_mean[] = {0.8, 0.3};
    UQMatrix* x_cov = uq_matrix_create(2, 2);
    uq_matrix_set(x_cov, 0, 0, 0.01);  /* σ²_k */
    uq_matrix_set(x_cov, 1, 1, 0.0025); /* σ²_c */
    uq_matrix_set(x_cov, 0, 1, 0.0);
    uq_matrix_set(x_cov, 1, 0, 0.0);

    double fosm_mean, fosm_var;
    uq_fosm_propagate(duffing_amplitude, NULL, x_mean, x_cov, 2,
                      &fosm_mean, &fosm_var);
    printf("  FOSM E[A] = %.4f (MC: %.4f)\n", fosm_mean, mc_mean);
    printf("  FOSM Var(A) = %.6f (MC: %.6f)\n", fosm_var, mc_var);

    /* ================================================================
     * Method 3: Rosenblueth Point Estimate
     * ================================================================ */
    printf("\n--- Method 3: Rosenblueth 2-Point ---\n");
    double x_std[] = {0.1, 0.05};
    double ros_mean, ros_std;
    uq_rosenblueth_2p(duffing_amplitude, NULL, x_mean, x_std, 2,
                      &ros_mean, &ros_std);
    printf("  Rosenblueth E[A] = %.4f (MC: %.4f)\n", ros_mean, mc_mean);
    printf("  Rosenblueth σ[A] = %.4f (MC: %.4f)\n", ros_std, sqrt(mc_var));

    /* ================================================================
     * Method 4: Polynomial Chaos Expansion
     * ================================================================ */
    printf("\n--- Method 4: PCE (Hermite, p=4) ---\n");
    UQPCE* pce = uq_pce_create(UQ_PCE_HERMITE, 2, 4);
    uq_pce_build_basis(pce);
    printf("  PCE basis size = %d\n", pce->n_basis_functions);

    /* Generate training data for regression-based PCE */
    int n_train = 200;
    double* X_train = (double*)malloc(n_train * 2 * sizeof(double));
    double* y_train = (double*)malloc(n_train * sizeof(double));
    for (int i = 0; i < n_train; i++) {
        X_train[i * 2] = uq_dist_sample(dist_k);
        X_train[i * 2 + 1] = uq_dist_sample(dist_c);
        double params[] = {X_train[i * 2], X_train[i * 2 + 1]};
        y_train[i] = duffing_amplitude(params, NULL);
    }

    uq_pce_fit_regression(pce, X_train, y_train, n_train);
    double pce_mean, pce_var;
    uq_pce_mean_variance(pce, &pce_mean, &pce_var);
    printf("  PCE R² = %.6f\n", pce->r_squared);
    printf("  PCE E[A] = %.4f (MC: %.4f)\n", pce_mean, mc_mean);
    printf("  PCE Var(A) = %.6f (MC: %.6f)\n", pce_var, mc_var);

    /* Sobol' indices from PCE */
    uq_pce_sobol_indices(pce);
    printf("  Sobol' main effect k: %.4f\n", pce->sobol_main_indices[0]);
    printf("  Sobol' total effect k: %.4f\n", pce->sobol_total_indices[0]);
    printf("  Sobol' main effect c: %.4f\n", pce->sobol_main_indices[1]);
    printf("  Sobol' total effect c: %.4f\n", pce->sobol_total_indices[1]);

    /* ================================================================
     * Method 5: Gaussian Process Surrogate
     * ================================================================ */
    printf("\n--- Method 5: Gaussian Process Emulator ---\n");
    UQGaussianProcess* gp = uq_gp_create(UQ_GP_KERNEL_MATERN52, 2, n_train);
    uq_gp_set_data(gp, X_train, y_train);
    uq_gp_train(gp);
    printf("  GP log marginal likelihood = %.4f\n", gp->log_marginal_likelihood);

    /* Predict at nominal point */
    double x_test[] = {0.8, 0.3};
    double gp_var;
    double gp_pred = uq_gp_predict(gp, x_test, &gp_var);
    printf("  GP pred at (0.8, 0.3) = %.4f ± %.4f\n", gp_pred, sqrt(gp_var));

    /* ================================================================
     * Method 6: Sobol' Direct (Saltelli estimator)
     * ================================================================ */
    printf("\n--- Method 6: Sobol' Direct (Saltelli) ---\n");
    UQSensitivityAnalysis* sa = uq_sa_create(2, (char*[]){"k","c"});
    sa->n_samples = 500;
    uq_sobol_saltelli(sa, duffing_amplitude, NULL);
    printf("  S_k (main)  = %.4f\n", uq_sobol_main_effect(sa, 0));
    printf("  S_k (total) = %.4f\n", uq_sobol_total_effect(sa, 0));
    printf("  S_c (main)  = %.4f\n", uq_sobol_main_effect(sa, 1));
    printf("  S_c (total) = %.4f\n", uq_sobol_total_effect(sa, 1));
    printf("  Model evaluations: %.0f\n", sa->computation_cost);

    /* ================================================================
     * Summary Comparison
     * ================================================================ */
    printf("\n=== Comparison of Methods for E[A] ===\n");
    printf("  Monte Carlo (N=10000):  %.4f ± %.4f\n", mc_mean, sqrt(mc_var));
    printf("  FOSM:                   %.4f ± %.4f\n", fosm_mean, sqrt(fosm_var));
    printf("  Rosenblueth 2-pt:      %.4f ± %.4f\n", ros_mean, ros_std);
    printf("  PCE (Hermite p=4):     %.4f ± %.4f\n", pce_mean, sqrt(pce_var));

    printf("\nSensitivity: stiffness k explains ~%.0f%% of variance, ",
           uq_sobol_main_effect(sa, 0) * 100.0);
    printf("damping c ~%.0f%%\n", uq_sobol_main_effect(sa, 1) * 100.0);

    /* Cleanup */
    free(mc_outputs);
    free(X_train);
    free(y_train);
    uq_dist_free(dist_k);
    uq_dist_free(dist_c);
    uq_matrix_free(x_cov);
    uq_pce_free(pce);
    uq_gp_free(gp);
    uq_sa_free(sa);

    printf("\n=== Example 3 Complete ===\n");
    return 0;
}
