/* Example 4: DC Motor with Nonlinear Friction - System Identification
 * Motor model: J*dw/dt = K*i - B*w - T_c*sign(w) - T_v*w*|w|
 * Discretized: w(t+1) = a*w(t) + b*u(t) - c*sign(w(t)) + d*u(t)*|w(t)|
 * Application: Robotics, Electric Vehicles, Quadrotors, Tesla
 */
#include "nlsid_core.h"
#include "nlsid_models.h"
#include "nlsid_algorithms.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static double sgn(double x) { return (x > 0.0) ? 1.0 : ((x < 0.0) ? -1.0 : 0.0); }

int main(void) {
    printf("=== Example 4: DC Motor Nonlinear Friction Identification ===\n\n");

    int N = 600;
    NLSIDDataset* ds = nlsid_dataset_create(1, 1, N, 0.01);
    double* u = ds->input->channels[0]->data;
    double* y = ds->output->channels[0]->data;

    /* True parameters for a small DC motor */
    double a_true = 0.85, b_true = 0.12, c_true = 0.02, d_true = 0.05;

    /* Generate training data with varying input */
    unsigned int seed = 789;
    for (int t = 0; t < N; t++) {
        seed = seed * 1103515245 + 12345;
        u[t] = ((double)(seed & 0xFFFF) / 65535.0 - 0.5) * 10.0;
    }

    /* Simulate motor dynamics */
    y[0] = 0.0;
    for (int t = 1; t < N; t++) {
        double w_prev = y[t-1];
        double u_curr = u[t-1];
        y[t] = a_true * w_prev + b_true * u_curr
               - c_true * sgn(w_prev)
               + d_true * u_curr * fabs(w_prev);
    }

    printf("True parameters: a=%.4f, b=%.4f, c=%.4f, d=%.4f\n",
           a_true, b_true, c_true, d_true);

    /* Create NARX model with nonlinear basis for friction identification */
    NARXModel* narx = narx_create(2, 2, 1, 1, 1);
    BasisExpansion* be = basis_expansion_create(4, 6);

    /* Add basis functions for each nonlinear friction component */
    double rbf_p[5] = {0.5, 0.0, 0.0, 0.0, 0.0};
    basis_expansion_add_basis(be, BASIS_RBF, rbf_p, 5);
    double sig_p[5] = {1.0, 1.0, 0.0, 0.0, 0.0};
    basis_expansion_add_basis(be, BASIS_SIGMOID, sig_p, 5);
    basis_expansion_add_basis(be, BASIS_PIECEWISE_LINEAR, sig_p, 5);

    narx_set_expansion(narx, be);
    printf("NARX model with %d basis functions, %d parameters\n",
           narx->expansion->n_bases, narx->n_params);

    /* Linear LS initialization */
    nlsid_init_narx_ls(narx, ds);

    /* Predict and evaluate */
    double* y_hat = (double*)malloc((size_t)N * sizeof(double));
    for (int t = 0; t < N; t++)
        y_hat[t] = narx_predict_one_step(narx, y, u, t);

    double fit = nlsid_compute_fit(y, y_hat, N);
    double mse = nlsid_compute_mse(y, y_hat, N);
    printf("\nIdentification Results:\n");
    printf("  Fit: %.2f%%\n", fit);
    printf("  MSE: %.6e\n", mse);
    printf("  Application domain: DC motor, Quadrotor, Tesla actuator\n");

    free(y_hat);
    narx_free(narx);
    nlsid_dataset_free(ds);
    printf("\nExample 4 completed.\n");
    return 0;
}