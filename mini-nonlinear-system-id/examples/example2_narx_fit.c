/* Example 2: NARX Model Fitting with Polynomial Basis
 * Fit a NARX model to data from a nonlinear oscillator:
 * y(t) = 0.5*y(t-1) + 0.3*y(t-2) + 0.2*u(t-1) - 0.1*y(t-1)^2
 */
#include "nlsid_core.h"
#include "nlsid_models.h"
#include "nlsid_algorithms.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

int main(void) {
    printf("=== Example 2: NARX Model Fitting ===\n\n");

    int N = 400;
    double Ts = 0.1;
    NLSIDDataset* ds = nlsid_dataset_create(1, 1, N, Ts);
    double* u = ds->input->channels[0]->data;
    double* y = ds->output->channels[0]->data;

    /* Generate data: nonlinear oscillator with quadratic term */
    unsigned int seed = 123;
    for (int t = 0; t < N; t++) {
        seed = seed * 1103515245 + 12345;
        u[t] = ((double)(seed & 0x3FF) / 512.0 - 1.0);
    }
    for (int t = 0; t < N; t++) {
        double yt = 0.0;
        if (t >= 1) yt += 0.5 * y[t-1];
        if (t >= 2) yt += 0.3 * y[t-2];
        if (t >= 1) yt += 0.2 * u[t-1];
        if (t >= 1) yt -= 0.1 * y[t-1] * y[t-1]; /* nonlinear term */
        y[t] = yt;
    }

    /* Create NARX model with polynomial basis expansion */
    NARXModel* narx = narx_create(2, 2, 1, 1, 1);
    BasisExpansion* be = basis_expansion_polynomial(narx->regressor_dim, 2);
    narx_set_expansion(narx, be);
    printf("NARX model: ny=%d, nu=%d, nk=%d, params=%d\n",
           narx->ny, narx->nu, narx->nk, narx->n_params);

    /* Initialize via linear least squares */
    nlsid_init_narx_ls(narx, ds);
    printf("Initial parameters set via linear LS\n");

    /* Compute one-step-ahead predictions */
    double* y_hat = (double*)malloc((size_t)N * sizeof(double));
    for (int t = 0; t < N; t++)
        y_hat[t] = narx_predict_one_step(narx, y, u, t);

    double fit = nlsid_compute_fit(y, y_hat, N);
    double mse = nlsid_compute_mse(y, y_hat, N);
    printf("Fit: %.2f%%, MSE: %.6e\n", fit, mse);

    printf("First 10 predictions vs actual:\n");
    for (int t = 10; t < 20; t++)
        printf("  t=%3d: actual=%.4f, predicted=%.4f\n", t, y[t], y_hat[t]);

    free(y_hat);
    nlsid_dataset_free(ds);
    narx_free(narx);
    printf("\nExample 2 completed.\n");
    return 0;
}