/* Demo: Overview of Nonlinear System Identification
 * Shows key features: data generation, model fitting, validation
 */
#include "nlsid_core.h"
#include "nlsid_models.h"
#include <stdio.h>
#include <math.h>

int main(void) {
    printf("=================================================\n");
    printf("  mini-nonlinear-system-id: Feature Demo\n");
    printf("=================================================\n\n");

    /* 1. Signal demo */
    printf("[1] Signal Operations\n");
    Signal* sig = nlsid_signal_create(100, 0.01);
    for (int i = 0; i < 100; i++)
        nlsid_signal_set(sig, i, sin(0.1 * i));
    printf("  Signal: mean=%.4f, rms=%.4f\n",
           nlsid_signal_mean(sig), nlsid_signal_rms(sig));
    nlsid_signal_free(sig);

    /* 2. Basis function demo */
    printf("\n[2] Basis Functions\n");
    double x[2] = {1.0, -0.5};
    double rbf_p[3] = {1.0, 0.0, 0.0};
    printf("  RBF(1.0,-0.5; center=(0,0), sigma=1) = %.4f\n",
           basis_eval_rbf(x, rbf_p, 3));
    double sig_p[3] = {2.0, 1.0, 0.0};
    printf("  Sigmoid(1.0,-0.5; slope=2, dir=(1,0)) = %.4f\n",
           basis_eval_sigmoid(x, sig_p, 3));

    /* 3. NARX model demo */
    printf("\n[3] NARX Model\n");
    NARXModel* narx = narx_create(2, 2, 1, 1, 1);
    BasisExpansion* be = basis_expansion_polynomial(4, 2);
    narx_set_expansion(narx, be);
    printf("  Created NARX(ny=2, nu=2, nk=1): %d params\n", narx->n_params);

    /* 4. Performance criteria */
    printf("\n[4] Model Selection Criteria\n");
    printf("  AIC(mse=0.01, N=500, d=5) = %.2f\n",
           nlsid_compute_aic(0.01, 500, 5));
    printf("  BIC(mse=0.01, N=500, d=5) = %.2f\n",
           nlsid_compute_bic(0.01, 500, 5));
    printf("  FPE(mse=0.01, N=500, d=5) = %.6f\n",
           nlsid_compute_fpe(0.01, 500, 5));

    narx_free(narx);
    printf("\n=================================================\n");
    printf("  Demo completed successfully.\n");
    printf("=================================================\n");
    return 0;
}
