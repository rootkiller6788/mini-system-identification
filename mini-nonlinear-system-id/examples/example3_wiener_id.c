/* Example 3: Wiener System Identification
 * Wiener model: u(t) -> x(t)=G(q)u(t) -> y(t)=h(x(t))
 * True system: x(t) = 0.6*u(t-1) + 0.4*u(t-2)
 *              y(t) = tanh(x(t))
 */
#include "nlsid_core.h"
#include "nlsid_models.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

int main(void) {
    printf("=== Example 3: Wiener System Identification ===\n\n");
    int N = 500;
    NLSIDDataset* ds = nlsid_dataset_create(1, 1, N, 0.1);
    double* u = ds->input->channels[0]->data;
    double* y = ds->output->channels[0]->data;

    /* Generate Wiener system data */
    unsigned int seed = 456;
    for (int t = 0; t < N; t++) {
        seed = seed * 1103515245 + 12345;
        u[t] = ((double)(seed & 0xFFFF) / 65535.0 - 0.5) * 2.0;
    }
    for (int t = 0; t < N; t++) {
        double xt = 0.0;
        if (t >= 1) xt += 0.6 * u[t-1];
        if (t >= 2) xt += 0.4 * u[t-2];
        y[t] = tanh(xt);
    }

    /* Create Wiener model */
    WienerModel* wm = wiener_create(1, 2, 1);
    BasisExpansion* nl = basis_expansion_polynomial(1, 3);
    wiener_set_static_nl(wm, nl);
    printf("Wiener model: na=%d, nb=%d, nk=%d\n", wm->na, wm->nb, wm->nk);

    /* Simulate */
    double y_sim[500];
    wiener_simulate(wm, u, N, NULL, y_sim);
    double fit = nlsid_compute_fit(y, y_sim, N);
    printf("Simulation fit (untrained): %.2f%%\n", fit);

    printf("Sample outputs (first 10):\n");
    for (int t = 0; t < 10; t++)
        printf("  t=%d: u=%.3f, x=%.3f, y=%.3f\n", t, u[t], 0.0, y[t]);

    wiener_free(wm);
    nlsid_dataset_free(ds);
    printf("\nExample 3 completed.\n");
    return 0;
}