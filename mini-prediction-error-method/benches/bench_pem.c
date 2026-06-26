#include "pem_core.h"
#include "pem_model.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* Benchmark: PEM estimation performance
 * Measures computation time for ARX LS estimation on large datasets. */

int main(void) {
    printf("=== PEM Performance Benchmark ===\n\n");

    int sizes[] = {100, 500, 1000, 5000, 10000};
    int n_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int s = 0; s < n_sizes; s++) {
        int N = sizes[s];
        PEMData *data = pem_data_alloc(N);

        /* Generate ARX(2,1,1) data */
        data->y[0] = 0.0; data->y[1] = 0.0;
        for (int t = 0; t < N; t++) {
            data->u[t] = (t % 10 < 5) ? 1.0 : -1.0;
            if (t >= 2)
                data->y[t] = 0.7*data->y[t-1] - 0.2*data->y[t-2] + 0.5*data->u[t-1];
        }

        PEMResult *result = pem_result_alloc(3);
        PEMOptions opts = pem_options_default();

        clock_t t0 = clock();
        pem_estimate_arx_ls(data, 2, 1, 1, result, &opts);
        clock_t t1 = clock();

        double elapsed = (double)(t1 - t0) / (double)CLOCKS_PER_SEC;
        printf("N=%6d: %.6f sec  (%.2f usec/sample)  loss=%.6e\n",
               N, elapsed, 1e6 * elapsed / (double)N, result->loss_final);

        pem_result_free(result);
        pem_data_free(data);
    }

    printf("\nDone.\n");
    return 0;
}