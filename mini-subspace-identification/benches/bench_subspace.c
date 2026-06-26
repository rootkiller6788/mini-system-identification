#include "subspace_core.h"
#include "subspace_algorithms.h"
#include "subspace_linalg.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

/* ============================================================================
 * Benchmark: Subspace Identification Performance
 *
 * Measures execution time for key operations at scale:
 *   - Block Hankel construction
 *   - Matrix multiplication
 *   - QR factorization
 *   - SVD computation
 *   - Full N4SID pipeline
 * ============================================================================ */

static double elapsed_sec(clock_t start) {
    return (double)(clock() - start) / (double)CLOCKS_PER_SEC;
}

int main(void) {
    printf("============================================================\n");
    printf("  Subspace Identification -- Performance Benchmark\n");
    printf("============================================================\n\n");

    /* Benchmark 1: Matrix multiply at scale */
    printf("Benchmark 1: Matrix Multiplication\n");
    int sizes[] = {50, 100, 200};
    for (int si = 0; si < 3; si++) {
        int n = sizes[si];
        SubspaceMatrix *A = subspace_matrix_alloc(n, n);
        SubspaceMatrix *B = subspace_matrix_alloc(n, n);
        SubspaceMatrix *C = subspace_matrix_alloc(n, n);
        if (!A || !B || !C) { printf("  Memory error at n=%d\n", n); continue; }

        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) {
                subspace_matrix_set(A, i, j, (double)(i + j) / (double)n);
                subspace_matrix_set(B, i, j, (double)(i - j) / (double)n);
            }

        clock_t start = clock();
        subspace_matrix_multiply(A, B, C);
        double t = elapsed_sec(start);
        printf("  %dx%d multiply: %.4f sec\n", n, n, t);

        subspace_matrix_free(A); subspace_matrix_free(B);
        subspace_matrix_free(C);
    }

    /* Benchmark 2: QR factorization */
    printf("\nBenchmark 2: QR Factorization\n");
    for (int si = 0; si < 2; si++) {
        int m = sizes[si], n = m / 2;
        SubspaceMatrix *A = subspace_matrix_alloc(m, n);
        SubspaceMatrix *Q = subspace_matrix_alloc(m, n);
        SubspaceMatrix *R = subspace_matrix_alloc(n, n);
        if (!A || !Q || !R) continue;

        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                subspace_matrix_set(A, i, j, (double)(i * j + 1));

        clock_t start = clock();
        subspace_qr_mgs(A, Q, R);
        double t = elapsed_sec(start);
        printf("  %dx%d QR (MGS): %.4f sec\n", m, n, t);

        subspace_matrix_free(A); subspace_matrix_free(Q);
        subspace_matrix_free(R);
    }

    /* Benchmark 3: SVD */
    printf("\nBenchmark 3: SVD Computation\n");
    int svd_sizes[] = {20, 40};
    for (int si = 0; si < 2; si++) {
        int n = svd_sizes[si];
        SubspaceMatrix *A = subspace_matrix_alloc(n, n);
        SubspaceSVD *svd = subspace_svd_alloc(n, n);
        if (!A || !svd) continue;

        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                subspace_matrix_set(A, i, j, (double)(i + j + 1));

        clock_t start = clock();
        subspace_svd_compute(A, svd);
        double t = elapsed_sec(start);
        printf("  %dx%d SVD: %.4f sec\n", n, n, t);

        subspace_matrix_free(A); subspace_svd_free(svd);
    }

    /* Benchmark 4: Full N4SID pipeline */
    printf("\nBenchmark 4: Full N4SID Pipeline\n");
    int N_vals[] = {500, 1000, 2000};
    for (int si = 0; si < 3; si++) {
        int N = N_vals[si];
        SubspaceData *data = subspace_data_alloc(N, 1, 1);
        if (!data) continue;

        srand(42);
        for (int k = 0; k < N; k++)
            data->u[k] = ((rand() % 2) ? 1.0 : -1.0);

        double x[2] = {0, 0};
        for (int k = 0; k < N; k++) {
            data->y[k] = 1.0*x[0] + 0.5*x[1] + 0.1*data->u[k]
                         + 0.01 * ((double)rand()/RAND_MAX - 0.5);
            double xn0 = 0.7*x[0] + 0.2*x[1] + 0.5*data->u[k];
            double xn1 = -0.1*x[0] + 0.8*x[1] + 0.3*data->u[k];
            x[0] = xn0; x[1] = xn1;
        }

        SubspaceOptions opts = subspace_options_default();
        opts.i = 10;
        opts.max_order = 8;
        SubspaceResult *result = subspace_result_alloc();

        clock_t start = clock();
        subspace_n4sid(data, &opts, result);
        double t = elapsed_sec(start);
        printf("  N=%d: %.4f sec (order=%d, fit=%.1f%%)\n",
               N, t, result->order_estimated, result->fit_percent);

        subspace_result_free(result);
        subspace_data_free(data);
    }

    printf("\nBenchmark complete.\n");
    return 0;
}
