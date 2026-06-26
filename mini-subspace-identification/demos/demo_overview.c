#include "subspace_core.h"
#include "subspace_algorithms.h"
#include "subspace_order.h"
#include "subspace_validation.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* ============================================================================
 * Demo: Subspace Identification Overview
 *
 * Demonstrates the complete subspace identification workflow:
 *   - Data generation
 *   - N4SID, MOESP, CVA comparison
 *   - Order estimation methods
 *   - Model validation
 * ============================================================================ */

int main(void) {
    printf("==============================================================\n");
    printf("  Subspace Identification -- Comprehensive Demo\n");
    printf("==============================================================\n\n");

    int N = 600, r = 1, m = 1;
    SubspaceData *data = subspace_data_alloc(N, r, m);
    if (!data) return 1;

    /* Generate data from a 2nd-order system */
    srand(54321);
    for (int k = 0; k < N; k++)
        data->u[k] = ((rand() % 2) ? -1.0 : 1.0);

    double x[2] = {0, 0};
    double A[4] = {0.8, 0.15, -0.15, 0.75};
    double B[2] = {0.6, 0.4};
    double C[2] = {1.0, 0.0};
    double D = 0.05;

    for (int k = 0; k < N; k++) {
        data->y[k] = C[0]*x[0] + C[1]*x[1] + D*data->u[k]
                     + 0.01 * ((double)rand()/RAND_MAX - 0.5);
        double xn0 = A[0]*x[0] + A[1]*x[1] + B[0]*data->u[k];
        double xn1 = A[2]*x[0] + A[3]*x[1] + B[1]*data->u[k];
        x[0] = xn0; x[1] = xn1;
    }

    printf("True system: 2nd-order, eigenvalues at %.4f +/- %.4fj\n",
           0.775, 0.148);
    printf("Data: %d samples\n\n", N);

    /* Run all three algorithms */
    SubspaceOptions opts = subspace_options_default();
    opts.i = 12;
    opts.max_order = 6;

    const char *names[] = {"N4SID", "MOESP", "CVA"};
    SubspaceAlgorithm algos[] = {SS_N4SID, SS_MOESP, SS_CVA};
    int alg_count = 3;

    printf("%-8s %8s %10s %10s %10s %8s\n",
           "Method", "Order", "Fit(%)", "Loss", "Cond(A)", "Time(s)");
    printf("----------------------------------------------------\n");

    for (int a = 0; a < alg_count; a++) {
        opts.algorithm = algos[a];
        SubspaceResult *res = subspace_result_alloc();
        subspace_identify(data, &opts, res);
        printf("%-8s %8d %10.2f %10.4e %10.2e %8.4f\n",
               names[a], res->order_estimated, res->fit_percent,
               res->loss, res->condition_A, res->elapsed_sec);
        subspace_result_free(res);
    }

    /* Order estimation methods comparison */
    printf("\n--- Order Estimation Methods ---\n");
    /* Generate singular values with gap at n=2 */
    double sv[10] = {50.0, 20.0, 0.5, 0.2, 0.08, 0.05, 0.02, 0.01, 0.005, 0.002};
    printf("Singular values: ");
    for (int i = 0; i < 5; i++) printf("%.2f ", sv[i]);
    printf("...\n");
    printf("  SVD Gap:      order = %d\n", subspace_order_svd_gap(sv, 10, 6));
    printf("  SVC (0.99):   order = %d\n", subspace_order_svc(sv, 10, 0.99, 6));
    printf("  AIC:          order = %d\n", subspace_order_aic(sv, 10, 500, 1, 1, 6));
    printf("  NIC:          order = %d\n", subspace_order_nic(sv, 10, 500, 1, 1, 6));
    printf("  Consensus:    order = %d\n", subspace_order_consensus(sv, 10, 500, 1, 1, 6));

    subspace_data_free(data);
    printf("\nDemo complete.\n");
    return 0;
}
