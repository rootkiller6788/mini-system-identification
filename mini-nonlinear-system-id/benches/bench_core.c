/* Benchmark: Core operations for nonlinear system identification */
#include "nlsid_core.h"
#include "nlsid_models.h"
#include "nlsid_algorithms.h"
#include <stdio.h>
#include <math.h>
#include <time.h>

static double wall_time(void) {
    return (double)clock() / (double)CLOCKS_PER_SEC;
}

int main(void) {
    printf("=== Benchmarks: mini-nonlinear-system-id ===\n\n");
    double t0, elapsed;

    /* Benchmark 1: Signal creation and fill (1M samples) */
    t0 = wall_time();
    Signal* sig = nlsid_signal_create(1000000, 0.001);
    nlsid_signal_fill(sig, 1.0);
    elapsed = wall_time() - t0;
    printf("Signal create + fill (1M): %.4f s\n", elapsed);

    /* Benchmark 2: Basis expansion evaluation */
    BasisExpansion* be = basis_expansion_polynomial(4, 3);
    double x[4] = {1.0, 0.5, -0.3, 0.8};
    t0 = wall_time();
    for (int i = 0; i < 100000; i++)
        basis_expansion_eval(be, x);
    elapsed = wall_time() - t0;
    printf("Basis expansion eval (100k): %.4f s\n", elapsed);

    /* Benchmark 3: NARX simulation */
    NARXModel* narx = narx_create(3, 3, 1, 1, 1);
    BasisExpansion* be2 = basis_expansion_polynomial(6, 2);
    narx_set_expansion(narx, be2);
    double input[1000];
    double output[1000];
    for (int i = 0; i < 1000; i++) input[i] = sin(0.1 * i);
    t0 = wall_time();
    narx_simulate(narx, input, 1000, NULL, output);
    elapsed = wall_time() - t0;
    printf("NARX simulation (1000 steps): %.6f s\n", elapsed);

    /* Benchmark 4: Matrix operations */
    int N = 100;
    double* A = (double*)malloc((size_t)(N*N) * sizeof(double));
    double* b = (double*)malloc((size_t)N * sizeof(double));
    double* sol = (double*)malloc((size_t)N * sizeof(double));
    for (int i = 0; i < N*N; i++) A[i] = (i % (N+1) == 0) ? 2.0 : 0.1;
    for (int i = 0; i < N; i++) b[i] = 1.0;
    t0 = wall_time();
    nlsid_solve_linear_system(A, b, N, sol);
    elapsed = wall_time() - t0;
    printf("Linear solve (%dx%d): %.6f s\n", N, N, elapsed);
    free(A); free(b); free(sol);

    nlsid_signal_free(sig);
    basis_expansion_free(be);
    narx_free(narx);
    printf("\nBenchmarks completed.\n");
    return 0;
}
