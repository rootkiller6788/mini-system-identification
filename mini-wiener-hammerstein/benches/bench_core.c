/**
 * bench_core.c ? Performance Benchmarks for Wiener-Hammerstein Module
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "wh_model.h"
#include "wh_linear.h"
#include "wh_nonlinear.h"
#include "wh_simulation.h"
#include "wh_signal.h"

static double now_sec(void) {
    return (double)clock() / CLOCKS_PER_SEC;
}

int main(void) {
    printf("??? Wiener-Hammerstein Benchmarks ?????????????????????????\n\n");
    int N = 50000;
    double* u = (double*)malloc(N * sizeof(double));
    double* y = (double*)malloc(N * sizeof(double));
    double* buf = (double*)malloc(N * sizeof(double));
    if (!u || !y || !buf) { printf("Memory error\n"); return 1; }
    wh_signal_prbs(u, N, 1.0, 10, 42);

    /* Bench 1: WH model evaluation speed */
    {
        WH_Model* m = wh_model_create();
        double b1[] = {1.0, 0.5, 0.2};
        wh_linear_init_fir(&m->L1, b1, 3, 1.0);
        double nl_coeffs[] = {0.0, 0.5, 0.0, 0.5};
        wh_nl_init_polynomial(&m->N, nl_coeffs, 3);
        double b2[] = {0.8, 0.2};
        wh_linear_init_fir(&m->L2, b2, 2, 1.0);

        double t0 = now_sec();
        wh_model_reset(m);
        for (int i = 0; i < N; i++) y[i] = wh_model_evaluate(m, u[i]);
        double t1 = now_sec();
        printf("WH eval (L1=3, N=cubic, L2=2): %.3f sec for %d samples (%.0f samples/sec)\n",
               t1 - t0, N, N / (t1 - t0 + 1e-12));
        wh_model_free(m);
    }

    /* Bench 2: Linear block FIR speed */
    {
        WH_LinearBlock blk;
        double b[] = {1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005};
        wh_linear_init_fir(&blk, b, 8, 1.0);
        double t0 = now_sec();
        wh_linear_reset(&blk);
        for (int i = 0; i < N; i++) buf[i] = wh_linear_evaluate(&blk, u[i]);
        double t1 = now_sec();
        printf("FIR 8-tap: %.3f sec for %d samples (%.0f samples/sec)\n",
               t1 - t0, N, N / (t1 - t0 + 1e-12));
    }

    /* Bench 3: Nonlinearity batch evaluation */
    {
        WH_Nonlinearity nl;
        double coeffs[] = {1.0, 2.0, 0.5, 0.1, 0.01};
        wh_nl_init_polynomial(&nl, coeffs, 4);
        for (int i = 0; i < N; i++) buf[i] = u[i] * 0.1;
        double t0 = now_sec();
        wh_nl_evaluate_batch(&nl, buf, y, N);
        double t1 = now_sec();
        printf("NL poly5 batch: %.3f sec for %d samples (%.0f samples/sec)\n",
               t1 - t0, N, N / (t1 - t0 + 1e-12));
    }

    /* Bench 4: Simulation batch */
    {
        WH_Model* m = wh_model_create();
        WH_SimConfig cfg = wh_sim_config_default();
        cfg.n_transient = 0;
        WH_SimOutput out;
        double t0 = now_sec();
        wh_sim_run(m, u, N, &cfg, &out);
        double t1 = now_sec();
        printf("Sim batch identity: %.3f sec for %d samples (%.0f samples/sec)\n",
               t1 - t0, N, N / (t1 - t0 + 1e-12));
        wh_sim_output_free(&out);
        wh_model_free(m);
    }

    /* Bench 5: Signal generation */
    {
        double t0 = now_sec();
        wh_signal_multisine(buf, N, 1000.0, 10.0, 400.0, 20, 1.0, 42);
        double t1 = now_sec();
        printf("Multisine 20 harmonics: %.3f sec for %d samples (%.0f samples/sec)\n",
               t1 - t0, N, N / (t1 - t0 + 1e-12));
    }

    free(u); free(y); free(buf);
    printf("\nDone.\n");
    return 0;
}
