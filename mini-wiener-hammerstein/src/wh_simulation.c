/**
 * wh_simulation.c ? Time-Domain Simulation of Wiener-Hammerstein Models
 *
 * Implements deterministic and stochastic simulation, performance metrics
 * (FIT, MSE, RMSE, MAE, NRMSE), Monte Carlo analysis, and special
 * simulation modes (impulse, step, frequency sweep).
 *
 * Knowledge Level: L5 (Algorithms/Methods), L6 (Canonical Problems)
 */

#include "wh_simulation.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ??? Pseudo-random number generator (LFSR-based) ???????????????????????? */

static unsigned int g_rand_state = 42;

static void set_seed(unsigned int seed) {
    g_rand_state = (seed == 0) ? 42 : seed;
}

static double rand_uniform(void) {
    /* LCG: X_{n+1} = (1103515245 * X_n + 12345) mod 2^31 */
    g_rand_state = 1103515245 * g_rand_state + 12345;
    return (double)(g_rand_state & 0x7FFFFFFF) / (double)0x7FFFFFFF;
}

static double rand_gaussian(void) {
    /* Box-Muller transform */
    double u1 = rand_uniform();
    double u2 = rand_uniform();
    if (u1 < 1e-12) u1 = 1e-12; /* Avoid log(0) */
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

/* ??? Default configuration ?????????????????????????????????????????????? */

WH_SimConfig wh_sim_config_default(void) {
    WH_SimConfig c;
    c.n_transient = 100;
    c.add_noise = 0;
    c.noise_std = 0.01;
    c.noise_seed = 42;
    c.record_intermediate = 0;
    c.verbosity = 0;
    return c;
}

/* ??? Core simulation ???????????????????????????????????????????????????? */

int wh_sim_run(const WH_Model* model, const double* u, int n_samples,
               const WH_SimConfig* config, WH_SimOutput* output) {
    if (!model || !u || !output || n_samples <= 0) return -1;
    WH_SimConfig cfg = config ? *config : wh_sim_config_default();
    memset(output, 0, sizeof(WH_SimOutput));

    set_seed(cfg.noise_seed);

    /* Create mutable copy for simulation */
    WH_Model sim_model;
    memcpy(&sim_model, model, sizeof(WH_Model));
    wh_model_reset(&sim_model);

    /* Run through transient period */
    for (int i = 0; i < cfg.n_transient; i++) {
        double u_val = (i < n_samples) ? u[i] : u[n_samples - 1];
        wh_model_evaluate(&sim_model, u_val);
    }

    /* Effective samples */
    int n_eff = n_samples - cfg.n_transient;
    if (n_eff < 0) n_eff = 0;

    /* Allocate output arrays */
    output->y = (double*)calloc(n_eff, sizeof(double));
    output->u = (double*)calloc(n_eff, sizeof(double));
    if (cfg.record_intermediate) {
        output->x = (double*)calloc(n_eff, sizeof(double));
        output->w = (double*)calloc(n_eff, sizeof(double));
    }
    if (!output->y || !output->u) {
        wh_sim_output_free(output);
        return -1;
    }

    /* Main simulation loop */
    wh_model_reset(&sim_model);
    /* Re-run through transient so that intermediate recording is correct */
    for (int i = 0; i < cfg.n_transient && i < n_samples; i++) {
        wh_model_evaluate(&sim_model, u[i]);
    }

    for (int i = cfg.n_transient; i < n_samples; i++) {
        int idx = i - cfg.n_transient;
        double y_val = wh_model_evaluate(&sim_model, u[i]);

        /* Add noise if requested */
        if (cfg.add_noise) {
            y_val += cfg.noise_std * rand_gaussian();
        }

        output->y[idx] = y_val;
        output->u[idx] = u[i];
        if (cfg.record_intermediate) {
            output->x[idx] = sim_model.x_current;
            output->w[idx] = sim_model.w_current;
        }
    }

    output->n_samples = n_eff;
    return 0;
}

/* ??? Simulation with reference ?????????????????????????????????????????? */

int wh_sim_run_with_reference(const WH_Model* model,
                               const double* u, const double* y_ref,
                               int n_samples, const WH_SimConfig* config,
                               WH_SimOutput* output) {
    int ret = wh_sim_run(model, u, n_samples, config, output);
    if (ret != 0) return ret;

    /* Compute performance metrics */
    int n = output->n_samples;
    if (n > 0 && y_ref) {
        output->mse = wh_sim_compute_mse(y_ref, output->y, n);
        output->fit_percent = wh_sim_compute_fit(y_ref, output->y, n);
    }
    return 0;
}

void wh_sim_output_free(WH_SimOutput* output) {
    if (!output) return;
    free(output->y);
    free(output->u);
    free(output->x);
    free(output->w);
    memset(output, 0, sizeof(WH_SimOutput));
}

/* ??? Performance metrics ???????????????????????????????????????????????? */

double wh_sim_compute_fit(const double* y_ref, const double* y_sim, int n) {
    if (!y_ref || !y_sim || n <= 0) return -1e100;

    /* Compute mean of reference */
    double mean_y = 0.0;
    for (int i = 0; i < n; i++) mean_y += y_ref[i];
    mean_y /= n;

    /* Compute ||y_ref - y_sim||? and ||y_ref - mean||? */
    double num = 0.0, den = 0.0;
    for (int i = 0; i < n; i++) {
        double e = y_ref[i] - y_sim[i];
        double d = y_ref[i] - mean_y;
        num += e * e;
        den += d * d;
    }

    if (den < 1e-16) {
        /* All reference values equal ? if model matches, FIT=100 */
        return (num < 1e-16) ? 100.0 : -1e100;
    }

    double fit = 100.0 * (1.0 - sqrt(num) / sqrt(den));
    return fit < -1e6 ? -1e6 : fit;
}

double wh_sim_compute_mse(const double* y_ref, const double* y_sim, int n) {
    if (!y_ref || !y_sim || n <= 0) return 1e100;
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        double e = y_ref[i] - y_sim[i];
        sum += e * e;
    }
    return sum / n;
}

double wh_sim_compute_rmse(const double* y_ref, const double* y_sim, int n) {
    return sqrt(wh_sim_compute_mse(y_ref, y_sim, n));
}

double wh_sim_compute_mae(const double* y_ref, const double* y_sim, int n) {
    if (!y_ref || !y_sim || n <= 0) return 1e100;
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += fabs(y_ref[i] - y_sim[i]);
    }
    return sum / n;
}

double wh_sim_compute_nrmse(const double* y_ref, const double* y_sim, int n) {
    if (!y_ref || !y_sim || n <= 0) return 1e100;
    /* Find range of reference signal */
    double y_min = y_ref[0], y_max = y_ref[0];
    for (int i = 1; i < n; i++) {
        if (y_ref[i] < y_min) y_min = y_ref[i];
        if (y_ref[i] > y_max) y_max = y_ref[i];
    }
    double rng = y_max - y_min;
    if (rng < 1e-12) rng = 1.0;
    return wh_sim_compute_rmse(y_ref, y_sim, n) / rng;
}

/* ??? Monte Carlo simulation ????????????????????????????????????????????? */

int wh_sim_monte_carlo(const WH_Model* model, const double* u, int n_samples,
                        int n_runs, const WH_SimConfig* config_base,
                        double* mean_y, double* std_y) {
    if (!model || !u || !mean_y || !std_y || n_samples <= 0 || n_runs <= 0)
        return -1;

    /* Accumulate sum and sum of squares for each sample */
    double* sum_y = (double*)calloc(n_samples, sizeof(double));
    double* sum_y2 = (double*)calloc(n_samples, sizeof(double));
    if (!sum_y || !sum_y2) { free(sum_y); free(sum_y2); return -1; }

    WH_SimConfig run_config = config_base ? *config_base : wh_sim_config_default();

    for (int run = 0; run < n_runs; run++) {
        run_config.noise_seed = 42 + run * 137;
        WH_SimOutput output;
        memset(&output, 0, sizeof(WH_SimOutput));

        /* Simulate without transient removal to get fixed-length output */
        WH_SimConfig local_cfg = run_config;
        local_cfg.n_transient = 0;
        local_cfg.add_noise = 1; /* Enable noise for MC */

        if (wh_sim_run(model, u, n_samples, &local_cfg, &output) == 0) {
            for (int i = 0; i < n_samples && i < output.n_samples; i++) {
                sum_y[i] += output.y[i];
                sum_y2[i] += output.y[i] * output.y[i];
            }
        }
        wh_sim_output_free(&output);
    }

    /* Compute statistics */
    for (int i = 0; i < n_samples; i++) {
        mean_y[i] = sum_y[i] / n_runs;
        double variance = sum_y2[i] / n_runs - mean_y[i] * mean_y[i];
        if (variance < 0.0) variance = 0.0;
        std_y[i] = sqrt(variance);
    }

    free(sum_y); free(sum_y2);
    return 0;
}

/* ??? Transient analysis ????????????????????????????????????????????????? */

int wh_sim_find_transient(const WH_Model* model, double tol) {
    if (!model) return 0;
    /* Find the slowest time constant of both linear blocks. */
    int n_samples = 500;
    double step_resp[500];
    WH_Model sim_model;
    memcpy(&sim_model, model, sizeof(WH_Model));
    wh_sim_step_response(&sim_model, 1.0, step_resp, n_samples);

    /* Find when output settles to within tol of final value */
    double final_val = step_resp[n_samples - 1];
    for (int i = n_samples - 1; i >= 0; i--) {
        if (fabs(final_val) > 1e-12) {
            double rel_err = fabs(step_resp[i] - final_val) / fabs(final_val);
            if (rel_err > tol) return i + 1;
        }
    }
    return n_samples / 4; /* Default: 25% of max */
}

/* ??? Impulse and step response ?????????????????????????????????????????? */

void wh_sim_impulse_response(const WH_Model* model, double amplitude,
                              double* response, int n) {
    if (!model || !response || n <= 0) return;
    WH_Model sim_model;
    memcpy(&sim_model, model, sizeof(WH_Model));
    wh_model_reset(&sim_model);

    /* Apply impulse */
    wh_model_evaluate(&sim_model, amplitude);
    for (int i = 0; i < n; i++) {
        response[i] = wh_model_evaluate(&sim_model, 0.0);
    }
}

void wh_sim_step_response(const WH_Model* model, double amplitude,
                           double* response, int n) {
    if (!model || !response || n <= 0) return;
    WH_Model sim_model;
    memcpy(&sim_model, model, sizeof(WH_Model));
    wh_model_reset(&sim_model);

    for (int i = 0; i < n; i++) {
        response[i] = wh_model_evaluate(&sim_model, amplitude);
    }
}

/* ??? Frequency sweep ???????????????????????????????????????????????????? */

int wh_sim_frequency_sweep(const WH_Model* model,
                            double f0, double f1, double fs,
                            double duration, double A, double* y_out) {
    if (!model || !y_out || fs <= 0.0 || duration <= 0.0) return -1;
    int N = (int)(fs * duration);
    if (N <= 0 || N > 65536) return -1;

    /* Generate chirp input: u(t) = A * sin(phase(t)) */
    WH_Model sim_model;
    memcpy(&sim_model, model, sizeof(WH_Model));
    wh_model_reset(&sim_model);

    for (int i = 0; i < N; i++) {
        double t = i / fs;
        (void)(f0 + (f1 - f0) * t / duration); /* f_inst not needed for phase computation */
        double phase = 2.0 * M_PI * (f0 * t + 0.5 * (f1 - f0) * t * t / duration);
        double u = A * sin(phase);
        y_out[i] = wh_model_evaluate(&sim_model, u);
    }
    return 0;
}
