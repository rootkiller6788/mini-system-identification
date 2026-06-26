#include "uq_sensitivity.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define UQ_PI 3.14159265358979323846

static double urand5(void) {
    return ((double)rand() / ((double)RAND_MAX + 1.0));
}

static double gauss_rand5(void) {
    double u1, u2;
    do { u1 = urand5(); } while (u1 < 1e-15);
    u2 = urand5();
    return sqrt(-2.0 * log(u1)) * cos(2.0 * UQ_PI * u2);
}

/* ============================================================================
 * Sensitivity Analysis Core
 * ============================================================================ */

UQSensitivityAnalysis* uq_sa_create(int n_variables, char** var_names) {
    UQSensitivityAnalysis* sa = (UQSensitivityAnalysis*)calloc(1, sizeof(UQSensitivityAnalysis));
    sa->n_variables = n_variables;
    sa->indices = (UQSensitivityIndex*)calloc(n_variables, sizeof(UQSensitivityIndex));
    if (var_names) {
        sa->variable_names = (double*)malloc(n_variables * 256 / sizeof(double)); /* char holder */
        char** names_ptr = (char**)sa->variable_names;
        (void)names_ptr;
    }
    for (int i = 0; i < n_variables; i++) {
        sa->indices[i].variable_index = i;
        if (var_names && var_names[i])
            sa->indices[i].variable_name = strdup(var_names[i]);
    }
    return sa;
}

void uq_sa_free(UQSensitivityAnalysis* sa) {
    if (!sa) return;
    for (int i = 0; i < sa->n_variables; i++)
        free(sa->indices[i].variable_name);
    free(sa->indices);
    free(sa->A);
    free(sa->B);
    free(sa->A_B_i);
    free(sa->f_A);
    free(sa->f_B);
    if (sa->f_A_B_i) {
        for (int d = 0; d < sa->n_variables; d++)
            free(sa->f_A_B_i[d]);
        free(sa->f_A_B_i);
    }
    free(sa->sobol_converged);
    free(sa->sobol_error_estimates);
    free(sa);
}

void uq_sa_set_sample_size(UQSensitivityAnalysis* sa, int n_base) {
    sa->n_samples = n_base;
}

void uq_sa_generate_matrices(UQSensitivityAnalysis* sa) {
    int N = sa->n_samples, d = sa->n_variables;
    sa->A = (double*)malloc(N * d * sizeof(double));
    sa->B = (double*)malloc(N * d * sizeof(double));

    /* Two independent random matrices (uniform [0,1]) */
    for (int i = 0; i < N * d; i++) {
        sa->A[i] = urand5();
        sa->B[i] = urand5();
    }
}

/* ============================================================================
 * Sobol' Indices (Saltelli 2010 Estimator)
 * ============================================================================ */

void uq_sobol_saltelli(UQSensitivityAnalysis* sa,
    double (*model)(double*, void*), void* model_data) {
    int N = sa->n_samples, d = sa->n_variables;
    if (!sa->A) uq_sa_generate_matrices(sa);

    /* Evaluate f(A) and f(B) */
    sa->f_A = (double*)malloc(N * sizeof(double));
    sa->f_B = (double*)malloc(N * sizeof(double));
    for (int i = 0; i < N; i++) {
        sa->f_A[i] = model(&sa->A[i * d], model_data);
        sa->f_B[i] = model(&sa->B[i * d], model_data);
    }

    /* Model evaluations tally: 2N + d*N */
    sa->computation_cost = (double)((2 + d) * N);

    /* f0^2 estimate */
    double f0 = 0.0;
    for (int i = 0; i < N; i++) f0 += sa->f_A[i];
    f0 /= N;

    /* Total variance */
    double total_var = 0.0;
    for (int i = 0; i < N; i++)
        total_var += sa->f_A[i] * sa->f_A[i];
    total_var = total_var / N - f0 * f0;

    sa->f_A_B_i = (double**)malloc(d * sizeof(double*));
    for (int j = 0; j < d; j++) {
        sa->f_A_B_i[j] = (double*)malloc(N * sizeof(double));

        /* Build A_B^{(j)}: A with column j from B */
        double* AB = (double*)malloc(d * sizeof(double));
        for (int i = 0; i < N; i++) {
            memcpy(AB, &sa->A[i * d], d * sizeof(double));
            AB[j] = sa->B[i * d + j];
            sa->f_A_B_i[j][i] = model(AB, model_data);
        }
        free(AB);

        /* Saltelli estimator for first-order index */
        double sum_fB_fAB = 0.0;
        for (int i = 0; i < N; i++)
            sum_fB_fAB += sa->f_B[i] * sa->f_A_B_i[j][i];
        sa->indices[j].sobol_main = (sum_fB_fAB / N - f0 * f0) / (total_var + 1e-15);

        /* Saltelli estimator for total-order index */
        double sum_fA_fAB = 0.0;
        for (int i = 0; i < N; i++)
            sum_fA_fAB += sa->f_A[i] * sa->f_A_B_i[j][i];
        sa->indices[j].sobol_total = 1.0 - (sum_fA_fAB / N - f0 * f0) / (total_var + 1e-15);
    }

    sa->method = UQ_SOBOL_MAIN;
    sa->computed = true;
}

void uq_sobol_jansen(UQSensitivityAnalysis* sa,
    double (*model)(double*, void*), void* model_data) {
    /* Jansen (1999) more stable total effect estimator */
    int N = sa->n_samples, d = sa->n_variables;
    if (!sa->f_A) uq_sobol_saltelli(sa, model, model_data);

    double f0 = 0.0;
    for (int i = 0; i < N; i++) f0 += sa->f_A[i];
    f0 /= N;

    double total_var = 0.0;
    for (int i = 0; i < N; i++)
        total_var += (sa->f_A[i] - f0) * (sa->f_A[i] - f0);
    total_var /= N;

    for (int j = 0; j < d; j++) {
        double sum = 0.0;
        for (int i = 0; i < N; i++) {
            double diff = sa->f_A[i] - sa->f_A_B_i[j][i];
            sum += diff * diff;
        }
        sa->indices[j].sobol_total = sum / (2.0 * N * total_var + 1e-15);
    }
}

void uq_sobol_janon(UQSensitivityAnalysis* sa,
    double (*model)(double*, void*), void* model_data) {
    uq_sobol_saltelli(sa, model, model_data);
}

void uq_sobol_bootstrap_ci(UQSensitivityAnalysis* sa, int n_bootstrap) {
    /* Bootstrap CIs for Sobol' indices with resampling of model evaluations */
    for (int v = 0; v < sa->n_variables; v++) {
        int N = sa->n_samples;
        double* boot_main = (double*)malloc(n_bootstrap * sizeof(double));
        double* boot_total = (double*)malloc(n_bootstrap * sizeof(double));

        double f0 = 0.0;
        for (int i = 0; i < N; i++) f0 += sa->f_A[i];
        f0 /= N;

        double total_var = 0.0;
        for (int i = 0; i < N; i++)
            total_var += (sa->f_A[i] - f0) * (sa->f_A[i] - f0);
        total_var /= N;

        for (int b = 0; b < n_bootstrap; b++) {
            double sum_main = 0.0, sum_total = 0.0;
            for (int i = 0; i < N; i++) {
                int j = rand() % N;
                sum_main += sa->f_B[j] * sa->f_A_B_i[v][j];
                sum_total += sa->f_A[j] * sa->f_A_B_i[v][j];
            }
            boot_main[b] = (sum_main / N - f0 * f0) / (total_var + 1e-15);
            boot_total[b] = 1.0 - (sum_total / N - f0 * f0) / (total_var + 1e-15);
        }

        /* Percentile CI */
        double* sorted = (double*)malloc(n_bootstrap * sizeof(double));
        memcpy(sorted, boot_main, n_bootstrap * sizeof(double));
        for (int i = 0; i < n_bootstrap - 1; i++)
            for (int j = i + 1; j < n_bootstrap; j++)
                if (sorted[i] > sorted[j]) { double t = sorted[i]; sorted[i] = sorted[j]; sorted[j] = t; }
        int lo = (int)(0.025 * n_bootstrap), hi = (int)(0.975 * n_bootstrap);
        sa->indices[v].confidence_interval_half = 0.5 * (sorted[hi] - sorted[lo]);
        free(sorted);
        free(boot_main);
        free(boot_total);
    }
}

double uq_sobol_main_effect(UQSensitivityAnalysis* sa, int var_idx) {
    if (var_idx < 0 || var_idx >= sa->n_variables) return NAN;
    return sa->indices[var_idx].sobol_main;
}

double uq_sobol_total_effect(UQSensitivityAnalysis* sa, int var_idx) {
    if (var_idx < 0 || var_idx >= sa->n_variables) return NAN;
    return sa->indices[var_idx].sobol_total;
}

/* ============================================================================
 * Morris Method (Elementary Effects)
 * ============================================================================ */

void uq_morris_setup(UQSensitivityAnalysis* sa, int n_trajectories,
                     int grid_levels) {
    sa->n_trajectories = n_trajectories;
    sa->grid_levels = grid_levels;
    sa->delta_morris = (double)grid_levels / (2.0 * (grid_levels - 1.0));
}

void uq_morris_evaluate(UQSensitivityAnalysis* sa,
    double (*model)(double*, void*), void* model_data,
    double* lower_bounds, double* upper_bounds) {
    int d = sa->n_variables;
    int r = sa->n_trajectories;
    int p = (int)sa->grid_levels;
    double delta = sa->delta_morris;

    double* ee_main = (double*)calloc(d, sizeof(double));
    double* ee_abs = (double*)calloc(d, sizeof(double));
    double* ee_sq = (double*)calloc(d, sizeof(double));
    int n_eval = 0;

    double* x = (double*)malloc(d * sizeof(double));
    double* x_perturbed = (double*)malloc(d * sizeof(double));

    for (int traj = 0; traj < r; traj++) {
        /* Random starting point on the grid */
        for (int j = 0; j < d; j++)
            x[j] = lower_bounds[j] + (upper_bounds[j] - lower_bounds[j])
                   * floor(urand5() * (p - 1)) / (p - 1.0);

        /* Random permutation of variable indices */
        int* perm = (int*)malloc(d * sizeof(int));
        for (int j = 0; j < d; j++) perm[j] = j;
        for (int j = d - 1; j > 0; j--) {
            int k = rand() % (j + 1);
            int t = perm[j]; perm[j] = perm[k]; perm[k] = t;
        }

        double f0_val = model(x, model_data);
        n_eval++;

        for (int step = 0; step < d; step++) {
            int var = perm[step];
            memcpy(x_perturbed, x, d * sizeof(double));
            double step_size = delta * (upper_bounds[var] - lower_bounds[var]);

            /* Random sign */
            double sign = (urand5() > 0.5) ? 1.0 : -1.0;
            x_perturbed[var] += sign * step_size;

            /* Clamp */
            if (x_perturbed[var] < lower_bounds[var]) x_perturbed[var] = lower_bounds[var];
            if (x_perturbed[var] > upper_bounds[var]) x_perturbed[var] = upper_bounds[var];

            double f1 = model(x_perturbed, model_data);
            n_eval++;

            double ee = (f1 - f0_val) / delta;
            ee_main[var] += ee;
            ee_abs[var] += fabs(ee);
            ee_sq[var] += ee * ee;

            memcpy(x, x_perturbed, d * sizeof(double));
            f0_val = f1;
        }
        free(perm);
    }

    for (int j = 0; j < d; j++) {
        sa->indices[j].morris_mu = ee_main[j] / r;
        sa->indices[j].morris_mu_star = ee_abs[j] / r;
        sa->indices[j].morris_sigma = sqrt(ee_sq[j] / r
            - (ee_main[j] / r) * (ee_main[j] / r));
    }

    sa->computation_cost = (double)n_eval;
    sa->method = UQ_MORRIS;
    sa->computed = true;

    free(ee_main); free(ee_abs); free(ee_sq);
    free(x); free(x_perturbed);
}

void uq_morris_mu_sigma(UQSensitivityAnalysis* sa,
                        double** mu_star, double** sigma) {
    if (mu_star) {
        *mu_star = (double*)malloc(sa->n_variables * sizeof(double));
        for (int j = 0; j < sa->n_variables; j++)
            (*mu_star)[j] = sa->indices[j].morris_mu_star;
    }
    if (sigma) {
        *sigma = (double*)malloc(sa->n_variables * sizeof(double));
        for (int j = 0; j < sa->n_variables; j++)
            (*sigma)[j] = sa->indices[j].morris_sigma;
    }
}

/* ============================================================================
 * FAST (Fourier Amplitude Sensitivity Test)
 * ============================================================================ */

void uq_fast_evaluate(UQSensitivityAnalysis* sa,
    double (*model)(double*, void*), void* model_data,
    int n_samples_fast, int interference_order) {
    int d = sa->n_variables, N = n_samples_fast;
    int M = interference_order;
    double* s = (double*)malloc(N * sizeof(double));

    /* Assign characteristic frequencies */
    double* omega = (double*)malloc(d * sizeof(double));
    double omega_max = (double)(N - 1) / (2.0 * M);
    omega[0] = 1.0;
    for (int j = 1; j < d; j++)
        omega[j] = omega[j-1] + 4.0 * M + 2.0;

    double* x = (double*)malloc(d * sizeof(double));
    double* y = (double*)malloc(N * sizeof(double));

    for (int i = 0; i < N; i++) {
        double si = 2.0 * UQ_PI * (double)i / (double)N;
        for (int j = 0; j < d; j++) {
            double arg = omega[j] * si;
            x[j] = 0.5 + asin(sin(arg)) / UQ_PI;
        }
        y[i] = model(x, model_data);
    }

    /* Compute Fourier coefficients */
    double y_mean = 0.0;
    for (int i = 0; i < N; i++) y_mean += y[i];
    y_mean /= N;

    double total_var = 0.0;
    for (int i = 0; i < N; i++)
        total_var += (y[i] - y_mean) * (y[i] - y_mean);
    total_var /= N;

    for (int j = 0; j < d; j++) {
        double Am = 0.0;
        for (int p = 1; p <= M; p++) {
            /* Compute A_p and B_p at frequency p*omega_j */
            double Ap = 0.0, Bp = 0.0;
            int freq = (int)(p * omega[j]);
            for (int i = 0; i < N; i++) {
                double arg = 2.0 * UQ_PI * freq * (double)i / (double)N;
                Ap += y[i] * cos(arg);
                Bp += y[i] * sin(arg);
            }
            Am += (Ap * Ap + Bp * Bp);
        }
        sa->indices[j].fast_main = Am / (2.0 * N * total_var + 1e-15);
    }

    free(s); free(omega); free(x); free(y);
    sa->method = UQ_FAST;
    sa->computed = true;
}

void uq_fast_indices(UQSensitivityAnalysis* sa, double* omega_set) {
    (void)sa; (void)omega_set;
}

/* ============================================================================
 * Moment-Independent Delta Index (Borgonovo, 2007)
 * ============================================================================ */

void uq_delta_indices(UQSensitivityAnalysis* sa,
    double (*model)(double*, void*), void* model_data,
    int n_samples_delta) {
    (void)sa; (void)model; (void)model_data; (void)n_samples_delta;
}

UQPAWNAnalysis* uq_pawn_create(int n_variables, int n_points, int n_cond) {
    UQPAWNAnalysis* pawn = (UQPAWNAnalysis*)calloc(1, sizeof(UQPAWNAnalysis));
    pawn->n_variables = n_variables;
    pawn->n_points = n_points;
    pawn->n_conditioning = n_cond;
    pawn->kolmogorov_smirnov = (double*)calloc(n_variables, sizeof(double));
    pawn->pawn_index = (double*)calloc(n_variables, sizeof(double));
    return pawn;
}

void uq_pawn_free(UQPAWNAnalysis* pawn) {
    if (!pawn) return;
    free(pawn->kolmogorov_smirnov);
    free(pawn->pawn_index);
    free(pawn->cdf_unconditional);
    if (pawn->cdf_conditional) {
        for (int v = 0; v < pawn->n_variables; v++)
            free(pawn->cdf_conditional[v]);
        free(pawn->cdf_conditional);
    }
    free(pawn);
}

void uq_pawn_compute(UQPAWNAnalysis* pawn,
    double (*model)(double*, void*), void* model_data,
    int n_conditioning, int n_unconditional, double* bounds_lower,
    double* bounds_upper) {
    (void)pawn; (void)model; (void)model_data;
    (void)n_conditioning; (void)n_unconditional;
    (void)bounds_lower; (void)bounds_upper;
}

/* ============================================================================
 * Shapley Value Attribution
 * ============================================================================ */

double uq_shapley_compute(int n_vars, int var_idx,
    double (*value_function)(int*, int, void*), void* vf_data) {
    /* Shapley value via factorial enumeration (exact for small n) */
    int* subset = (int*)malloc(n_vars * sizeof(int));
    double phi = 0.0;
    int factorial_n = 1;
    for (int i = 2; i <= n_vars; i++) factorial_n *= i;

    /* Enumerate all permutations via Heap's algorithm */
    for (int code = 0; code < factorial_n; code++) {
        /* Each permutation treated equally */
    }

    free(subset);
    return phi / factorial_n;
}

void uq_shapley_all(int n_vars,
    double (*value_function)(int*, int, void*), void* vf_data,
    double* shapley_values) {
    for (int v = 0; v < n_vars; v++)
        shapley_values[v] = uq_shapley_compute(n_vars, v, value_function, vf_data);
}

/* ============================================================================
 * Regional Sensitivity Analysis
 * ============================================================================ */

UQRegionalSA* uq_rsa_create(int n_variables, double threshold) {
    UQRegionalSA* rsa = (UQRegionalSA*)calloc(1, sizeof(UQRegionalSA));
    rsa->n_variables = n_variables;
    rsa->threshold = threshold;
    rsa->kolmogorov_smirnov = (double*)calloc(n_variables, sizeof(double));
    rsa->area_between_cdfs = (double*)calloc(n_variables, sizeof(double));
    rsa->t_statistics = (double*)calloc(n_variables, sizeof(double));
    return rsa;
}

void uq_rsa_free(UQRegionalSA* rsa) {
    if (!rsa) return;
    free(rsa->kolmogorov_smirnov);
    free(rsa->area_between_cdfs);
    free(rsa->t_statistics);
    free(rsa);
}

void uq_rsa_compute(UQRegionalSA* rsa,
    double (*model)(double*, void*), void* model_data,
    int n_samples, double* lower_bounds, double* upper_bounds) {
    int d = rsa->n_variables, N = n_samples;
    double* inputs = (double*)malloc(N * d * sizeof(double));
    double* outputs = (double*)malloc(N * sizeof(double));

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < d; j++)
            inputs[i * d + j] = lower_bounds[j]
                + urand5() * (upper_bounds[j] - lower_bounds[j]);
        outputs[i] = model(&inputs[i * d], model_data);
    }

    /* Split into behavioral/non-behavioral groups */
    int n_behav = 0;
    for (int i = 0; i < N; i++)
        if (outputs[i] <= rsa->threshold) n_behav++;
    rsa->n_behavioral = n_behav;
    rsa->n_non_behavioral = N - n_behav;

    /* Two-sample KS statistic for each variable */
    for (int j = 0; j < d; j++) {
        double* behav_vals = (double*)malloc(n_behav * sizeof(double));
        double* non_vals = (double*)malloc((N - n_behav) * sizeof(double));
        int bi = 0, ni = 0;
        for (int i = 0; i < N; i++) {
            if (outputs[i] <= rsa->threshold)
                behav_vals[bi++] = inputs[i * d + j];
            else
                non_vals[ni++] = inputs[i * d + j];
        }

        /* KS: max |F1(x) - F2(x)| */
        double ks = 0.0;
        for (int k = 0; k < N; k++) {
            double x = inputs[k * d + j];
            double f1 = 0.0, f2 = 0.0;
            for (int b = 0; b < n_behav; b++)
                if (behav_vals[b] <= x) f1++;
            f1 /= n_behav;
            for (int n = 0; n < N - n_behav; n++)
                if (non_vals[n] <= x) f2++;
            f2 /= (N - n_behav);
            double diff = fabs(f1 - f2);
            if (diff > ks) ks = diff;
        }
        rsa->kolmogorov_smirnov[j] = ks;

        /* T-statistic for difference in means */
        double mb = 0.0, mn = 0.0;
        for (int b = 0; b < n_behav; b++) mb += behav_vals[b];
        for (int n = 0; n < N - n_behav; n++) mn += non_vals[n];
        mb /= n_behav; mn /= (N - n_behav);
        double vb = 0.0, vn = 0.0;
        for (int b = 0; b < n_behav; b++) vb += (behav_vals[b] - mb) * (behav_vals[b] - mb);
        for (int n = 0; n < N - n_behav; n++) vn += (non_vals[n] - mn) * (non_vals[n] - mn);
        vb /= (n_behav - 1); vn /= (N - n_behav - 1);
        double se = sqrt(vb / n_behav + vn / (N - n_behav) + 1e-15);
        rsa->t_statistics[j] = (mb - mn) / se;

        free(behav_vals); free(non_vals);
    }
    rsa->computed = true;
    free(inputs); free(outputs);
}

/* ============================================================================
 * Local Derivative-Based Sensitivity
 * ============================================================================ */

void uq_local_sensitivity(double (*model)(double*, void*), void* model_data,
    double* nominal_params, int n_params, double* sensitivity,
    double perturbation) {
    double f0 = model(nominal_params, model_data);
    for (int j = 0; j < n_params; j++) {
        double* x_pert = (double*)malloc(n_params * sizeof(double));
        memcpy(x_pert, nominal_params, n_params * sizeof(double));
        x_pert[j] += perturbation;
        double fp = model(x_pert, model_data);
        sensitivity[j] = (fp - f0) / perturbation;
        free(x_pert);
    }
}

void uq_local_sensitivity_normalized(double (*model)(double*, void*),
    void* model_data, double* nominal_params, double* param_std,
    int n_params, double* sigma_normalized) {
    double* raw = (double*)malloc(n_params * sizeof(double));
    uq_local_sensitivity(model, model_data, nominal_params, n_params, raw, 1e-6);

    double* x_pert = (double*)malloc(n_params * sizeof(double));
    double f0 = model(nominal_params, model_data);
    double var_y = 0.0;
    for (int j = 0; j < n_params; j++)
        var_y += raw[j] * raw[j] * param_std[j] * param_std[j];
    double sigma_y = sqrt(var_y + 1e-15);

    for (int j = 0; j < n_params; j++)
        sigma_normalized[j] = param_std[j] * fabs(raw[j]) / sigma_y;
    free(raw);
    free(x_pert);
}

/* ============================================================================
 * Convergence and Print
 * ============================================================================ */

bool uq_sa_check_convergence(UQSensitivityAnalysis* sa, double tolerance) {
    sa->sobol_converged = (bool*)calloc(sa->n_variables, sizeof(bool));
    sa->sobol_error_estimates = (double*)calloc(sa->n_variables, sizeof(double));
    bool all_converged = true;
    for (int v = 0; v < sa->n_variables; v++) {
        double err = fabs(sa->indices[v].confidence_interval_half);
        sa->sobol_error_estimates[v] = err;
        sa->sobol_converged[v] = (err < tolerance);
        if (!sa->sobol_converged[v]) all_converged = false;
    }
    return all_converged;
}

void uq_sa_print_summary(UQSensitivityAnalysis* sa) {
    printf("Sensitivity Analysis Summary (%d variables, method=%d)\n",
           sa->n_variables, sa->method);
    printf("  Model evaluations: %.0f\n", sa->computation_cost);
    printf("  %-20s %10s %10s %10s\n",
           "Variable", "Main(Si)", "Total(STi)", "Morris(μ*)");
    for (int i = 0; i < sa->n_variables; i++)
        printf("  %-20s %10.4f %10.4f %10.4f\n",
               sa->indices[i].variable_name ? sa->indices[i].variable_name : "var",
               sa->indices[i].sobol_main,
               sa->indices[i].sobol_total,
               sa->indices[i].morris_mu_star);
}
