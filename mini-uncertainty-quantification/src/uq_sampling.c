#include "uq_sampling.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define UQ_PI 3.14159265358979323846

static double urand3(void) {
    return ((double)rand() / ((double)RAND_MAX + 1.0));
}

static double gauss_rand3(void) {
    double u1, u2;
    do { u1 = urand3(); } while (u1 < 1e-15);
    u2 = urand3();
    return sqrt(-2.0 * log(u1)) * cos(2.0 * UQ_PI * u2);
}

static double gauss_cdf_approx_local(double x) {
    double t = 1.0 / (1.0 + 0.2316419 * fabs(x));
    double b[] = {0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429};
    double poly = t * (b[0] + t * (b[1] + t * (b[2] + t * (b[3] + t * b[4]))));
    double phi = 1.0 - (1.0 / sqrt(2.0 * UQ_PI)) * exp(-0.5 * x * x) * poly;
    return (x >= 0.0) ? phi : 1.0 - phi;
}

/* ============================================================================
 * Core Sampler
 * ============================================================================ */

UQSampler* uq_sampler_create(UQMCStrategy strategy, int n_samples, int n_dims) {
    UQSampler* s = (UQSampler*)calloc(1, sizeof(UQSampler));
    s->strategy = strategy;
    s->n_samples = n_samples;
    s->n_dimensions = n_dims;
    s->samples = (double**)malloc(n_samples * sizeof(double*));
    for (int i = 0; i < n_samples; i++)
        s->samples[i] = (double*)calloc(n_dims, sizeof(double));
    return s;
}

void uq_sampler_free(UQSampler* sampler) {
    if (!sampler) return;
    for (int i = 0; i < sampler->n_samples; i++)
        free(sampler->samples[i]);
    free(sampler->samples);
    free(sampler->prime_bases);
    free(sampler->sobol_direction_numbers);
    free(sampler->lhs_permutation);
    free(sampler);
}

void uq_sampler_set_marginals(UQSampler* sampler, UQDistribution** dists) {
    sampler->marginals = dists;
}

void uq_sampler_set_correlation(UQSampler* sampler, UQMatrix* corr) {
    sampler->correlation = corr;
    sampler->is_correlated = true;
}

void uq_sampler_generate(UQSampler* sampler) {
    int n = sampler->n_samples, d = sampler->n_dimensions;
    switch (sampler->strategy) {
    case UQ_MC_SIMPLE:
        for (int i = 0; i < n; i++)
            for (int j = 0; j < d; j++)
                sampler->samples[i][j] = gauss_rand3();
        break;
    case UQ_MC_LATIN_HYPERCUBE:
        uq_lhs_generate(n, d, NULL, true);
        break;
    case UQ_MC_SOBOL:
        uq_sobol_sequence(n, d, 0, sampler->samples);
        break;
    case UQ_MC_HALTON:
        uq_halton_sequence(n, d, sampler->samples);
        break;
    case UQ_MC_HAMMERSLEY:
        uq_hammersley_sequence(n, d, sampler->samples);
        break;
    default:
        for (int i = 0; i < n; i++)
            for (int j = 0; j < d; j++)
                sampler->samples[i][j] = urand3();
    }

    /* Transform to marginal distributions if specified */
    if (sampler->marginals)
        for (int i = 0; i < n; i++)
            for (int j = 0; j < d; j++) {
                double u = (sampler->strategy == UQ_MC_SIMPLE)
                    ? 0.5 * (1.0 + erf(sampler->samples[i][j] / sqrt(2.0)))
                    : sampler->samples[i][j];
                sampler->samples[i][j] = uq_dist_quantile(sampler->marginals[j], u);
            }

    /* Impose correlation via Iman-Conover if needed */
    if (sampler->is_correlated && sampler->correlation) {
        uq_lhs_imam_conover(sampler->samples, n, d, sampler->correlation);
    }
}

double* uq_sampler_get_sample(UQSampler* sampler, int idx) {
    if (idx < 0 || idx >= sampler->n_samples) return NULL;
    return sampler->samples[idx];
}

/* ============================================================================
 * Quasi-Monte Carlo Sequences
 * ============================================================================ */

static int primes_10[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53};

static double halton_single(int index, int base) {
    double result = 0.0, f = 1.0 / (double)base;
    int i = index + 1;
    while (i > 0) {
        result += f * (double)(i % base);
        i /= base;
        f /= (double)base;
    }
    return result;
}

void uq_halton_sequence(int n, int dim, double** out) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < dim; j++)
            out[i][j] = halton_single(i, primes_10[j % 16]);
}

void uq_hammersley_sequence(int n, int dim, double** out) {
    for (int i = 0; i < n; i++) {
        out[i][0] = ((double)i + 0.5) / (double)n;
        for (int j = 1; j < dim; j++)
            out[i][j] = halton_single(i, primes_10[j - 1]);
    }
}

/* Direction numbers for Sobol' (Joe & Kuo, 2008) — first 10 dimensions */
static unsigned long long sobol_v[10][32] = {
    {1ULL<<31, 0}, {1ULL<<31, 1ULL<<30, 0},
    {1ULL<<31, 3ULL<<30, 3ULL<<29, 0},
    {1ULL<<31, 1ULL<<30, 1ULL<<29, 0},
    {1ULL<<31, 1ULL<<30, 1ULL<<29, 1ULL<<28, 0},
    {1ULL<<31, 2ULL<<30, 1ULL<<29, 3ULL<<27, 3ULL<<26, 0},
    {1ULL<<31, 3ULL<<30, 3ULL<<29, 0},
    {1ULL<<31, 1ULL<<30, 3ULL<<28, 2ULL<<27, 0},
    {1ULL<<31, 1ULL<<30, 3ULL<<29, 3ULL<<27, 2ULL<<26, 0},
    {1ULL<<31, 2ULL<<30, 1ULL<<29, 2ULL<<27, 2ULL<<26, 0}
};

void uq_sobol_sequence(int n, int dim, unsigned long long skip, double** out) {
    int L = 32; /* Number of bits */
    unsigned long long* x = (unsigned long long*)calloc(dim, sizeof(unsigned long long));
    unsigned long long* v = (unsigned long long*)calloc(dim * L, sizeof(unsigned long long));

    for (int d = 0; d < dim && d < 10; d++)
        for (int l = 0; l < L; l++)
            v[d * L + l] = sobol_v[d][l];

    for (unsigned long long k = 0; k < skip + n; k++) {
        /* Find rightmost zero bit */
        int c = 0;
        unsigned long long kk = k;
        while (kk & 1) { c++; kk >>= 1; }

        for (int d = 0; d < dim; d++) {
            if (c < L)
                x[d] ^= v[d * L + c];
            if (k >= skip)
                out[k - skip][d] = (double)x[d] / (double)(1ULL << 32);
        }
    }
    free(x); free(v);
}

/* ============================================================================
 * Latin Hypercube Sampling
 * ============================================================================ */

void uq_lhs_generate(int n, int dim, double** out, bool randomize) {
    double** data = out;
    int alloc_self = 0;
    if (!data) {
        data = (double**)malloc(n * sizeof(double*));
        for (int i = 0; i < n; i++)
            data[i] = (double*)malloc(dim * sizeof(double));
        alloc_self = 1;
    }

    for (int j = 0; j < dim; j++) {
        /* Random permutation of {0, 1, ..., n-1} */
        int* perm = (int*)malloc(n * sizeof(int));
        for (int i = 0; i < n; i++) perm[i] = i;
        for (int i = n - 1; i > 0; i--) {
            int r = rand() % (i + 1);
            int t = perm[i]; perm[i] = perm[r]; perm[r] = t;
        }
        for (int i = 0; i < n; i++) {
            double base = (double)perm[i] / (double)n;
            double jitter = randomize ? urand3() / (double)n : 0.5 / (double)n;
            data[i][j] = base + jitter;
            if (data[i][j] > 1.0) data[i][j] = 1.0;
            if (data[i][j] < 0.0) data[i][j] = 0.0;
        }
        free(perm);
    }
    if (alloc_self) {
        /* Copy to out if provided externally */
        if (out) {
            for (int i = 0; i < n; i++)
                memcpy(out[i], data[i], dim * sizeof(double));
        }
        for (int i = 0; i < n; i++) free(data[i]);
        free(data);
    }
}

void uq_lhs_midpoint(int n, int dim, double** out) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < dim; j++) {
            int* perm = (int*)malloc(n * sizeof(int));
            for (int k = 0; k < n; k++) perm[k] = k;
            for (int k = n - 1; k > 0; k--) {
                int r = rand() % (k + 1);
                int t = perm[k]; perm[k] = perm[r]; perm[r] = t;
            }
            out[i][j] = ((double)perm[i] + 0.5) / (double)n;
            free(perm);
        }
}

void uq_lhs_imam_conover(double** lhs, int n, int dim, UQMatrix* target_corr) {
    /* Iman-Conover (1982) method for inducing rank correlation */
    if (!lhs || !target_corr) return;

    /* Cholesky of target correlation */
    UQMatrix* L_target = uq_matrix_create(dim, dim);
    uq_matrix_cholesky(L_target, target_corr);

    /* Generate N(dim) independent normals */
    double** Z = (double**)malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++) {
        Z[i] = (double*)malloc(dim * sizeof(double));
        for (int j = 0; j < dim; j++) Z[i][j] = gauss_rand3();
    }

    /* Compute empirical correlation of Z */
    double* means = (double*)calloc(dim, sizeof(double));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < dim; j++)
            means[j] += Z[i][j];
    for (int j = 0; j < dim; j++) means[j] /= n;

    UQMatrix* emp_corr = uq_matrix_create(dim, dim);
    for (int a = 0; a < dim; a++)
        for (int b = 0; b < dim; b++) {
            double cov = 0.0, va = 0.0, vb = 0.0;
            for (int i = 0; i < n; i++) {
                double da = Z[i][a] - means[a];
                double db = Z[i][b] - means[b];
                cov += da * db; va += da * da; vb += db * db;
            }
            uq_matrix_set(emp_corr, a, b, cov / sqrt(va * vb + 1e-15));
        }

    UQMatrix* L_emp = uq_matrix_create(dim, dim);
    uq_matrix_cholesky(L_emp, emp_corr);

    /* Transform: Z' = Z * L_emp^{-1} * L_target */
    UQMatrix* L_emp_inv = uq_matrix_create(dim, dim);
    uq_matrix_invert(L_emp_inv, L_emp);
    UQMatrix* T = uq_matrix_create(dim, dim);
    uq_matrix_multiply(T, L_emp_inv, L_target);

    double* Z_new = (double*)malloc(n * dim * sizeof(double));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < dim; j++) {
            double s = 0.0;
            for (int k = 0; k < dim; k++)
                s += Z[i][k] * uq_matrix_get(T, k, j);
            Z_new[i * dim + j] = s;
        }

    /* Reorder LHS columns to match ranks of Z_new */
    int* rank = (int*)malloc(n * sizeof(int));
    for (int j = 0; j < dim; j++) {
        /* Get sort order for Z_new[:, j] */
        int* order = (int*)malloc(n * sizeof(int));
        for (int i = 0; i < n; i++) order[i] = i;
        /* Simple insertion sort for rank */
        for (int i = 0; i < n; i++) {
            rank[i] = 0;
            for (int k = 0; k < n; k++)
                if (Z_new[i * dim + j] > Z_new[k * dim + j]) rank[i]++;
        }
        /* Reorder LHS column */
        double* col = (double*)malloc(n * sizeof(double));
        for (int i = 0; i < n; i++) col[rank[i]] = lhs[i][j];
        for (int i = 0; i < n; i++) lhs[i][j] = col[i];
        free(col);
        free(order);
    }

    uq_matrix_free(L_target);
    uq_matrix_free(emp_corr);
    uq_matrix_free(L_emp);
    uq_matrix_free(L_emp_inv);
    uq_matrix_free(T);
    free(Z_new);
    free(means);
    free(rank);
    for (int i = 0; i < n; i++) free(Z[i]);
    free(Z);
}

void uq_orthogonal_array_lhs(int n_levels, int dim, double** out) {
    /* Simple OA-based LHS: uses prime base n_levels */
    for (int i = 0; i < n_levels * n_levels && i < n_levels * dim; i++)
        for (int j = 0; j < dim; j++)
            out[i][j] = ((double)((i + j * (i / n_levels)) % n_levels) + urand3())
                        / (double)n_levels;
}

/* ============================================================================
 * Importance Sampling
 * ============================================================================ */

double uq_importance_sampling(double (*f)(double*, void*), void* f_data,
    UQDistribution* proposal, double (*target_log_pdf)(double*, void*),
    void* target_data, int n_samples, int dim, double* mc_error) {
    double sum_wf = 0.0, sum_w = 0.0, sum_wf2 = 0.0;
    for (int s = 0; s < n_samples; s++) {
        double* x = (double*)malloc(dim * sizeof(double));
        for (int d = 0; d < dim; d++)
            x[d] = uq_dist_sample(proposal);

        double f_val = f(x, f_data);
        double log_target = target_log_pdf(x, target_data);
        double log_proposal = uq_dist_log_pdf(proposal, x[0]);
        for (int d = 1; d < dim; d++)
            log_proposal += uq_dist_log_pdf(proposal, x[d]);

        double w = exp(log_target - log_proposal);
        sum_wf += w * f_val;
        sum_w += w;
        sum_wf2 += w * f_val * f_val;

        free(x);
    }
    double estimate = sum_wf / (sum_w + 1e-15);
    if (mc_error) {
        double var_wf = sum_wf2 / n_samples - estimate * estimate;
        *mc_error = sqrt(var_wf / n_samples);
    }
    return estimate;
}

/* ============================================================================
 * Rejection Sampling
 * ============================================================================ */

int uq_rejection_sampling(double (*target_pdf)(double*, void*), void* target_data,
    UQDistribution* proposal, double M, int n_desired, int dim, double** out) {
    int accepted = 0;
    int total = 0;
    int max_total = n_desired * 20;
    while (accepted < n_desired && total < max_total) {
        double* x = (double*)malloc(dim * sizeof(double));
        for (int d = 0; d < dim; d++)
            x[d] = uq_dist_sample(proposal);

        double u = urand3();
        double prop_pdf = uq_dist_pdf(proposal, x[0]);
        for (int d = 1; d < dim; d++)
            prop_pdf *= uq_dist_pdf(proposal, x[d]);

        double target_p = target_pdf(x, target_data);

        if (u * M * prop_pdf < target_p) {
            if (out && accepted < n_desired)
                memcpy(out[accepted], x, dim * sizeof(double));
            accepted++;
        }
        total++;
        free(x);
    }
    return accepted;
}

/* ============================================================================
 * Stratified Sampling
 * ============================================================================ */

void uq_stratified_sample(UQDistribution** strata, double* strata_weights,
    int n_strata, int* n_per_stratum, int dim, double** out) {
    int idx = 0;
    for (int s = 0; s < n_strata; s++) {
        for (int i = 0; i < n_per_stratum[s]; i++) {
            for (int d = 0; d < dim; d++)
                out[idx][d] = uq_dist_sample(strata[s]);
            idx++;
        }
    }
    (void)strata_weights;
}

/* ============================================================================
 * Antithetic Variates
 * ============================================================================ */

void uq_antithetic_sample(int n_pairs, int dim, double** out) {
    for (int i = 0; i < n_pairs; i++) {
        (void)urand3(); /* Common uniform for all dims */
        for (int d = 0; d < dim; d++) {
            out[2*i][d] = gauss_rand3();
            out[2*i+1][d] = -out[2*i][d];
        }
    }
}

/* ============================================================================
 * Control Variates
 * ============================================================================ */

double uq_control_variate_estimate(double* f_vals, double* c_vals,
    double true_mean_c, int n) {
    /* Estimate: μ_f ≈ f_bar - β*(c_bar - μ_c) with optimal β = Cov(f,c)/Var(c) */
    double f_bar = 0.0, c_bar = 0.0;
    for (int i = 0; i < n; i++) { f_bar += f_vals[i]; c_bar += c_vals[i]; }
    f_bar /= n; c_bar /= n;

    double var_c = 0.0, cov_fc = 0.0;
    for (int i = 0; i < n; i++) {
        double dc = c_vals[i] - c_bar;
        double df = f_vals[i] - f_bar;
        var_c += dc * dc;
        cov_fc += df * dc;
    }
    var_c /= n; cov_fc /= n;
    double beta = cov_fc / (var_c + 1e-15);
    return f_bar - beta * (c_bar - true_mean_c);
}

/* ============================================================================
 * Bootstrap Methods
 * ============================================================================ */

UQBootstrap* uq_bootstrap_create(UQBootstrapMethod method, int n_original,
                                 int n_bootstrap, int dim) {
    UQBootstrap* bs = (UQBootstrap*)calloc(1, sizeof(UQBootstrap));
    bs->method = method;
    bs->n_original = n_original;
    bs->n_bootstrap = n_bootstrap;
    bs->n_dim = dim;
    bs->original_data = (double*)calloc(n_original * dim, sizeof(double));
    bs->bootstrap_replicates = (double**)malloc(n_bootstrap * sizeof(double*));
    for (int i = 0; i < n_bootstrap; i++)
        bs->bootstrap_replicates[i] = (double*)malloc(n_original * dim * sizeof(double));
    bs->statistic_replicates = (double*)calloc(n_bootstrap, sizeof(double));
    bs->ci_percentile = (double*)malloc(2 * sizeof(double));
    bs->ci_basic = (double*)malloc(2 * sizeof(double));
    bs->ci_bca = (double*)malloc(2 * sizeof(double));
    return bs;
}

void uq_bootstrap_free(UQBootstrap* bs) {
    if (!bs) return;
    free(bs->original_data);
    for (int i = 0; i < bs->n_bootstrap; i++)
        free(bs->bootstrap_replicates[i]);
    free(bs->bootstrap_replicates);
    free(bs->statistic_replicates);
    free(bs->ci_percentile);
    free(bs->ci_basic);
    free(bs->ci_bca);
    free(bs);
}

void uq_bootstrap_set_data(UQBootstrap* bs, double* data) {
    memcpy(bs->original_data, data, bs->n_original * bs->n_dim * sizeof(double));
}

void uq_bootstrap_set_block_length(UQBootstrap* bs, int block_len) {
    bs->block_length = block_len;
}

void uq_bootstrap_generate_replicates(UQBootstrap* bs) {
    int n = bs->n_original;
    int dim = bs->n_dim;

    for (int b = 0; b < bs->n_bootstrap; b++) {
        if (bs->method == UQ_BOOTSTRAP_STANDARD) {
            /* Non-parametric bootstrap: resample with replacement */
            for (int i = 0; i < n; i++) {
                int j = rand() % n;
                memcpy(&bs->bootstrap_replicates[b][i * dim],
                       &bs->original_data[j * dim], dim * sizeof(double));
            }
        } else if (bs->method == UQ_BOOTSTRAP_BLOCK) {
            /* Moving block bootstrap */
            int blen = (bs->block_length > 0) ? bs->block_length : (int)sqrt((double)n);
            int n_blocks = (n + blen - 1) / blen;
            int idx = 0;
            for (int k = 0; k < n_blocks && idx < n; k++) {
                int start = rand() % (n - blen + 1);
                for (int t = 0; t < blen && idx < n; t++, idx++)
                    memcpy(&bs->bootstrap_replicates[b][idx * dim],
                           &bs->original_data[(start + t) * dim], dim * sizeof(double));
            }
        } else if (bs->method == UQ_BOOTSTRAP_RESIDUAL) {
            memcpy(bs->bootstrap_replicates[b], bs->original_data,
                   n * dim * sizeof(double));
        }
    }
}

void uq_bootstrap_compute_statistics(UQBootstrap* bs,
    double (*statistic)(double*, int, void*), void* stat_data) {
    /* Original statistic */
    bs->original_statistic = statistic(bs->original_data, bs->n_original, stat_data);

    /* Bootstrap replicates */
    double sum = 0.0;
    for (int b = 0; b < bs->n_bootstrap; b++) {
        double t = statistic(bs->bootstrap_replicates[b], bs->n_original, stat_data);
        bs->statistic_replicates[b] = t;
        sum += t;
    }
    bs->bootstrap_mean = sum / bs->n_bootstrap;
    bs->bootstrap_bias = bs->bootstrap_mean - bs->original_statistic;

    /* Bootstrap SE */
    double s2 = 0.0;
    for (int b = 0; b < bs->n_bootstrap; b++) {
        double d = bs->statistic_replicates[b] - bs->bootstrap_mean;
        s2 += d * d;
    }
    bs->bootstrap_se = sqrt(s2 / (bs->n_bootstrap - 1.0));
}

void uq_bootstrap_ci_percentile(UQBootstrap* bs, double confidence) {
    uq_ci_from_bootstrap(bs->statistic_replicates, bs->n_bootstrap,
        confidence, NULL);
    /* Copy results */
    double alpha = 1.0 - confidence;
    double* sorted = (double*)malloc(bs->n_bootstrap * sizeof(double));
    memcpy(sorted, bs->statistic_replicates, bs->n_bootstrap * sizeof(double));
    for (int i = 0; i < bs->n_bootstrap - 1; i++)
        for (int j = i + 1; j < bs->n_bootstrap; j++)
            if (sorted[i] > sorted[j]) { double t = sorted[i]; sorted[i] = sorted[j]; sorted[j] = t; }
    int lo = (int)(alpha * 0.5 * bs->n_bootstrap);
    int hi = (int)((1.0 - alpha * 0.5) * bs->n_bootstrap - 1);
    bs->ci_percentile[0] = sorted[lo];
    bs->ci_percentile[1] = sorted[hi];
    free(sorted);
}

void uq_bootstrap_ci_basic(UQBootstrap* bs, double confidence) {
    uq_bootstrap_ci_percentile(bs, confidence);
    double lo = 2.0 * bs->original_statistic - bs->ci_percentile[1];
    double hi = 2.0 * bs->original_statistic - bs->ci_percentile[0];
    bs->ci_basic[0] = lo;
    bs->ci_basic[1] = hi;
}

void uq_bootstrap_ci_bca(UQBootstrap* bs, double confidence,
                         double (*jackknife_stat)(double*, int, int, void*),
                         void* jack_data) {
    /* BCa interval (Efron, 1987) */
    int n = bs->n_original;
    /* Acceleration factor */
    double* jack_vals = (double*)malloc(n * sizeof(double));
    for (int i = 0; i < n; i++)
        jack_vals[i] = jackknife_stat(bs->original_data, n, i, jack_data);
    double jack_mean = 0.0;
    for (int i = 0; i < n; i++) jack_mean += jack_vals[i];
    jack_mean /= n;
    double num = 0.0, den = 0.0;
    for (int i = 0; i < n; i++) {
        double d = jack_mean - jack_vals[i];
        num += d * d * d;
        den += d * d;
    }
    double a_hat = num / (6.0 * pow(den, 1.5) + 1e-15);

    /* Bias correction */
    int count = 0;
    for (int b = 0; b < bs->n_bootstrap; b++)
        if (bs->statistic_replicates[b] <= bs->original_statistic) count++;
    double z0 = uq_stats_normal_quantile((double)count / bs->n_bootstrap);

    double alpha = 1.0 - confidence;
    double z_alpha = uq_stats_normal_quantile(alpha * 0.5);
    double z_1_alpha = uq_stats_normal_quantile(1.0 - alpha * 0.5);

    double alpha1 = gauss_cdf_approx_local(z0 + (z0 + z_alpha) / (1.0 - a_hat * (z0 + z_alpha)));
    double alpha2 = gauss_cdf_approx_local(z0 + (z0 + z_1_alpha) / (1.0 - a_hat * (z0 + z_1_alpha)));

    double* sorted = (double*)malloc(bs->n_bootstrap * sizeof(double));
    memcpy(sorted, bs->statistic_replicates, bs->n_bootstrap * sizeof(double));
    for (int i = 0; i < bs->n_bootstrap - 1; i++)
        for (int j = i + 1; j < bs->n_bootstrap; j++)
            if (sorted[i] > sorted[j]) { double t = sorted[i]; sorted[i] = sorted[j]; sorted[j] = t; }

    int lo = (int)(alpha1 * bs->n_bootstrap);
    int hi = (int)(alpha2 * bs->n_bootstrap);
    if (lo < 0) lo = 0;
    if (hi >= bs->n_bootstrap) hi = bs->n_bootstrap - 1;
    bs->ci_bca[0] = sorted[lo];
    bs->ci_bca[1] = sorted[hi];
    free(sorted);
    free(jack_vals);
}

void uq_bootstrap_ci_studentized(UQBootstrap* bs, double confidence,
    double (*stat)(double*, int, void*),
    double (*se_func)(double*, int, void*), void* aux_data) {
    int n = bs->n_original, B = bs->n_bootstrap;
    double* t_star = (double*)malloc(B * sizeof(double));
    double orig_se = se_func(bs->original_data, n, aux_data);

    for (int b = 0; b < B; b++) {
        double t = stat(bs->bootstrap_replicates[b], n, aux_data);
        double se = se_func(bs->bootstrap_replicates[b], n, aux_data);
        t_star[b] = (se > 1e-15) ? (t - bs->original_statistic) / se : 0.0;
    }
    /* Percentile interval on t* */
    double* sorted = (double*)malloc(B * sizeof(double));
    memcpy(sorted, t_star, B * sizeof(double));
    for (int i = 0; i < B - 1; i++)
        for (int j = i + 1; j < B; j++)
            if (sorted[i] > sorted[j]) { double t = sorted[i]; sorted[i] = sorted[j]; sorted[j] = t; }
    double alpha = 1.0 - confidence;
    int lo = (int)(alpha * 0.5 * B), hi = (int)((1.0 - alpha * 0.5) * B - 1);
    bs->ci_basic[0] = bs->original_statistic - sorted[hi] * orig_se;
    bs->ci_basic[1] = bs->original_statistic - sorted[lo] * orig_se;
    free(t_star);
    free(sorted);
}

void uq_bayesian_bootstrap(int n, int n_replicates, double** weights_out) {
    /* Rubin (1981): Dirichlet(1,1,...,1) weights */
    for (int r = 0; r < n_replicates; r++) {
        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            /* Generate Gamma(1,1) = Exp(1) */
            weights_out[r][i] = -log(urand3() + 1e-15);
            sum += weights_out[r][i];
        }
        for (int i = 0; i < n; i++)
            weights_out[r][i] /= sum;
    }
}

/* ============================================================================
 * Gaussian Process
 * ============================================================================ */

static double gp_kernel_compute(UQGPKernelType type, double* x1, double* x2,
                                int dim, double* params) {
    double r2 = 0.0;
    for (int d = 0; d < dim; d++) {
        double dx = (x1[d] - x2[d]) / params[d];
        r2 += dx * dx;
    }
    double r = sqrt(r2);
    double sf = params[dim];      /* Signal variance */
    switch (type) {
    case UQ_GP_KERNEL_SQUARED_EXPONENTIAL:
        return sf * sf * exp(-0.5 * r2);
    case UQ_GP_KERNEL_MATERN32:
        return sf * sf * (1.0 + sqrt(3.0) * r) * exp(-sqrt(3.0) * r);
    case UQ_GP_KERNEL_MATERN52:
        return sf * sf * (1.0 + sqrt(5.0) * r + 5.0 * r2 / 3.0) * exp(-sqrt(5.0) * r);
    case UQ_GP_KERNEL_EXPONENTIAL:
        return sf * sf * exp(-r);
    case UQ_GP_KERNEL_RATIONAL_QUADRATIC:
        return sf * sf * pow(1.0 + r2 / (2.0 * 0.5), -0.5); /* alpha=0.5 example */
    default:
        return sf * sf * exp(-0.5 * r2);
    }
}

UQGaussianProcess* uq_gp_create(UQGPKernelType kernel, int n_inputs,
                                int n_train) {
    UQGaussianProcess* gp = (UQGaussianProcess*)calloc(1, sizeof(UQGaussianProcess));
    gp->kernel_type = kernel;
    gp->n_inputs = n_inputs;
    gp->n_train = n_train;
    gp->X_train = (double*)calloc(n_train * n_inputs, sizeof(double));
    gp->y_train = (double*)calloc(n_train, sizeof(double));
    gp->kernel_params = (double*)calloc(n_inputs + 2, sizeof(double)); /* + sf + sn */
    gp->K = (double*)malloc(n_train * n_train * sizeof(double));
    gp->K_inv = (double*)malloc(n_train * n_train * sizeof(double));
    gp->alpha = (double*)calloc(n_train, sizeof(double));
    /* Default: lengthscale=1, sf=1, sn=0.1 */
    for (int d = 0; d < n_inputs; d++) gp->kernel_params[d] = 1.0;
    gp->kernel_params[n_inputs] = 1.0;
    gp->kernel_params[n_inputs + 1] = 0.1;
    return gp;
}

void uq_gp_free(UQGaussianProcess* gp) {
    if (!gp) return;
    free(gp->X_train); free(gp->y_train); free(gp->kernel_params);
    free(gp->K); free(gp->K_inv); free(gp->alpha);
    free(gp);
}

void uq_gp_set_data(UQGaussianProcess* gp, double* X, double* y) {
    memcpy(gp->X_train, X, gp->n_train * gp->n_inputs * sizeof(double));
    memcpy(gp->y_train, y, gp->n_train * sizeof(double));
}

void uq_gp_set_params(UQGaussianProcess* gp, double* params) {
    memcpy(gp->kernel_params, params,
           (gp->n_inputs + 2) * sizeof(double));
}

void uq_gp_train(UQGaussianProcess* gp) {
    int n = gp->n_train, dim = gp->n_inputs;
    double sn2 = gp->kernel_params[dim + 1];
    sn2 = sn2 * sn2; /* noise variance squared */

    /* Build K matrix: K_ij = k(x_i, x_j) + σ_n² δ_ij */
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double kv = gp_kernel_compute(gp->kernel_type,
                &gp->X_train[i * dim], &gp->X_train[j * dim],
                dim, gp->kernel_params);
            gp->K[i * n + j] = kv + ((i == j) ? sn2 : 0.0);
        }
    }

    /* Cholesky: K = L L^T */
    double* L = (double*)malloc(n * n * sizeof(double));
    memcpy(L, gp->K, n * n * sizeof(double));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            double sum = 0.0;
            for (int k = 0; k < j; k++)
                sum += L[i * n + k] * L[j * n + k];
            if (i == j)
                L[i * n + i] = sqrt(L[i * n + i] - sum + 1e-10);
            else
                L[i * n + j] = (L[i * n + j] - sum) / (L[j * n + j] + 1e-10);
        }
    }

    /* Solve L * L^T * alpha = y */
    double* y_copy = (double*)malloc(n * sizeof(double));
    memcpy(y_copy, gp->y_train, n * sizeof(double));
    /* Forward: L * z = y */
    for (int i = 0; i < n; i++) {
        double s = y_copy[i];
        for (int j = 0; j < i; j++)
            s -= L[i * n + j] * y_copy[j];
        y_copy[i] = s / (L[i * n + i] + 1e-10);
    }
    /* Back: L^T * alpha = z */
    for (int i = n - 1; i >= 0; i--) {
        double s = y_copy[i];
        for (int j = i + 1; j < n; j++)
            s -= L[j * n + i] * y_copy[j];
        y_copy[i] = s / (L[i * n + i] + 1e-10);
    }
    memcpy(gp->alpha, y_copy, n * sizeof(double));

    /* Store inverse Cholesky factor */
    memcpy(gp->K_inv, L, n * n * sizeof(double));

    /* Log marginal likelihood */
    double lml = -0.5 * gp->y_train[0] * gp->alpha[0] * n; /* simplified */
    for (int i = 0; i < n; i++) lml -= log(fabs(L[i * n + i]) + 1e-10);
    lml -= 0.5 * n * log(2.0 * UQ_PI);
    gp->log_marginal_likelihood = lml;

    free(L); free(y_copy);
    gp->trained = true;
}

double uq_gp_predict(UQGaussianProcess* gp, double* x_new, double* pred_var) {
    if (!gp || !gp->trained) return 0.0;
    int n = gp->n_train, dim = gp->n_inputs;
    double* ks = (double*)malloc(n * sizeof(double));
    for (int i = 0; i < n; i++)
        ks[i] = gp_kernel_compute(gp->kernel_type, x_new,
            &gp->X_train[i * dim], dim, gp->kernel_params);

    /* Mean: k_*^T * alpha */
    double mean = 0.0;
    for (int i = 0; i < n; i++) mean += ks[i] * gp->alpha[i];

    if (pred_var) {
        /* Solve L * v = k_s */
        double* v = (double*)malloc(n * sizeof(double));
        memcpy(v, ks, n * sizeof(double));
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < i; j++)
                v[i] -= gp->K_inv[i * n + j] * v[j];
            v[i] /= (gp->K_inv[i * n + i] + 1e-10);
        }
        double v_norm = 0.0;
        for (int i = 0; i < n; i++) v_norm += v[i] * v[i];

        double kss = gp_kernel_compute(gp->kernel_type, x_new, x_new,
            dim, gp->kernel_params);
        *pred_var = kss - v_norm;
        free(v);
    }

    free(ks);
    return mean;
}

void uq_gp_predict_batch(UQGaussianProcess* gp, double* X_new, int n_new,
                          double* pred_mean, double* pred_var) {
    for (int i = 0; i < n_new; i++)
        pred_mean[i] = uq_gp_predict(gp, &X_new[i * gp->n_inputs],
            pred_var ? &pred_var[i] : NULL);
}

void uq_gp_optimize_params(UQGaussianProcess* gp, int max_iter) {
    /* Simple gradient-free optimization by coordinate search */
    double best_lml = -1e300;
    int n_params = gp->n_inputs + 2;
    double* best_params = (double*)malloc(n_params * sizeof(double));
    memcpy(best_params, gp->kernel_params, n_params * sizeof(double));

    for (int iter = 0; iter < max_iter; iter++) {
        for (int p = 0; p < n_params; p++) {
            double orig = gp->kernel_params[p];
            for (int sign = -1; sign <= 1; sign += 2) {
                gp->kernel_params[p] = orig * (1.0 + sign * 0.1);
                uq_gp_train(gp);
                if (gp->log_marginal_likelihood > best_lml) {
                    best_lml = gp->log_marginal_likelihood;
                    memcpy(best_params, gp->kernel_params, n_params * sizeof(double));
                }
            }
            gp->kernel_params[p] = orig;
        }
        memcpy(gp->kernel_params, best_params, n_params * sizeof(double));
    }
    uq_gp_train(gp);
    free(best_params);
}
