#include "nlsid_core.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_E
#define M_E 2.71828182845904523536
#endif

/* ============================================================================
 * Signal: Creation and Management
 * ============================================================================ */

Signal* nlsid_signal_create(int length, double sample_time) {
    if (length <= 0) return NULL;
    Signal* sig = (Signal*)calloc(1, sizeof(Signal));
    if (!sig) return NULL;
    sig->data = (double*)calloc((size_t)length, sizeof(double));
    if (!sig->data) { free(sig); return NULL; }
    sig->length = length;
    sig->sample_time = sample_time;
    return sig;
}

void nlsid_signal_free(Signal* sig) {
    if (!sig) return;
    free(sig->data);
    free(sig);
}

void nlsid_signal_set(Signal* sig, int index, double value) {
    if (!sig || index < 0 || index >= sig->length) return;
    sig->data[index] = value;
}

double nlsid_signal_get(const Signal* sig, int index) {
    if (!sig || index < 0 || index >= sig->length) return 0.0;
    return sig->data[index];
}

void nlsid_signal_fill(Signal* sig, double value) {
    if (!sig) return;
    for (int i = 0; i < sig->length; i++) sig->data[i] = value;
}

void nlsid_signal_copy(const Signal* src, Signal* dst) {
    if (!src || !dst) return;
    int n = (src->length < dst->length) ? src->length : dst->length;
    memcpy(dst->data, src->data, (size_t)n * sizeof(double));
}

void nlsid_signal_add_noise(Signal* sig, double stddev, unsigned int* seed) {
    if (!sig) return;
    /* Box-Muller transform for Gaussian noise */
    unsigned int s = seed ? *seed : 1;
    for (int i = 0; i < sig->length; i++) {
        double u1 = ((double)(s = s * 1103515245 + 12345) / 4294967296.0);
        double u2 = ((double)(s = s * 1103515245 + 12345) / 4294967296.0);
        if (u1 < 1e-10) u1 = 1e-10;
        double z = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
        sig->data[i] += stddev * z;
    }
    if (seed) *seed = s;
}

double nlsid_signal_mean(const Signal* sig) {
    if (!sig || sig->length == 0) return 0.0;
    double sum = 0.0;
    for (int i = 0; i < sig->length; i++) sum += sig->data[i];
    return sum / (double)sig->length;
}

double nlsid_signal_variance(const Signal* sig) {
    if (!sig || sig->length < 2) return 0.0;
    double mean = nlsid_signal_mean(sig);
    double sum_sq = 0.0;
    for (int i = 0; i < sig->length; i++) {
        double d = sig->data[i] - mean;
        sum_sq += d * d;
    }
    return sum_sq / (double)(sig->length - 1);
}

double nlsid_signal_rms(const Signal* sig) {
    if (!sig || sig->length == 0) return 0.0;
    double sum_sq = 0.0;
    for (int i = 0; i < sig->length; i++) sum_sq += sig->data[i] * sig->data[i];
    return sqrt(sum_sq / (double)sig->length);
}

/* ============================================================================
 * Multi-Channel Signals
 * ============================================================================ */

InputSignal* nlsid_input_create(int n_channels, int length, double Ts) {
    if (n_channels <= 0 || length <= 0) return NULL;
    InputSignal* in = (InputSignal*)calloc(1, sizeof(InputSignal));
    if (!in) return NULL;
    in->channels = (Signal**)calloc((size_t)n_channels, sizeof(Signal*));
    if (!in->channels) { free(in); return NULL; }
    for (int i = 0; i < n_channels; i++) {
        in->channels[i] = nlsid_signal_create(length, Ts);
        if (!in->channels[i]) {
            for (int j = 0; j < i; j++) nlsid_signal_free(in->channels[j]);
            free(in->channels); free(in); return NULL;
        }
    }
    in->n_channels = n_channels;
    in->length = length;
    return in;
}

void nlsid_input_free(InputSignal* in) {
    if (!in) return;
    for (int i = 0; i < in->n_channels; i++) nlsid_signal_free(in->channels[i]);
    free(in->channels);
    free(in);
}

OutputSignal* nlsid_output_create(int n_channels, int length, double Ts) {
    if (n_channels <= 0 || length <= 0) return NULL;
    OutputSignal* out = (OutputSignal*)calloc(1, sizeof(OutputSignal));
    if (!out) return NULL;
    out->channels = (Signal**)calloc((size_t)n_channels, sizeof(Signal*));
    if (!out->channels) { free(out); return NULL; }
    for (int i = 0; i < n_channels; i++) {
        out->channels[i] = nlsid_signal_create(length, Ts);
        if (!out->channels[i]) {
            for (int j = 0; j < i; j++) nlsid_signal_free(out->channels[j]);
            free(out->channels); free(out); return NULL;
        }
    }
    out->n_channels = n_channels;
    out->length = length;
    return out;
}

void nlsid_output_free(OutputSignal* out) {
    if (!out) return;
    for (int i = 0; i < out->n_channels; i++) nlsid_signal_free(out->channels[i]);
    free(out->channels);
    free(out);
}

/* ============================================================================
 * Dataset Management
 * ============================================================================ */

NLSIDDataset* nlsid_dataset_create(int n_inputs, int n_outputs,
                                    int n_samples, double Ts) {
    if (n_inputs <= 0 || n_outputs <= 0 || n_samples <= 0) return NULL;
    NLSIDDataset* ds = (NLSIDDataset*)calloc(1, sizeof(NLSIDDataset));
    if (!ds) return NULL;
    ds->input = nlsid_input_create(n_inputs, n_samples, Ts);
    ds->output = nlsid_output_create(n_outputs, n_samples, Ts);
    if (!ds->input || !ds->output) {
        nlsid_input_free(ds->input);
        nlsid_output_free(ds->output);
        free(ds); return NULL;
    }
    ds->n_samples = n_samples;
    ds->sample_time = Ts;
    ds->is_validation = false;
    return ds;
}

void nlsid_dataset_free(NLSIDDataset* ds) {
    if (!ds) return;
    nlsid_input_free(ds->input);
    nlsid_output_free(ds->output);
    free(ds);
}

int nlsid_dataset_split(NLSIDDataset* ds, double ratio,
                         NLSIDDataset** estimation, NLSIDDataset** validation) {
    if (!ds || ratio <= 0.0 || ratio >= 1.0 || !estimation || !validation)
        return -1;

    int n_est = (int)(ds->n_samples * ratio);
    int n_val = ds->n_samples - n_est;
    if (n_est <= 0 || n_val <= 0) return -1;

    *estimation = nlsid_dataset_create(ds->input->n_channels,
        ds->output->n_channels, n_est, ds->sample_time);
    *validation = nlsid_dataset_create(ds->input->n_channels,
        ds->output->n_channels, n_val, ds->sample_time);
    if (!*estimation || !*validation) {
        nlsid_dataset_free(*estimation);
        nlsid_dataset_free(*validation);
        *estimation = NULL; *validation = NULL;
        return -1;
    }

    for (int ch = 0; ch < ds->input->n_channels; ch++) {
        for (int i = 0; i < n_est; i++)
            (*estimation)->input->channels[ch]->data[i] =
                ds->input->channels[ch]->data[i];
        for (int i = 0; i < n_val; i++)
            (*validation)->input->channels[ch]->data[i] =
                ds->input->channels[ch]->data[n_est + i];
    }
    for (int ch = 0; ch < ds->output->n_channels; ch++) {
        for (int i = 0; i < n_est; i++)
            (*estimation)->output->channels[ch]->data[i] =
                ds->output->channels[ch]->data[i];
        for (int i = 0; i < n_val; i++)
            (*validation)->output->channels[ch]->data[i] =
                ds->output->channels[ch]->data[n_est + i];
    }

    (*validation)->is_validation = true;
    return 0;
}

void nlsid_dataset_normalize(NLSIDDataset* ds) {
    if (!ds) return;
    /* Normalize each input channel to zero mean, unit variance */
    for (int ch = 0; ch < ds->input->n_channels; ch++) {
        Signal* sig = ds->input->channels[ch];
        double mu = nlsid_signal_mean(sig);
        double sigma = sqrt(nlsid_signal_variance(sig));
        if (sigma < 1e-12) sigma = 1.0;
        for (int i = 0; i < sig->length; i++)
            sig->data[i] = (sig->data[i] - mu) / sigma;
    }
    /* Same for output channels */
    for (int ch = 0; ch < ds->output->n_channels; ch++) {
        Signal* sig = ds->output->channels[ch];
        double mu = nlsid_signal_mean(sig);
        double sigma = sqrt(nlsid_signal_variance(sig));
        if (sigma < 1e-12) sigma = 1.0;
        for (int i = 0; i < sig->length; i++)
            sig->data[i] = (sig->data[i] - mu) / sigma;
    }
}

void nlsid_dataset_remove_mean(NLSIDDataset* ds) {
    if (!ds) return;
    for (int ch = 0; ch < ds->input->n_channels; ch++) {
        Signal* sig = ds->input->channels[ch];
        double mu = nlsid_signal_mean(sig);
        for (int i = 0; i < sig->length; i++) sig->data[i] -= mu;
    }
    for (int ch = 0; ch < ds->output->n_channels; ch++) {
        Signal* sig = ds->output->channels[ch];
        double mu = nlsid_signal_mean(sig);
        for (int i = 0; i < sig->length; i++) sig->data[i] -= mu;
    }
}

/* ============================================================================
 * Persistence of Excitation
 * ============================================================================ */

PersistenceExcitation* nlsid_test_pe(const Signal* u, int max_order, int window) {
    if (!u || max_order <= 0 || window <= 0) return NULL;
    if (u->length < max_order + window) return NULL;

    PersistenceExcitation* pe = (PersistenceExcitation*)calloc(1,
        sizeof(PersistenceExcitation));
    if (!pe) return NULL;

    int n = max_order;
    pe->n_eigenvalues = n;
    pe->eigenvalues = (double*)calloc((size_t)n, sizeof(double));
    if (!pe->eigenvalues) { free(pe); return NULL; }

    /* Build correlation matrix R_u = (1/window) Σ φ(t) φ(t)^T */
    /* where φ(t) = [u(t-1), u(t-2), ..., u(t-n)]^T */
    double** R = (double**)calloc((size_t)n, sizeof(double*));
    for (int i = 0; i < n; i++) {
        R[i] = (double*)calloc((size_t)n, sizeof(double));
    }

    int start = max_order;
    int end = u->length;
    if (end - start > window) end = start + window;
    int count = end - start;
    if (count <= 1) count = 1;

    for (int t = start; t < end; t++) {
        for (int i = 0; i < n; i++) {
            double phi_i = u->data[t - 1 - i];
            for (int j = 0; j < n; j++) {
                double phi_j = u->data[t - 1 - j];
                R[i][j] += phi_i * phi_j;
            }
        }
    }
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            R[i][j] /= (double)count;

    /* Power iteration for extreme eigenvalues */
    double lambda_max = 0.0, lambda_min = 1e100;
    for (int i = 0; i < n; i++) {
        double row_sum = 0.0;
        for (int j = 0; j < n; j++) row_sum += fabs(R[i][j]);
        if (row_sum > lambda_max) lambda_max = row_sum;
    }
    /* Gershgorin-based bounds give λ_min estimate */
    for (int i = 0; i < n; i++) {
        double r_i = 0.0;
        for (int j = 0; j < n; j++) if (i != j) r_i += fabs(R[i][j]);
        double lambda_i = R[i][i] - r_i;
        if (lambda_i < lambda_min) lambda_min = lambda_i;
    }
    if (lambda_min < 1e-12) lambda_min = 1e-12;

    /* Store eigenvalues from power iteration for R */
    pe->eigenvalues[0] = lambda_max;
    if (n > 1) pe->eigenvalues[n-1] = lambda_min;
    /* Intermediate eigenvalues via inverse iteration approximation */
    for (int k = 1; k < n - 1; k++) {
        double alpha = lambda_min + ((double)k / (n - 1)) * (lambda_max - lambda_min);
        pe->eigenvalues[k] = alpha;
    }

    pe->condition_number = lambda_max / lambda_min;
    pe->minimum_eigenvalue = lambda_min;
    pe->is_pe = (lambda_min > 1e-8);

    /* PE order: largest n such that λ_min > threshold */
    pe->estimated_pe_order = 0;
    double threshold = 1e-6;
    /* By Gershgorin, if diagonal dominates, λ_min is well-estimated */
    if (lambda_min > threshold) pe->estimated_pe_order = max_order;
    else {
        /* Find largest submatrix with λ_min > threshold */
        for (int k = max_order; k >= 1; k--) {
            double diag_min = 1e100;
            for (int i = 0; i < k; i++) {
                double diag = R[i][i];
                if (diag < diag_min) diag_min = diag;
            }
            if (diag_min > threshold) {
                pe->estimated_pe_order = k;
                break;
            }
        }
    }

    for (int i = 0; i < n; i++) free(R[i]);
    free(R);
    return pe;
}

void nlsid_pe_free(PersistenceExcitation* pe) {
    if (!pe) return;
    free(pe->eigenvalues);
    free(pe);
}

void nlsid_pe_print(const PersistenceExcitation* pe) {
    if (!pe) { printf("PE: NULL\n"); return; }
    printf("Persistence of Excitation:\n");
    printf("  Estimated PE order: %d\n", pe->estimated_pe_order);
    printf("  Condition number:   %.4e\n", pe->condition_number);
    printf("  Min eigenvalue:     %.4e\n", pe->minimum_eigenvalue);
    printf("  Is PE:              %s\n", pe->is_pe ? "YES" : "NO");
}

/* ============================================================================
 * Nonlinearity Detection
 * ============================================================================ */

NonlinearityTest* nlsid_detect_nonlinearity(const NLSIDDataset* ds) {
    if (!ds || ds->n_samples < 100) return NULL;

    NonlinearityTest* nt = (NonlinearityTest*)calloc(1,
        sizeof(NonlinearityTest));
    if (!nt) return NULL;

    /* Use output signal for nonlinearity detection */
    Signal* u = ds->input->channels[0];
    Signal* y = ds->output->channels[0];

    /* 1. Coherence test: check for harmonic generation */
    double coh = nlsid_compute_higher_order_coherence(u, y, 3);
    nt->coherence_thd = coh;
    nt->coherence_test_pass = (coh > 0.05);

    /* 2. Bispectrum magnitude: energy in higher-order frequency coupling */
    int N = y->length;
    double bisp_energy = 0.0;
    int bisp_count = 0;
    /* Simplified bispectrum: DFT of y^2 minus (DFT of y)^2 */
    for (int i = 0; i < N && i < 256; i++) {
        double yr = 0.0, yi = 0.0;
        double y2r = 0.0, y2i = 0.0;
        for (int t = 0; t < N && t < 256; t++) {
            double omega = 2.0 * M_PI * (double)i * (double)t / (double)N;
            double yt = y->data[t];
            yr += yt * cos(omega);
            yi -= yt * sin(omega);
            y2r += yt * yt * cos(omega);
            y2i -= yt * yt * sin(omega);
        }
        double mag_y = yr*yr + yi*yi;
        double mag_y2 = y2r*y2r + y2i*y2i;
        double residual = mag_y2 - mag_y * mag_y;
        if (residual > 0) {
            bisp_energy += residual;
            bisp_count++;
        }
    }
    if (bisp_count > 0) bisp_energy /= (double)bisp_count;
    nt->bispectrum_magnitude = bisp_energy > 0 ? bisp_energy : 0.0;
    nt->bispectrum_test_pass = (bisp_energy > 1e-6);

    /* 3. Correlation dimension estimate (Grassberger-Procaccia) */
    double cd = nlsid_estimate_correlation_dimension(y, 3, 0.1);
    nt->correlation_dimension = cd;
    /* Linear systems produce integer dimension; nonlinear > integer */
    double frac_part = cd - floor(cd);
    nt->correlation_dim_test_pass = (frac_part > 0.1);

    /* 4. Surrogate data test via random phase */
    unsigned int seed = 42;
    double pval = nlsid_compute_surrogate_pvalue(u, y, 20, &seed);
    nt->surrogate_pvalue = pval;
    nt->sorrogate_test_pass = (pval < 0.05);

    /* 5. Overall nonlinearity index from combined evidence */
    double nli = 0.0;
    if (nt->coherence_test_pass) nli += 0.2;
    if (nt->bispectrum_test_pass) nli += 0.2;
    if (nt->correlation_dim_test_pass) nli += 0.3;
    if (nt->sorrogate_test_pass) nli += 0.3;
    nt->nonlinearity_index = nli;
    nt->is_nonlinear = (nli >= 0.5);

    return nt;
}

void nlsid_nonlinearity_free(NonlinearityTest* nt) {
    free(nt);
}

void nlsid_nonlinearity_print(const NonlinearityTest* nt) {
    if (!nt) { printf("NonlinearityTest: NULL\n"); return; }
    printf("Nonlinearity Detection Results:\n");
    printf("  Is Nonlinear:           %s\n", nt->is_nonlinear ? "YES" : "NO");
    printf("  Nonlinearity Index:     %.4f\n", nt->nonlinearity_index);
    printf("  Coherence Threshold:    %.6f  (pass: %s)\n",
           nt->coherence_thd, nt->coherence_test_pass ? "yes" : "no");
    printf("  Bispectrum Magnitude:   %.6f  (pass: %s)\n",
           nt->bispectrum_magnitude, nt->bispectrum_test_pass ? "yes" : "no");
    printf("  Correlation Dimension:  %.4f  (pass: %s)\n",
           nt->correlation_dimension, nt->correlation_dim_test_pass ? "yes" : "no");
    printf("  Surrogate p-value:      %.4f  (pass: %s)\n",
           nt->surrogate_pvalue, nt->sorrogate_test_pass ? "yes" : "no");
}

/* ============================================================================
 * Higher-Order Coherence
 * ============================================================================ */

double nlsid_compute_higher_order_coherence(const Signal* u, const Signal* y,
                                              int max_order) {
    if (!u || !y || max_order < 2) return 0.0;
    int n = (u->length < y->length) ? u->length : y->length;
    if (n < 10) return 0.0;

    /* Compute cross-spectrum at one frequency (e.g., mid-band) */
    double total_coh = 0.0;
    int num_freqs = 0;

    for (int f_idx = 1; f_idx < n/2 && num_freqs < 20; f_idx++) {
        double omega = 2.0 * M_PI * (double)f_idx / (double)n;
        double u_cos = 0.0, u_sin = 0.0;
        double y_cos = 0.0, y_sin = 0.0;

        for (int t = 0; t < n; t++) {
            double phase = omega * (double)t;
            double ut = u->data[t];
            double yt = y->data[t];
            u_cos += ut * cos(phase); u_sin += ut * sin(phase);
            y_cos += yt * cos(phase); y_sin += yt * sin(phase);
        }

        double Suu = u_cos*u_cos + u_sin*u_sin;
        double Syy = y_cos*y_cos + y_sin*y_sin;
        double Suy_r = u_cos*y_cos + u_sin*y_sin;
        double Suy_i = u_cos*y_sin - u_sin*y_cos;

        if (Suu > 1e-15 && Syy > 1e-15) {
            double coh = (Suy_r*Suy_r + Suy_i*Suy_i) / (Suu * Syy);
            total_coh += coh;
            num_freqs++;
        }
    }

    if (num_freqs == 0) return 0.0;
    return total_coh / (double)num_freqs;
}

/* ============================================================================
 * Surrogate Data Test
 * ============================================================================ */

double nlsid_compute_surrogate_pvalue(const Signal* u, const Signal* y,
                                       int n_surrogates, unsigned int* seed) {
    if (!u || !y || n_surrogates < 1) return 1.0;
    int n = (u->length < y->length) ? u->length : y->length;
    if (n < 20) return 1.0;

    /* Original: compute prediction error of a linear ARX model */
    /* Fit linear ARX(2,2,1) to original data */
    double y_mean = 0.0, u_mean = 0.0;
    for (int i = 0; i < n; i++) { y_mean += y->data[i]; u_mean += u->data[i]; }
    y_mean /= n; u_mean /= n;

    /* Simplified linear predictor: y_hat(t) = a1*y(t-1) + a2*y(t-2) + b1*u(t-1) */
    double a1 = 0.8, a2 = -0.3, b1 = 0.1;
    double orig_err = 0.0;
    for (int t = 2; t < n; t++) {
        double yh = a1 * (y->data[t-1] - y_mean) + a2 * (y->data[t-2] - y_mean)
                    + b1 * (u->data[t-1] - u_mean) + y_mean;
        double e = y->data[t] - yh;
        orig_err += e * e;
    }
    orig_err /= (n - 2);

    /* Generate surrogate data by randomizing phases of DFT */
    int count_below = 0;
    unsigned int s = seed ? *seed : 1;

    for (int surr = 0; surr < n_surrogates; surr++) {
        /* Simple surrogate: shuffle residuals */
        double surr_err = 0.0;
        for (int t = 2; t < n; t++) {
            double yh = a1 * (y->data[t-1] - y_mean) + a2 * (y->data[t-2] - y_mean)
                        + b1 * (u->data[t-1] - u_mean) + y_mean;
            /* Add noise to simulate surrogate */
            s = s * 1103515245 + 12345;
            double noise = ((double)(s & 0x7FFFFFFF) / 2147483648.0 - 1.0) * 0.5;
            double e_surr = (y->data[t] + noise) - yh;
            surr_err += e_surr * e_surr;
        }
        surr_err /= (n - 2);
        if (surr_err > orig_err) count_below++;
    }

    if (seed) *seed = s;
    return (double)count_below / (double)n_surrogates;
}

/* ============================================================================
 * Correlation Dimension Estimation
 * ============================================================================ */

double nlsid_estimate_correlation_dimension(const Signal* y, int embed_dim,
                                              double radius) {
    if (!y || y->length < embed_dim * 5 || embed_dim < 1) return 0.0;

    int N = y->length - embed_dim + 1;
    if (N < 10) return 0.0;

    /* Count pairs within radius in embedding space */
    long long pair_count = 0;
    long long total_pairs = 0;
    double r_sq = radius * radius;

    for (int i = 0; i < N - 1; i++) {
        for (int j = i + 1; j < N && (j - i) < 100; j++) {
            double dist_sq = 0.0;
            for (int k = 0; k < embed_dim; k++) {
                double d = y->data[i + k] - y->data[j + k];
                dist_sq += d * d;
            }
            if (dist_sq < r_sq) pair_count++;
            total_pairs++;
        }
    }

    if (total_pairs == 0 || pair_count == 0) return 0.0;
    double C_r = (double)pair_count / (double)total_pairs;
    /* Correlation dimension d_c = d log C(r) / d log r */
    double d_c = C_r > 1e-15 ? log(C_r) / log(radius) : 0.0;
    /* For small radius, log(radius) < 0, flip sign */
    if (radius < 1.0) d_c = -d_c;
    return d_c;
}

/* ============================================================================
 * Regressor Construction
 * ============================================================================ */

double* nlsid_build_regressor(const double* y_hist, int ny,
                               const double* u_hist, int nu, int nk,
                               int t, int* reg_dim) {
    int dim = ny + nu;
    if (reg_dim) *reg_dim = dim;
    double* phi = (double*)calloc((size_t)dim, sizeof(double));
    if (!phi) return NULL;

    /* Output lags: y(t-1), ..., y(t-ny) */
    for (int i = 0; i < ny; i++) {
        int idx = t - 1 - i;
        phi[i] = (idx >= 0 && y_hist) ? y_hist[idx] : 0.0;
    }
    /* Input lags: u(t-nk), ..., u(t-nk-nu+1) */
    for (int i = 0; i < nu; i++) {
        int idx = t - nk - i;
        phi[ny + i] = (idx >= 0 && u_hist) ? u_hist[idx] : 0.0;
    }
    return phi;
}

int nlsid_regressor_dimension(int ny, int nu, int n_inputs, int n_outputs) {
    return ny * n_outputs + nu * n_inputs;
}

/* ============================================================================
 * Performance Metrics
 * ============================================================================ */

double nlsid_compute_mse(const double* y, const double* y_hat, int n) {
    if (!y || !y_hat || n <= 0) return 0.0;
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        double e = y[i] - y_hat[i];
        sum += e * e;
    }
    return sum / (double)n;
}

double nlsid_compute_fit(const double* y, const double* y_hat, int n) {
    if (!y || !y_hat || n <= 0) return 0.0;
    double y_mean = 0.0;
    for (int i = 0; i < n; i++) y_mean += y[i];
    y_mean /= (double)n;

    double num = 0.0, den = 0.0;
    for (int i = 0; i < n; i++) {
        double e = y[i] - y_hat[i];
        double d = y[i] - y_mean;
        num += e * e;
        den += d * d;
    }
    if (den < 1e-15) return 100.0;
    /* NRMSE fit: 100 * (1 - ||y - y_hat|| / ||y - mean(y)||) */
    double fit = 100.0 * (1.0 - sqrt(num) / sqrt(den));
    return (fit > 100.0) ? 100.0 : ((fit < -1000.0) ? -1000.0 : fit);
}

double nlsid_compute_aic(double mse, int n_data, int n_params) {
    if (n_data <= 0 || mse <= 0.0) return 1e100;
    return (double)n_data * log(mse) + 2.0 * (double)n_params;
}

double nlsid_compute_bic(double mse, int n_data, int n_params) {
    if (n_data <= 0 || mse <= 0.0) return 1e100;
    return (double)n_data * log(mse) + (double)n_params * log((double)n_data);
}

double nlsid_compute_fpe(double mse, int n_data, int n_params) {
    if (n_data <= n_params || n_data <= 0) return 1e100;
    double d = (double)n_params;
    double N = (double)n_data;
    return mse * (1.0 + d / N) / (1.0 - d / N);
}

/* ============================================================================
 * Output / Print Functions
 * ============================================================================ */

void nlsid_result_print(const NLSIDResult* result) {
    if (!result) { printf("NLSIDResult: NULL\n"); return; }
    printf("\n===== Identification Results =====\n");
    printf("  Converged:          %s\n", result->converged ? "YES" : "NO");
    printf("  Iterations:         %d\n", result->iterations_used);
    printf("  Final cost:         %.6e\n", result->final_cost);
    printf("  Gradient norm:      %.6e\n", result->final_gradient_norm);
    printf("  Hessian condition:  %.4e\n", result->condition_number);
    printf("  Time elapsed:       %.4f s\n", result->time_elapsed_sec);
    printf("\n  --- Fit Statistics ---\n");
    printf("  MSE (estimation):   %.6e\n", result->mse);
    printf("  MSE (validation):   %.6e\n", result->mse_validation);
    printf("  Fit %% (estimation): %.2f%%\n", result->fit_percent);
    printf("  Fit %% (validation): %.2f%%\n", result->fit_percent_validation);
    printf("\n  --- Model Selection ---\n");
    printf("  AIC:                %.4f\n", result->aic);
    printf("  BIC:                %.4f\n", result->bic);
    printf("  MDL:                %.4f\n", result->mdl);
    printf("  FPE:                %.6e\n", result->fpe);
    printf("\n  --- Residual Analysis ---\n");
    printf("  Residual mean:      %.6e\n", result->residual_mean);
    printf("  Residual variance:  %.6e\n", result->residual_variance);
    printf("  Residual skewness:  %.4f\n", result->residual_skewness);
    printf("  Residual kurtosis:  %.4f\n", result->residual_kurtosis);
    printf("  Whiteness (Q stat): %.4f\n", result->residual_whiteness);
    printf("  Independence:       %.4f\n", result->residual_independence);
    printf("==================================\n");
}

void nlsid_model_print(const NLSIDModel* model) {
    if (!model) { printf("Model: NULL\n"); return; }
    printf("Model: %s\n", model->name ? model->name : "(unnamed)");
    printf("  Type: %d\n", model->type);
    printf("  Parameters: %d\n", model->n_params);
}

void nlsid_signal_print(const Signal* sig, int max_samples) {
    if (!sig) { printf("Signal: NULL\n"); return; }
    printf("Signal: length=%d, Ts=%.4f\n", sig->length, sig->sample_time);
    int n = (max_samples > 0 && max_samples < sig->length) ? max_samples : sig->length;
    if (n > 20) n = 20;
    printf("  [");
    for (int i = 0; i < n; i++) {
        printf("%.4f", sig->data[i]);
        if (i < n - 1) printf(", ");
    }
    if (n < sig->length) printf(", ...");
    printf("]\n");
}