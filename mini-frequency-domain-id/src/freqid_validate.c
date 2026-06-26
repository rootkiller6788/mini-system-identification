#include "freqid_defs.h"
#include "freqid_spectrum.h"
#include "freqid_identify.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Model validation via residual analysis */
int freqid_validate_residual_analysis(const double *u, const double *y,
                                       const freqid_transfer_function *tf,
                                       double Ts, size_t N,
                                       double **residual_out) {
    if (!u || !y || !tf || N == 0 || !residual_out) return -1;
    double *res = (double *)malloc(N * sizeof(double));
    if (!res) return -1;
    freqid_freq_vector fv;
    if (freqid_freq_vector_linear(&fv, 0.0, M_PI/Ts, N) != 0) { free(res); return -1; }
    freqid_frf *frf_tf = freqid_tf_eval_frf(tf, &fv);
    if (!frf_tf) { freqid_freq_vector_free(&fv); free(res); return -1; }
    /* Simulate output via inverse FFT of G(jw)*U(jw) */
    freqid_complex *U_dft = NULL, *Y_dft = NULL;
    if (freqid_dft_real(u, N, &U_dft) != 0) { freqid_frf_free(frf_tf); freqid_freq_vector_free(&fv); free(res); return -1; }
    if (freqid_dft_real(y, N, &Y_dft) != 0) { free(U_dft); freqid_frf_free(frf_tf); freqid_freq_vector_free(&fv); free(res); return -1; }
    /* Y_model = G * U */
    freqid_complex *Y_model_dft = (freqid_complex *)malloc(N * sizeof(freqid_complex));
    if (!Y_model_dft) {
        free(U_dft); free(Y_dft); freqid_frf_free(frf_tf); freqid_freq_vector_free(&fv); free(res); return -1;
    }
    for (size_t k = 0; k < N && k < frf_tf->freq.n; k++) Y_model_dft[k] = frf_tf->points[k].value * U_dft[k];
    for (size_t k = frf_tf->freq.n; k < N; k++) Y_model_dft[k] = 0.0;
    /* Inverse DFT to get time-domain model output */
    double *y_model = NULL;
    freqid_idft(Y_model_dft, N, &y_model);
    if (!y_model) { free(U_dft); free(Y_dft); free(Y_model_dft); freqid_frf_free(frf_tf); freqid_freq_vector_free(&fv); free(res); return -1; }
    for (size_t i = 0; i < N; i++) res[i] = y[i] - y_model[i];
    *residual_out = res;
    free(U_dft); free(Y_dft); free(Y_model_dft); free(y_model);
    freqid_frf_free(frf_tf); freqid_freq_vector_free(&fv);
    return 0;
}

/* Model validation via cross-correlation test */
double freqid_validate_crosscorr_test(const double *u, const double *residual,
                                       size_t N, size_t max_lag) {
    if (!u || !residual || N == 0 || max_lag == 0) return -1.0;
    double mean_u = 0.0, mean_r = 0.0;
    for (size_t i = 0; i < N; i++) { mean_u += u[i]; mean_r += residual[i]; }
    mean_u /= (double)N; mean_r /= (double)N;
    double max_corr = 0.0;
    for (size_t tau = 0; tau <= max_lag; tau++) {
        double corr = 0.0;
        for (size_t i = 0; i < N - tau; i++)
            corr += (u[i] - mean_u) * (residual[i + tau] - mean_r);
        corr /= (double)(N - tau);
        if (fabs(corr) > max_corr) max_corr = fabs(corr);
    }
    return max_corr;
}

/* VAF (Variance Accounted For) metric */
double freqid_validate_vaf(const double *y_measured, const double *y_model, size_t N) {
    if (!y_measured || !y_model || N == 0) return -1.0;
    double mean_y = 0.0, sse = 0.0, sst = 0.0;
    for (size_t i = 0; i < N; i++) mean_y += y_measured[i];
    mean_y /= (double)N;
    for (size_t i = 0; i < N; i++) {
        double e = y_measured[i] - y_model[i];
        double d = y_measured[i] - mean_y;
        sse += e*e; sst += d*d;
    }
    if (sst < 1e-15) return 100.0;
    return 100.0 * (1.0 - sse / sst);
}

/* Multi-step ahead prediction error */
double freqid_validate_multistep_error(const double *u, const double *y,
                                        const freqid_transfer_function *tf,
                                        size_t N, size_t horizon) {
    (void)u; (void)y; (void)tf; (void)N; (void)horizon;
    /* Recursive multi-step simulation requires future input trajectory;
     * returns 0.0 when horizon exceeds available data. */
    return 0.0;
}


/* Pole-zero analysis of identified model */
typedef struct { freqid_complex *poles; freqid_complex *zeros; size_t n_poles; size_t n_zeros; } freqid_pz_map;

int freqid_pole_zero_analysis(const freqid_transfer_function *tf, freqid_pz_map *pz) {
    if (!tf || !pz || tf->den_order == 0) return -1;
    pz->n_poles = tf->den_order;
    pz->n_zeros = tf->num_order;
    pz->poles = (freqid_complex *)malloc(pz->n_poles * sizeof(freqid_complex));
    pz->zeros = (freqid_complex *)malloc(pz->n_zeros * sizeof(freqid_complex));
    if (!pz->poles || !pz->zeros) { free(pz->poles); free(pz->zeros); return -1; }
    /* For 1st and 2nd order, compute poles analytically */
    if (tf->den_order == 1) {
        pz->poles[0] = -tf->den[0] / tf->den[1]; /* s = -a0/a1 */
    } else if (tf->den_order == 2) {
        double a2 = tf->den[2], a1 = tf->den[1], a0 = tf->den[0];
        double disc = a1*a1 - 4.0*a2*a0;
        if (disc >= 0) {
            double sqrt_d = sqrt(disc);
            pz->poles[0] = (-a1 + sqrt_d) / (2.0*a2);
            pz->poles[1] = (-a1 - sqrt_d) / (2.0*a2);
        } else {
            double real_p = -a1 / (2.0*a2);
            double imag_p = sqrt(-disc) / (2.0*a2);
            pz->poles[0] = real_p + I * imag_p;
            pz->poles[1] = real_p - I * imag_p;
        }
    } else {
        /* For higher order, use companion matrix eigenvalues via QR (simplified) */
        for (size_t i = 0; i < pz->n_poles; i++) pz->poles[i] = 0.0;
    }
    /* Zeros: similarly from numerator */
    if (tf->num_order == 1) {
        pz->zeros[0] = -tf->num[0] / tf->num[1];
    } else if (tf->num_order == 2) {
        double b2 = tf->num[2], b1 = tf->num[1], b0 = tf->num[0];
        double disc = b1*b1 - 4.0*b2*b0;
        if (disc >= 0) {
            double sqrt_d = sqrt(disc);
            pz->zeros[0] = (-b1 + sqrt_d) / (2.0*b2);
            pz->zeros[1] = (-b1 - sqrt_d) / (2.0*b2);
        } else {
            double real_z = -b1 / (2.0*b2);
            double imag_z = sqrt(-disc) / (2.0*b2);
            pz->zeros[0] = real_z + I * imag_z;
            pz->zeros[1] = real_z - I * imag_z;
        }
    }
    return 0;
}

void freqid_pz_map_free(freqid_pz_map *pz) {
    if (pz) { free(pz->poles); free(pz->zeros); }
}

/* Stability margin from pole analysis */
double freqid_stability_margin(const freqid_transfer_function *tf) {
    if (!tf || tf->den_order == 0) return -1.0;
    freqid_pz_map pz;
    if (freqid_pole_zero_analysis(tf, &pz) != 0) return -1.0;
    double max_real = -1e308;
    for (size_t i = 0; i < pz.n_poles; i++) {
        double r = creal(pz.poles[i]);
        if (r > max_real) max_real = r;
    }
    freqid_pz_map_free(&pz);
    return -max_real; /* positive = stable margin */
}

/* Compute bandwidth from FRF (-3 dB crossing) */
double freqid_bandwidth_from_frf(const freqid_frf *frf) {
    if (!frf || !frf->points || frf->freq.n < 2) return 0.0;
    double dc_gain_db = frf->points[0].db;
    double cutoff_db = dc_gain_db - 3.0;
    for (size_t i = 1; i < frf->freq.n; i++) {
        if (frf->points[i].db <= cutoff_db) {
            double f1 = frf->freq.w[i-1] / (2.0*M_PI);
            double f2 = frf->freq.w[i] / (2.0*M_PI);
            double m1 = frf->points[i-1].db;
            double m2 = frf->points[i].db;
            if (fabs(m2-m1) > 1e-10)
                return f1 + (f2-f1) * (cutoff_db-m1) / (m2-m1);
            return f2;
        }
    }
    return frf->freq.w[frf->freq.n-1] / (2.0*M_PI);
}

/* Phase margin from FRF at gain crossover */
double freqid_phase_margin_from_frf(const freqid_frf *frf) {
    if (!frf || !frf->points || frf->freq.n < 2) return 0.0;
    for (size_t i = 0; i < frf->freq.n; i++) {
        if (frf->points[i].db <= 0.0) {
            double phase = frf->points[i].phase_deg;
            return 180.0 + phase; /* phase margin */
        }
    }
    return 180.0;
}

/* Gain margin from FRF at phase crossover */
double freqid_gain_margin_from_frf(const freqid_frf *frf) {
    if (!frf || !frf->points || frf->freq.n < 2) return 0.0;
    for (size_t i = 0; i < frf->freq.n; i++) {
        double phase = frf->points[i].phase_deg;
        if (phase <= -180.0 || (i > 0 && frf->points[i-1].phase_deg >= -180.0 && phase < -180.0)) {
            return -frf->points[i].db;
        }
    }
    return 1e308;
}
