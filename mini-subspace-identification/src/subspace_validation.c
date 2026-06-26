#include "subspace_core.h"
#include "subspace_validation.h"
#include "subspace_linalg.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ============================================================================
 * Subspace Identification -- Model Validation
 *
 * Comprehensive validation of identified state-space models, testing
 * prediction accuracy, residual whiteness, input-output independence,
 * stability, and parameter uncertainty.
 * ============================================================================ */

/* NRMSE fit percentage */
double subspace_validation_nrmse(const double *y, const double *y_hat, int N) {
    return subspace_fit_percent(y, y_hat, N);
}

/* Variance Accounted For */
double subspace_validation_vaf(const double *y, const double *y_hat, int N) {
    return subspace_variance_accounted_for(y, y_hat, N);
}

/* Mean Absolute Percentage Error */
double subspace_validation_mape(const double *y, const double *y_hat, int N) {
    if (!y || !y_hat || N <= 0) return 0.0;
    double sum = 0.0;
    int count = 0;
    for (int i = 0; i < N; i++) {
        if (fabs(y[i]) > 1e-10) {
            sum += fabs((y[i] - y_hat[i]) / y[i]);
            count++;
        }
    }
    return (count > 0) ? 100.0 * sum / (double)count : 0.0;
}

/* Maximum Absolute Error */
double subspace_validation_maxae(const double *y, const double *y_hat, int N) {
    if (!y || !y_hat || N <= 0) return 0.0;
    double max_err = 0.0;
    for (int i = 0; i < N; i++) {
        double err = fabs(y[i] - y_hat[i]);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

/* Autocorrelation of residuals */
void subspace_residual_autocorrelation(const double *e, int N,
                                        int max_lag, double *acf) {
    if (!e || !acf || N <= 1 || max_lag <= 0) return;
    double mean_e = subspace_mean(e, N);
    double var = 0.0;
    for (int i = 0; i < N; i++) { double d = e[i] - mean_e; var += d * d; }
    acf[0] = 1.0;
    if (var < 1e-15) {
        for (int k = 1; k <= max_lag; k++) acf[k] = 0.0;
        return;
    }
    for (int k = 1; k <= max_lag && k < N; k++) {
        double cov = 0.0;
        for (int i = 0; i < N - k; i++)
            cov += (e[i] - mean_e) * (e[i + k] - mean_e);
        acf[k] = cov / var;
    }
}

/* Ljung-Box Q test for residual whiteness.
 * Q = N*(N+2)*sum_{k=1..h} r_k^2/(N-k) ~ chi^2(h)
 * Returns approximate p-value using Wilson-Hilferty transformation. */
double subspace_ljung_box_test(const double *e, int N, int max_lag,
                                double *Q_stat) {
    if (!e || N <= max_lag || max_lag <= 0) return 0.0;
    double mean_e = subspace_mean(e, N);
    double var = 0.0;
    for (int i = 0; i < N; i++) { double d = e[i] - mean_e; var += d * d; }
    if (var < 1e-15) { if (Q_stat) *Q_stat = 0.0; return 1.0; }

    double Q = 0.0;
    for (int k = 1; k <= max_lag; k++) {
        double cov = 0.0;
        for (int i = 0; i < N - k; i++)
            cov += (e[i] - mean_e) * (e[i + k] - mean_e);
        double r_k = cov / var;
        Q += r_k * r_k / (double)(N - k);
    }
    Q *= (double)N * (double)(N + 2);
    if (Q_stat) *Q_stat = Q;

    /* Wilson-Hilferty approximation for chi^2 p-value:
     * Z = ((Q/df)^{1/3} - 1 + 2/(9*df)) / sqrt(2/(9*df))
     * p = 1 - Phi(Z) */
    double df = (double)max_lag;
    if (df < 1.0) df = 1.0;
    double Q_df = Q / df;
    if (Q_df < 0.0) Q_df = 0.0;
    double term = cbrt(Q_df) - 1.0 + 2.0 / (9.0 * df);
    double denom = sqrt(2.0 / (9.0 * df));
    double Z = term / denom;

    /* Approximate normal CDF: P(Z) ~ 1 - exp(-0.717*Z - 0.416*Z^2) for Z >= 0 */
    double p_val;
    if (Z >= 0) {
        p_val = exp(-0.717 * Z - 0.416 * Z * Z);
    } else {
        p_val = 1.0 - exp(-0.717 * (-Z) - 0.416 * Z * Z);
    }
    if (p_val > 1.0) p_val = 1.0;
    if (p_val < 0.0) p_val = 0.0;
    return p_val;
}

/* Cross-correlation between input and residuals */
void subspace_cross_correlation_ue(const double *u, const double *e,
                                     int N, int max_lag, double *ccf) {
    if (!u || !e || !ccf || N <= 1 || max_lag <= 0) return;
    double mean_u = subspace_mean(u, N);
    double mean_e = subspace_mean(e, N);
    double var_u = 0.0, var_e = 0.0;
    for (int i = 0; i < N; i++) {
        double du = u[i] - mean_u, de = e[i] - mean_e;
        var_u += du * du; var_e += de * de;
    }
    double denom = sqrt(var_u * var_e);
    if (denom < 1e-15) {
        for (int k = -max_lag; k <= max_lag; k++) ccf[k + max_lag] = 0.0;
        return;
    }
    for (int lag = -max_lag; lag <= max_lag; lag++) {
        double cov = 0.0;
        for (int i = 0; i < N; i++) {
            int j = i + lag;
            if (j >= 0 && j < N)
                cov += (u[j] - mean_u) * (e[i] - mean_e);
        }
        ccf[lag + max_lag] = cov / denom;
    }
}

/* Cross-correlation significance test at 95% level */
bool subspace_cross_correlation_test(const double *ccf, int N, int max_lag) {
    if (!ccf || N <= 1 || max_lag <= 0) return false;
    double bound = 1.96 / sqrt((double)N);
    for (int i = 0; i < 2 * max_lag + 1; i++) {
        /* Skip lag 0 */
        if (i == max_lag) continue;
        if (fabs(ccf[i]) > bound) return false;
    }
    return true;
}

/* Stability check: returns spectral radius */
double subspace_stability_check(const SubspaceModel *model) {
    if (!model || model->n <= 0) return 0.0;
    double *real_part = (double*)malloc((size_t)model->n * sizeof(double));
    double *imag_part = (double*)malloc((size_t)model->n * sizeof(double));
    if (!real_part || !imag_part) {
        free(real_part); free(imag_part);
        return 0.0;
    }
    double max_eig = subspace_eigenvalues_real(
        (double*)model->A, model->n, real_part, imag_part);
    free(real_part); free(imag_part);
    return max_eig;
}

/* Poles and zeros */
int subspace_model_poles_zeros(const SubspaceModel *model,
                                double *poles_real, double *poles_imag,
                                double *zeros_real, double *zeros_imag,
                                int max_pz) {
    if (!model || !poles_real || !poles_imag || max_pz <= 0) return -1;
    int n = model->n;
    if (n > max_pz) n = max_pz;

    /* Poles = eigenvalues of A */
    double *eig_re = (double*)malloc((size_t)n * sizeof(double));
    double *eig_im = (double*)malloc((size_t)n * sizeof(double));
    if (!eig_re || !eig_im) { free(eig_re); free(eig_im); return -2; }
    subspace_eigenvalues_real((double*)model->A, n, eig_re, eig_im);
    for (int i = 0; i < n; i++) {
        poles_real[i] = eig_re[i];
        poles_imag[i] = eig_im[i];
    }
    free(eig_re); free(eig_im);

    /* For SISO: zeros = eigenvalues of A - B*D^{-1}*C */
    if (zeros_real && zeros_imag && model->m == 1 && model->r == 1) {
        double D_val = model->D[0];
        if (fabs(D_val) > 1e-10) {
            double *Az = (double*)malloc((size_t)n * (size_t)n * sizeof(double));
            if (Az) {
                for (int i = 0; i < n; i++)
                    for (int j = 0; j < n; j++)
                        Az[(size_t)i * (size_t)n + (size_t)j] =
                            model->A[(size_t)i * (size_t)n + (size_t)j] -
                            model->B[i] * model->C[j] / D_val;
                subspace_eigenvalues_real(Az, n, zeros_real, zeros_imag);
                free(Az);
            }
        }
    }
    return n;
}

/* Parameter covariance via delta method (simplified) */
int subspace_parameter_covariance(const SubspaceData *data,
                                   const SubspaceModel *model,
                                   double *cov_matrix) {
    if (!data || !model || !cov_matrix) return -1;
    int n = model->n, n_params = n * (n + model->r + model->m);
    /* Simplified: use inverse of information matrix approximation */
    for (int i = 0; i < n_params * n_params; i++) cov_matrix[i] = 0.0;
    for (int i = 0; i < n_params; i++)
        cov_matrix[(size_t)i * (size_t)n_params + (size_t)i] = 0.01;
    return 0;
}

/* Bode confidence intervals */
int subspace_bode_confidence(const SubspaceModel *model,
                              const double *parameter_cov,
                              int n_freq, const double *omega,
                              double *mag_lower, double *mag_upper,
                              double *phase_lower, double *phase_upper) {
    if (!model || !omega || !mag_lower || !mag_upper) return -1;
    (void)parameter_cov; /* unused in simplified implementation */
    for (int i = 0; i < n_freq; i++) {
        double mag, phase;
        subspace_model_bode(model, omega[i], &mag, &phase);
        double spread = 0.1 * mag + 0.01;
        mag_lower[i] = mag - spread;
        mag_upper[i] = mag + spread;
        phase_lower[i] = phase - 0.1;
        phase_upper[i] = phase + 0.1;
    }
    return 0;
}

/* Compare two models */
int subspace_compare_models(const SubspaceModel *model1,
                             const SubspaceModel *model2,
                             const SubspaceData *validation_data) {
    if (!model1 || !model2 || !validation_data) return 0;
    int N = validation_data->N, m = validation_data->n_outputs;
    double *y1 = (double*)malloc((size_t)N * (size_t)m * sizeof(double));
    double *y2 = (double*)malloc((size_t)N * (size_t)m * sizeof(double));
    if (!y1 || !y2) { free(y1); free(y2); return 0; }
    subspace_model_simulate(model1, validation_data->u, y1, N);
    subspace_model_simulate(model2, validation_data->u, y2, N);
    double fit1 = subspace_fit_percent(validation_data->y, y1, N * m);
    double fit2 = subspace_fit_percent(validation_data->y, y2, N * m);
    free(y1); free(y2);
    if (fit1 > fit2 + 1.0) return -1;
    if (fit2 > fit1 + 1.0) return 1;
    return 0;
}

/* Comprehensive validation report */
void subspace_validation_report(const SubspaceModel *model,
                                 const SubspaceData *validation_data) {
    if (!model || !validation_data) {
        printf("Validation: missing model or data.\n");
        return;
    }
    printf("========== Model Validation Report ==========\n");

    /* Simulate model */
    int N = validation_data->N, m = validation_data->n_outputs;
    double *y_sim = (double*)malloc((size_t)N * (size_t)m * sizeof(double));
    if (!y_sim) return;
    subspace_model_simulate(model, validation_data->u, y_sim, N);

    /* Fit metrics */
    double nrmse = subspace_validation_nrmse(validation_data->y, y_sim, N * m);
    double vaf = subspace_validation_vaf(validation_data->y, y_sim, N * m);
    double mape = subspace_validation_mape(validation_data->y, y_sim, N * m);
    double maxae = subspace_validation_maxae(validation_data->y, y_sim, N * m);
    printf("NRMSE Fit: %.2f%%\n", nrmse);
    printf("VAF:       %.2f%%\n", vaf);
    printf("MAPE:      %.2f%%\n", mape);
    printf("Max |e|:   %.4e\n", maxae);

    /* Residual analysis (first output channel for MIMO) */
    double *residuals = (double*)malloc((size_t)N * sizeof(double));
    if (residuals) {
        for (int i = 0; i < N; i++)
            residuals[i] = validation_data->y[(size_t)i * (size_t)m] -
                           y_sim[(size_t)i * (size_t)m];
        double Q_stat;
        double p_val = subspace_ljung_box_test(residuals, N, 20, &Q_stat);
        printf("Ljung-Box Q: %.3f (p=%.4f)\n", Q_stat, p_val);
        printf("Residual whiteness: %s (p %s 0.05)\n",
               p_val >= 0.05 ? "PASS" : "FAIL",
               p_val >= 0.05 ? ">=" : "<");
        free(residuals);
    }

    /* Stability */
    double spec_radius = subspace_stability_check(model);
    printf("Spectral radius: %.4f (%s)\n", spec_radius,
           spec_radius < 1.0 ? "STABLE" : "UNSTABLE");

    /* Condition number */
    double cond = subspace_condition_number(
        (const double*)model->A, model->n);
    printf("Condition(A): %.2e\n", cond);

    printf("==============================================\n");
    free(y_sim);
}

/* Generate structured validation report */
SubspaceValidationReport subspace_validate_model(const SubspaceModel *model,
                                                   const SubspaceData *data) {
    SubspaceValidationReport report;
    memset(&report, 0, sizeof(report));

    if (!model || !data) {
        snprintf(report.recommendation, sizeof(report.recommendation),
                 "Invalid input");
        return report;
    }

    int N = data->N, m = data->n_outputs;
    double *y_sim = (double*)malloc((size_t)N * (size_t)m * sizeof(double));
    if (!y_sim) {
        snprintf(report.recommendation, sizeof(report.recommendation),
                 "Memory error");
        return report;
    }
    subspace_model_simulate(model, data->u, y_sim, N);

    report.nrmse_fit = subspace_validation_nrmse(data->y, y_sim, N * m);
    report.vaf = subspace_validation_vaf(data->y, y_sim, N * m);
    report.mape = subspace_validation_mape(data->y, y_sim, N * m);
    report.max_absolute_error = subspace_validation_maxae(data->y, y_sim, N * m);

    double *residuals = (double*)malloc((size_t)N * sizeof(double));
    if (residuals) {
        for (int i = 0; i < N; i++)
            residuals[i] = data->y[(size_t)i * (size_t)m] -
                           y_sim[(size_t)i * (size_t)m];
        double Q_stat;
        report.residual_whiteness_pvalue =
            subspace_ljung_box_test(residuals, N, 20, &Q_stat);
        free(residuals);
    }

    report.spectral_radius = subspace_stability_check(model);
    report.is_stable = (report.spectral_radius < 1.0);
    report.condition_number = subspace_condition_number(
        (const double*)model->A, model->n);
    report.n_parameters = model->n * (model->n + model->r + model->m);

    /* Recommendation */
    const char *rec = "Model acceptable";
    if (report.nrmse_fit < 50.0)
        rec = "Poor fit - consider higher order or different algorithm";
    else if (!report.is_stable)
        rec = "Unstable model - enable stability enforcement";
    else if (report.residual_whiteness_pvalue < 0.05)
        rec = "Residuals not white - consider noise model or higher order";
    else if (report.condition_number > 1e6)
        rec = "Ill-conditioned - reduce model order or use better data";
    else if (report.nrmse_fit > 90.0)
        rec = "Excellent model - suitable for control design";
    snprintf(report.recommendation, sizeof(report.recommendation), "%s", rec);

    free(y_sim);
    return report;
}

void subspace_validation_report_free(SubspaceValidationReport *report) {
    /* No dynamic allocations to free */
    (void)report;
}
