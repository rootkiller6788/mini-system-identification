/**
 * wh_validation.c ? Model Validation
 *
 * Implements FIT computation, residual analysis (whiteness, independence,
 * higher-order correlation tests), cross-validation, stability and
 * frequency-domain validation.
 *
 * Knowledge Level: L4 (Fundamental Laws), L6 (Canonical Problems)
 */

#include "wh_validation.h"
#include "wh_simulation.h"
#include "wh_identification.h"
#include "wh_nonlinear.h"
#include "wh_signal.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ??? FIT metric ????????????????????????????????????????????????????????? */

double wh_validate_fit(const WH_Model* model,
                        const double* u, const double* y_ref,
                        int n, const WH_SimConfig* config) {
    if (!model || !u || !y_ref || n <= 0) return -1e100;

    WH_SimOutput output;
    memset(&output, 0, sizeof(WH_SimOutput));

    WH_SimConfig cfg = config ? *config : wh_sim_config_default();

    if (wh_sim_run_with_reference(model, u, y_ref, n, &cfg, &output) == 0) {
        double fit = output.fit_percent;
        wh_sim_output_free(&output);
        return fit;
    }
    return -1e100;
}

double wh_validate_multi_fit(const WH_Model* model,
                              const double** u_sets,
                              const double** y_sets,
                              int n_per_set, int k,
                              const WH_SimConfig* config,
                              double* individual_fits) {
    if (!model || !u_sets || !y_sets || k <= 0) return -1e100;

    double sum_fit = 0.0;
    for (int i = 0; i < k; i++) {
        double fit = wh_validate_fit(model, u_sets[i], y_sets[i],
                                      n_per_set, config);
        if (individual_fits) individual_fits[i] = fit;
        sum_fit += fit;
    }
    return sum_fit / k;
}

/* ??? Residual analysis ?????????????????????????????????????????????????? */

int wh_validate_residuals(const WH_Model* model,
                           const double* u, const double* y_ref,
                           int n, int max_lag,
                           const WH_SimConfig* config,
                           WH_ResidualAnalysis* result) {
    if (!model || !u || !y_ref || n <= 0 || !result) return -1;
    memset(result, 0, sizeof(WH_ResidualAnalysis));

    if (max_lag <= 0) max_lag = 25;
    if (max_lag > n / 4) max_lag = n / 4;
    if (max_lag > 100) max_lag = 100;
    result->max_lag = max_lag;

    /* Simulate model */
    WH_SimOutput output;
    memset(&output, 0, sizeof(WH_SimOutput));
    WH_SimConfig cfg = config ? *config : wh_sim_config_default();
    cfg.add_noise = 0; /* No extra noise during validation */

    if (wh_sim_run(model, u, n, &cfg, &output) != 0) return -1;

    /* Compute residuals */
    double* residuals = (double*)calloc(n, sizeof(double));
    if (!residuals) { wh_sim_output_free(&output); return -1; }

    double sum_e = 0.0, sum_e2 = 0.0;
    int n_eff = output.n_samples;
    if (n_eff <= 0) { free(residuals); wh_sim_output_free(&output); return -1; }

    for (int i = 0; i < n_eff; i++) {
        residuals[i] = y_ref[i] - output.y[i];
        sum_e += residuals[i];
        sum_e2 += residuals[i] * residuals[i];
    }
    result->mean = sum_e / n_eff;
    result->variance = sum_e2 / n_eff - result->mean * result->mean;
    if (result->variance < 0.0) result->variance = 0.0;

    /* Autocorrelation R_??(?) */
    result->auto_corr = (double*)calloc(max_lag + 1, sizeof(double));
    if (result->auto_corr) {
        for (int tau = 0; tau <= max_lag; tau++) {
            result->auto_corr[tau] =
                wh_signal_autocorrelation(residuals, n_eff, tau);
            /* Normalize by R_??(0) */
            if (result->auto_corr[0] > 1e-12) {
                result->auto_corr[tau] /= result->auto_corr[0];
            }
        }
        /* Ljung-Box test */
        double Q = 0.0;
        for (int tau = 1; tau <= max_lag; tau++) {
            Q += result->auto_corr[tau] * result->auto_corr[tau]
                 / (n_eff - tau);
        }
        Q *= n_eff * (n_eff + 2);
        /* Chi-squared approximation with max_lag degrees of freedom */
        /* P-value approximation via normal approximation */
        double z = (Q - max_lag) / sqrt(2.0 * max_lag);
        /* One-sided p-value using standard normal CDF approximation */
        result->whiteness_pvalue = 1.0 / (1.0 + exp(0.7 * z + 0.6 * z * z * z));
        /* 95% confidence bound: ?1.96/?N */
        double bound_95 = 1.96 / sqrt((double)n_eff);
        for (int tau = 1; tau <= max_lag; tau++) {
            if (fabs(result->auto_corr[tau]) > bound_95) {
                result->is_white_95 = 0;
                break;
            }
            if (tau == max_lag) result->is_white_95 = 1;
        }
    }

    /* Cross-correlation R_?u(?) for ? ? 0 */
    result->cross_corr_eu = (double*)calloc(2 * max_lag + 1, sizeof(double));
    if (result->cross_corr_eu) {
        result->is_independent_95 = 1;
        double bound_95 = 1.96 / sqrt((double)n_eff);
        for (int tau = 0; tau <= max_lag; tau++) {
            double sum = 0.0;
            for (int i = tau; i < n_eff; i++) {
                sum += residuals[i] * u[i - tau];
            }
            result->cross_corr_eu[max_lag + tau] = sum / (n_eff - tau);
            /* Normalize */
            if (result->auto_corr && result->auto_corr[0] > 1e-12) {
                double norm = sqrt(result->auto_corr[0] * result->variance) * n_eff;
                if (norm > 1e-12) {
                    result->cross_corr_eu[max_lag + tau] /= norm;
                }
            }
            if (fabs(result->cross_corr_eu[max_lag + tau]) > bound_95) {
                result->is_independent_95 = 0;
            }
        }
    }

    /* Higher-order nonlinear correlation test */
    result->cross_corr_e2u2 = (double*)calloc(2 * max_lag + 1, sizeof(double));
    result->passes_nl_test = 1;
    if (result->cross_corr_e2u2) {
        double bound_95 = 1.96 / sqrt((double)n_eff);
        for (int tau = 0; tau <= max_lag; tau++) {
            double sum = 0.0;
            for (int i = tau; i < n_eff; i++) {
                double e2_prime = residuals[i] * residuals[i] - result->variance;
                double u2_prime = u[i - tau] * u[i - tau];
                double u2_mean = 0.0;
                for (int j = 0; j < n_eff; j++) u2_mean += u[j] * u[j];
                u2_mean /= n_eff;
                u2_prime -= u2_mean;
                sum += e2_prime * u2_prime;
            }
            result->cross_corr_e2u2[max_lag + tau] = sum / (n_eff - tau);
            if (fabs(result->cross_corr_e2u2[max_lag + tau]) > bound_95) {
                result->passes_nl_test = 0;
            }
        }
    }

    free(residuals);
    wh_sim_output_free(&output);
    return 0;
}

void wh_validate_residuals_free(WH_ResidualAnalysis* result) {
    if (!result) return;
    free(result->auto_corr);
    free(result->cross_corr_eu);
    free(result->cross_corr_e2u2);
    memset(result, 0, sizeof(WH_ResidualAnalysis));
}

/* ??? Cross-validation ??????????????????????????????????????????????????? */

int wh_validate_crossval(const double* u, const double* y, int n,
                          int k_folds,
                          const WH_IdentConfig* id_config,
                          const WH_SimConfig* sim_config,
                          WH_CrossValidation* result) {
    if (!u || !y || n <= 0 || k_folds < 2 || !result) return -1;
    memset(result, 0, sizeof(WH_CrossValidation));

    result->k_folds = k_folds;
    result->fold_fits = (double*)calloc(k_folds, sizeof(double));
    if (!result->fold_fits) return -1;

    int fold_size = n / k_folds;
    if (fold_size < 10) { free(result->fold_fits); return -1; }

    double sum_fit = 0.0, sum_fit2 = 0.0;
    double min_fit = 1e100, max_fit = -1e100;

    for (int fold = 0; fold < k_folds; fold++) {
        /* Split data */
        int test_start = fold * fold_size;
        int test_end = (fold == k_folds - 1) ? n : test_start + fold_size;
        int n_test = test_end - test_start;
        int n_train = n - n_test;

        /* Allocate training and test sets */
        double* u_train = (double*)calloc(n_train, sizeof(double));
        double* y_train = (double*)calloc(n_train, sizeof(double));
        double* u_test = (double*)calloc(n_test, sizeof(double));
        double* y_test = (double*)calloc(n_test, sizeof(double));

        if (!u_train || !y_train || !u_test || !y_test) {
            free(u_train); free(y_train); free(u_test); free(y_test);
            free(result->fold_fits);
            return -1;
        }

        /* Copy training data (all except fold) */
        int ti = 0;
        for (int i = 0; i < n; i++) {
            if (i >= test_start && i < test_end) continue;
            u_train[ti] = u[i];
            y_train[ti] = y[i];
            ti++;
        }

        /* Copy test data */
        for (int i = 0; i < n_test; i++) {
            u_test[i] = u[test_start + i];
            y_test[i] = y[test_start + i];
        }

        /* Identify model on training data */
        WH_IdentConfig cfg = id_config ? *id_config : wh_ident_config_default();
        WH_IdentResult ident_res;
        memset(&ident_res, 0, sizeof(WH_IdentResult));

        double fit = -1e100;
        if (wh_identify(u_train, y_train, n_train, &cfg, &ident_res) == 0
            && ident_res.model) {
            /* Evaluate on test data */
            fit = wh_validate_fit(ident_res.model, u_test, y_test,
                                   n_test, sim_config);
            result->fold_fits[fold] = fit;
            wh_ident_result_free(&ident_res);
        }

        free(u_train); free(y_train); free(u_test); free(y_test);

        if (fit > -1e6) {
            sum_fit += fit;
            sum_fit2 += fit * fit;
            if (fit < min_fit) min_fit = fit;
            if (fit > max_fit) max_fit = fit;
        }
    }

    result->mean_fit = sum_fit / k_folds;
    result->std_fit = sqrt(sum_fit2 / k_folds - result->mean_fit * result->mean_fit);
    if (result->std_fit < 0.0) result->std_fit = 0.0;
    result->min_fit = min_fit;
    result->max_fit = max_fit;
    result->is_reliable = (result->std_fit < 15.0); /* < 15% std deviation */

    return 0;
}

void wh_validate_crossval_free(WH_CrossValidation* result) {
    if (!result) return;
    free(result->fold_fits);
    memset(result, 0, sizeof(WH_CrossValidation));
}

/* ??? Stability and pole-zero checks ????????????????????????????????????? */

int wh_validate_stability(const WH_Model* model) {
    return wh_model_is_stable(model);
}

int wh_validate_delay(const WH_Model* model, int max_delay) {
    if (!model) return 0;
    return wh_model_get_delay(model) <= max_delay;
}

int wh_validate_monotonic(const WH_Model* model) {
    if (!model) return 0;
    return wh_nl_is_monotonic(&model->N);
}

/* ??? Frequency-domain validation ???????????????????????????????????????? */

int wh_validate_frequency(const WH_Model* model,
                           const double* u, const double* y,
                           int n_period,
                           double* max_freq_error,
                           double* mean_freq_error) {
    if (!model || !u || !y || n_period <= 0
        || !max_freq_error || !mean_freq_error) return -1;

    /* Simulate model on same input */
    double* y_model = (double*)calloc(n_period, sizeof(double));
    if (!y_model) return -1;

    WH_Model sim_model;
    memcpy(&sim_model, model, sizeof(WH_Model));
    wh_model_reset(&sim_model);
    for (int i = 0; i < n_period; i++) {
        y_model[i] = wh_model_evaluate(&sim_model, u[i]);
    }

    /* Simple frequency-domain comparison via DFT */
    /* Compute magnitude at a few representative frequencies */
    double max_err = 0.0, sum_err = 0.0;
    int n_freq = n_period / 8;
    if (n_freq < 4) n_freq = 4;
    if (n_freq > 128) n_freq = 128;

    for (int k = 1; k <= n_freq; k++) {
        double re_data = 0.0, im_data = 0.0;
        double re_model = 0.0, im_model = 0.0;

        for (int i = 0; i < n_period; i++) {
            double angle = 2.0 * M_PI * k * i / n_period;
            re_data += y[i] * cos(angle);
            im_data -= y[i] * sin(angle);
            re_model += y_model[i] * cos(angle);
            im_model -= y_model[i] * sin(angle);
        }

        double mag_data = sqrt(re_data * re_data + im_data * im_data) / n_period;
        double mag_model = sqrt(re_model * re_model + im_model * im_model) / n_period;

        double err_db = 20.0 * log10((mag_data + 1e-12) / (mag_model + 1e-12));
        double abs_err = fabs(err_db);
        if (abs_err > max_err) max_err = abs_err;
        sum_err += abs_err;
    }

    *max_freq_error = max_err;
    *mean_freq_error = sum_err / n_freq;

    free(y_model);
    return 0;
}

/* ??? Comprehensive validation report ???????????????????????????????????? */

int wh_validate_comprehensive(const WH_Model* model,
                               const double* u, const double* y,
                               int n, int k_folds,
                               WH_ValidationReport* report) {
    if (!model || !u || !y || n <= 0 || !report) return -1;
    memset(report, 0, sizeof(WH_ValidationReport));
    report->n_tests_total = 7;

    /* 1. FIT metric */
    WH_SimConfig sim_cfg = wh_sim_config_default();
    report->fit_percent = wh_validate_fit(model, u, y, n, &sim_cfg);

    /* 2. MSE / RMSE */
    WH_SimOutput output;
    memset(&output, 0, sizeof(WH_SimOutput));
    if (wh_sim_run_with_reference(model, u, y, n, &sim_cfg, &output) == 0) {
        report->mse = output.mse;
        report->rmse = sqrt(output.mse);
        wh_sim_output_free(&output);
    }

    /* 3. Stability */
    report->is_stable = wh_validate_stability(model);
    if (report->is_stable) report->n_tests_passed++;

    /* 4. Residual analysis */
    WH_ResidualAnalysis res_analysis;
    memset(&res_analysis, 0, sizeof(res_analysis));
    if (wh_validate_residuals(model, u, y, n, 20, &sim_cfg, &res_analysis) == 0) {
        report->passes_residuals = res_analysis.is_white_95
                                    && res_analysis.is_independent_95;
        if (report->passes_residuals) report->n_tests_passed++;
        wh_validate_residuals_free(&res_analysis);
    }

    /* 5. Cross-validation */
    WH_IdentConfig id_cfg = wh_ident_config_default();
    WH_CrossValidation cv;
    memset(&cv, 0, sizeof(cv));
    if (wh_validate_crossval(u, y, n, k_folds, &id_cfg, &sim_cfg, &cv) == 0) {
        report->crossval_mean_fit = cv.mean_fit;
        report->crossval_std_fit = cv.std_fit;
        if (cv.is_reliable) report->n_tests_passed++;
        wh_validate_crossval_free(&cv);
    }

    /* 6. Monotonicity of nonlinearity */
    report->is_monotonic = wh_validate_monotonic(model);
    if (report->is_monotonic) report->n_tests_passed++;

    /* 7. Frequency-domain check (use first 1/4 of data) */
    int n_period = n / 4;
    if (n_period > 100) {
        wh_validate_frequency(model, u, y, n_period,
                               &report->max_freq_error_db,
                               &report->mean_freq_error_db);
        if (report->mean_freq_error_db < 20.0) report->n_tests_passed++;
    }

    /* Overall verdict */
    report->model_is_valid = (report->n_tests_passed >= 5);
    return 0;
}

void wh_validate_report_print(const WH_ValidationReport* report) {
    if (!report) return;
    printf("??? WH Model Validation Report ??????????????????????????????\n");
    printf("?  FIT: %.2f%%  MSE: %.6f  RMSE: %.6f\n",
           report->fit_percent, report->mse, report->rmse);
    printf("?  Stability: %s\n", report->is_stable ? "PASS" : "FAIL");
    printf("?  Residual whiteness/independence: %s\n",
           report->passes_residuals ? "PASS" : "FAIL");
    printf("?  Cross-val FIT: %.2f%% ? %.2f%%\n",
           report->crossval_mean_fit, report->crossval_std_fit);
    printf("?  NL monotonicity: %s\n",
           report->is_monotonic ? "PASS" : "N/A");
    printf("?  Freq error: max %.2f dB, mean %.2f dB\n",
           report->max_freq_error_db, report->mean_freq_error_db);
    printf("?  Tests passed: %d/%d\n",
           report->n_tests_passed, report->n_tests_total);
    printf("?  OVERALL: %s\n",
           report->model_is_valid ? "VALID ?" : "INVALID ??");
    printf("????????????????????????????????????????????????????????????????\n");
}
