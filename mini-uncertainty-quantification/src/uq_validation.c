#include "uq_validation.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define UQ_PI 3.14159265358979323846

/* ============================================================================
 * Validation Metrics
 * ============================================================================ */

UQValidationResult uq_validate_compute(double* observed, double* predicted,
    int n, UQValidationMetric metric) {
    UQValidationResult result = {0};
    result.metric = metric;

    switch (metric) {
    case UQ_VAL_RMSE: return uq_validate_rmse(observed, predicted, n);
    case UQ_VAL_MAE: return uq_validate_mae(observed, predicted, n);
    case UQ_VAL_MAPE: return uq_validate_mape(observed, predicted, n);
    case UQ_VAL_R_SQUARED: return uq_validate_r_squared(observed, predicted, n, 1);
    case UQ_VAL_CONCORDANCE: return uq_validate_concordance(observed, predicted, n);
    case UQ_VAL_THEIL_U: return uq_validate_theil_u(observed, predicted, n);
    default: return result;
    }
}

UQValidationResult uq_validate_residual(double* observed, double* predicted,
    int n) {
    UQValidationResult result = {0};
    result.metric = UQ_VAL_RESIDUAL;
    double sum = 0.0;
    for (int i = 0; i < n; i++)
        sum += observed[i] - predicted[i];
    result.value = sum / n;
    result.passed = fabs(result.value) < 0.05 * fabs(sum / n + 1e-15);
    return result;
}

UQValidationResult uq_validate_rmse(double* observed, double* predicted,
    int n) {
    UQValidationResult result = {0};
    result.metric = UQ_VAL_RMSE;
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        double d = observed[i] - predicted[i];
        sum += d * d;
    }
    result.value = sqrt(sum / n);
    result.passed = true;
    return result;
}

UQValidationResult uq_validate_mae(double* observed, double* predicted, int n) {
    UQValidationResult result = {0};
    result.metric = UQ_VAL_MAE;
    double sum = 0.0;
    for (int i = 0; i < n; i++)
        sum += fabs(observed[i] - predicted[i]);
    result.value = sum / n;
    result.passed = true;
    return result;
}

UQValidationResult uq_validate_mape(double* observed, double* predicted, int n) {
    UQValidationResult result = {0};
    result.metric = UQ_VAL_MAPE;
    double sum = 0.0;
    int count = 0;
    for (int i = 0; i < n; i++) {
        if (fabs(observed[i]) > 1e-12) {
            sum += fabs((observed[i] - predicted[i]) / observed[i]);
            count++;
        }
    }
    result.value = (count > 0) ? 100.0 * sum / count : INFINITY;
    result.passed = result.value < 20.0;
    return result;
}

UQValidationResult uq_validate_r_squared(double* observed, double* predicted,
    int n, int n_params) {
    UQValidationResult result = {0};
    result.metric = UQ_VAL_R_SQUARED;
    double y_bar = 0.0;
    for (int i = 0; i < n; i++) y_bar += observed[i];
    y_bar /= n;
    double sst = 0.0, sse = 0.0;
    for (int i = 0; i < n; i++) {
        double d = observed[i] - y_bar;
        sst += d * d;
        double e = observed[i] - predicted[i];
        sse += e * e;
    }
    result.value = 1.0 - sse / (sst + 1e-15);
    result.passed = result.value > 0.5;
    (void)n_params;
    return result;
}

UQValidationResult uq_validate_concordance(double* observed, double* predicted,
    int n) {
    /* Lin's Concordance Correlation Coefficient */
    UQValidationResult result = {0};
    result.metric = UQ_VAL_CONCORDANCE;
    double mo = 0.0, mp = 0.0;
    for (int i = 0; i < n; i++) { mo += observed[i]; mp += predicted[i]; }
    mo /= n; mp /= n;

    double so2 = 0.0, sp2 = 0.0, sop = 0.0;
    for (int i = 0; i < n; i++) {
        double do_ = observed[i] - mo;
        double dp = predicted[i] - mp;
        so2 += do_ * do_;
        sp2 += dp * dp;
        sop += do_ * dp;
    }
    so2 /= n; sp2 /= n; sop /= n;

    double pearson_r = sop / sqrt(so2 * sp2 + 1e-15);
    result.value = 2.0 * pearson_r * sqrt(so2 * sp2)
                   / (so2 + sp2 + (mo - mp) * (mo - mp) + 1e-15);
    result.passed = result.value > 0.8;
    return result;
}

UQValidationResult uq_validate_theil_u(double* observed, double* predicted,
    int n) {
    UQValidationResult result = {0};
    result.metric = UQ_VAL_THEIL_U;
    double mse = 0.0, naive = 0.0;
    for (int i = 1; i < n; i++) {
        double err = predicted[i] - observed[i];
        double naive_err = observed[i] - observed[i-1];
        mse += err * err;
        naive += naive_err * naive_err;
    }
    result.value = sqrt(mse / naive + 1e-15);
    result.passed = result.value < 1.0;
    return result;
}

UQValidationResult uq_validate_mase(double* observed, double* predicted,
    int n, int seasonal_period) {
    UQValidationResult result = {0};
    result.metric = UQ_VAL_MASE;
    double mae = 0.0, naive_mae = 0.0;
    for (int i = 0; i < n; i++)
        mae += fabs(observed[i] - predicted[i]);
    mae /= n;

    if (seasonal_period <= 0) seasonal_period = 1;
    for (int i = seasonal_period; i < n; i++)
        naive_mae += fabs(observed[i] - observed[i - seasonal_period]);
    naive_mae /= (n - seasonal_period);

    result.value = mae / (naive_mae + 1e-15);
    result.passed = result.value < 1.0;
    return result;
}

/* ============================================================================
 * Predictive Assessment
 * ============================================================================ */

UQPredictiveAssessment* uq_predictive_create(UQLinearModel* lm,
    double* X_pred, int n_pred, int p) {
    UQPredictiveAssessment* pa = (UQPredictiveAssessment*)calloc(1, sizeof(UQPredictiveAssessment));
    pa->model = lm;
    pa->n_predictions = n_pred;
    pa->predictions = (double*)calloc(n_pred, sizeof(double));
    pa->prediction_variance = (double*)calloc(n_pred, sizeof(double));
    pa->residuals = (double*)calloc(n_pred, sizeof(double));
    pa->pi_lower = (double*)calloc(n_pred, sizeof(double));
    pa->pi_upper = (double*)calloc(n_pred, sizeof(double));
    pa->cv_predictions = (double*)calloc(n_pred, sizeof(double));
    (void)X_pred; (void)p;
    return pa;
}

void uq_predictive_free(UQPredictiveAssessment* pa) {
    if (!pa) return;
    free(pa->predictions);
    free(pa->prediction_variance);
    free(pa->residuals);
    free(pa->pi_lower);
    free(pa->pi_upper);
    free(pa->cv_predictions);
    free(pa->outlier_indices);
    free(pa);
}

void uq_predictive_compute(UQPredictiveAssessment* pa, double* y_obs) {
    if (!pa || !pa->model) return;
    for (int i = 0; i < pa->n_predictions; i++) {
        double yh, se_fit, se_pred;
        /* Note: requires X_pred row i — use model coefficients for prediction */
        yh = pa->model->coefficients->components[0]; /* Intercept only simplified */
        pa->predictions[i] = yh;
    }
    (void)y_obs;
}

void uq_prediction_intervals(UQPredictiveAssessment* pa, double confidence) {
    if (!pa || !pa->model) return;
    double t_val = 1.96; /* Normal approximation */
    double sigma = sqrt(pa->model->sigma_squared);
    for (int i = 0; i < pa->n_predictions; i++) {
        double se = sigma;
        pa->pi_lower[i] = pa->predictions[i] - t_val * se;
        pa->pi_upper[i] = pa->predictions[i] + t_val * se;
    }

    /* Coverage */
    int covered = 0;
    for (int i = 0; i < pa->n_predictions; i++)
        if (pa->residuals[i] >= pa->pi_lower[i] && pa->residuals[i] <= pa->pi_upper[i])
            covered++;
    pa->pi_coverage = (double)covered / pa->n_predictions;
    pa->pi_average_width = 0.0;
    for (int i = 0; i < pa->n_predictions; i++)
        pa->pi_average_width += pa->pi_upper[i] - pa->pi_lower[i];
    pa->pi_average_width /= pa->n_predictions;
    (void)confidence;
}

void uq_residual_diagnostics(UQPredictiveAssessment* pa, double* y_obs, int n) {
    for (int i = 0; i < n; i++)
        pa->residuals[i] = y_obs[i] - pa->predictions[i];
    uq_durbin_watson(pa);
    uq_shapiro_wilk_test(pa->residuals, n, &pa->shapiro_wilk_stat, &pa->shapiro_wilk_p);
    pa->residuals_normal = pa->shapiro_wilk_p > 0.05;
}

void uq_cross_validate(UQPredictiveAssessment* pa, double* X, double* y,
                       int n, int p) {
    double prss = 0.0;
    for (int i = 0; i < n; i++) {
        double h = pa->model->leverages[i];
        double r = pa->model->residuals[i];
        double r_loo = r / (1.0 - h + 1e-15);
        prss += r_loo * r_loo;
    }
    pa->cv_rmse = sqrt(prss / n);
    (void)X; (void)y; (void)p;
}

void uq_durbin_watson(UQPredictiveAssessment* pa) {
    double num = 0.0, den = 0.0;
    for (int i = 0; i < pa->n_predictions - 1; i++)
        num += (pa->residuals[i+1] - pa->residuals[i])
               * (pa->residuals[i+1] - pa->residuals[i]);
    for (int i = 0; i < pa->n_predictions; i++)
        den += pa->residuals[i] * pa->residuals[i];
    pa->dw_statistic = num / (den + 1e-15);
    pa->residuals_independent = pa->dw_statistic > 1.5 && pa->dw_statistic < 2.5;
}

void uq_shapiro_wilk_test(double* residuals, int n, double* W, double* p_value) {
    /* Simplified SW using correlation of sorted residuals with expected normal quantiles */
    double* sorted = (double*)malloc(n * sizeof(double));
    memcpy(sorted, residuals, n * sizeof(double));
    for (int i = 0; i < n - 1; i++)
        for (int j = i + 1; j < n; j++)
            if (sorted[i] > sorted[j]) { double t = sorted[i]; sorted[i] = sorted[j]; sorted[j] = t; }

    double* expected = (double*)malloc(n * sizeof(double));
    for (int i = 0; i < n; i++)
        expected[i] = uq_stats_normal_quantile((i + 0.5) / n);

    double r = uq_stats_correlation(sorted, expected, n);
    *W = r * r;
    *p_value = (r < 0.9) ? 0.01 : 0.50; /* Very approximate */
    free(sorted);
    free(expected);
}

void uq_breusch_pagan_test(UQPredictiveAssessment* pa, double* X, int n, int p) {
    /* Fit squared residuals ~ X to test heteroskedasticity */
    double rs2 = 0.0;
    for (int i = 0; i < n; i++)
        rs2 += pa->residuals[i] * pa->residuals[i];
    double sigma2 = rs2 / n;

    double ss_reg = 0.0;
    for (int i = 0; i < n; i++) {
        double e2 = pa->residuals[i] * pa->residuals[i] / sigma2 - 1.0;
        ss_reg += e2 * e2;
    }
    pa->bp_statistic = 0.5 * ss_reg;
    pa->homoskedastic = pa->bp_statistic < 3.84; /* χ²_1, α=0.05 */
    (void)X; (void)p;
}

void uq_outlier_detect(UQPredictiveAssessment* pa, double threshold) {
    int cap = pa->n_predictions;
    pa->outlier_indices = (int*)malloc(cap * sizeof(int));
    pa->n_outliers = 0;
    for (int i = 0; i < pa->n_predictions; i++) {
        if (fabs(pa->model->studentized_residuals[i]) > threshold) {
            if (pa->n_outliers < cap)
                pa->outlier_indices[pa->n_outliers++] = i;
        }
    }
}

/* ============================================================================
 * Verification Suite
 * ============================================================================ */

UQVerificationSuite* uq_verify_create(int n_cases) {
    UQVerificationSuite* suite = (UQVerificationSuite*)calloc(1, sizeof(UQVerificationSuite));
    suite->n_test_cases = n_cases;
    suite->case_names = (char**)calloc(n_cases, sizeof(char*));
    suite->exact_solutions = (double*)calloc(n_cases, sizeof(double));
    suite->computed_solutions = (double*)calloc(n_cases, sizeof(double));
    suite->absolute_errors = (double*)calloc(n_cases, sizeof(double));
    suite->relative_errors = (double*)calloc(n_cases, sizeof(double));
    suite->passed = (bool*)calloc(n_cases, sizeof(bool));
    return suite;
}

void uq_verify_free(UQVerificationSuite* suite) {
    if (!suite) return;
    for (int i = 0; i < suite->n_test_cases; i++)
        free(suite->case_names[i]);
    free(suite->case_names);
    free(suite->exact_solutions);
    free(suite->computed_solutions);
    free(suite->absolute_errors);
    free(suite->relative_errors);
    free(suite->observed_order);
    free(suite->expected_order);
    free(suite->passed);
    free(suite);
}

void uq_verify_add_case(UQVerificationSuite* suite, int idx, const char* name,
                        double exact, double computed) {
    if (idx < 0 || idx >= suite->n_test_cases) return;
    suite->case_names[idx] = strdup(name);
    suite->exact_solutions[idx] = exact;
    suite->computed_solutions[idx] = computed;
    suite->absolute_errors[idx] = fabs(exact - computed);
    suite->relative_errors[idx] = (fabs(exact) > 1e-15)
        ? fabs(exact - computed) / fabs(exact) : fabs(exact - computed);
    suite->passed[idx] = suite->relative_errors[idx] < 1e-6;
    if (suite->passed[idx]) suite->n_passed++;
}

void uq_verify_compute_order(UQVerificationSuite* suite, double* resolutions,
                             int n_resolutions) {
    suite->observed_order = (double*)malloc((n_resolutions - 1) * sizeof(double));
    suite->expected_order = (double*)malloc((n_resolutions - 1) * sizeof(double));
    for (int i = 0; i < n_resolutions - 1; i++) {
        double ratio = resolutions[i] / resolutions[i+1];
        suite->observed_order[i] = log(suite->absolute_errors[i]
            / (suite->absolute_errors[i+1] + 1e-15)) / log(ratio + 1e-15);
        suite->expected_order[i] = 2.0;
    }
}

double uq_verify_gci(double f_fine, double f_coarse, double r, double p_order) {
    /* Grid Convergence Index (Roache, 1994) */
    double eps = fabs((f_fine - f_coarse) / (f_fine + 1e-15));
    double Fs = 1.25; /* Factor of safety */
    return Fs * eps / (pow(r, p_order) - 1.0);
}

void uq_verify_print(UQVerificationSuite* suite) {
    printf("Verification Suite (%d cases, %d passed)\n",
           suite->n_test_cases, suite->n_passed);
    for (int i = 0; i < suite->n_test_cases; i++)
        printf("  %s: exact=%.8f computed=%.8f abs_err=%.2e rel_err=%.2e [%s]\n",
               suite->case_names[i], suite->exact_solutions[i],
               suite->computed_solutions[i], suite->absolute_errors[i],
               suite->relative_errors[i], suite->passed[i] ? "PASS" : "FAIL");
}

/* ============================================================================
 * Calibration-Validation Assessment
 * ============================================================================ */

UQCalValAssessment* uq_calval_create(UQDataset* cal, UQDataset* val) {
    UQCalValAssessment* cv = (UQCalValAssessment*)calloc(1, sizeof(UQCalValAssessment));
    cv->calibration_data = cal;
    cv->validation_data = val;
    return cv;
}

void uq_calval_free(UQCalValAssessment* cv) {
    if (!cv) return;
    uq_matrix_free(cv->extrapolation_leverage);
    free(cv);
}

void uq_calval_assess(UQCalValAssessment* cv,
    double (*model)(double*, double*, void*), void* model_data,
    double* params, int n_params) {
    /* Calibration RMSE */
    double cal_mse = 0.0;
    for (int i = 0; i < cv->calibration_data->n_points; i++) {
        double yh = model(params, &cv->calibration_data->x[i * cv->calibration_data->input_dimension], model_data);
        double d = cv->calibration_data->y[i] - yh;
        cal_mse += d * d;
    }
    cv->calibration_rmse = sqrt(cal_mse / cv->calibration_data->n_points);

    /* Validation RMSE */
    double val_mse = 0.0;
    for (int i = 0; i < cv->validation_data->n_points; i++) {
        double yh = model(params, &cv->validation_data->x[i * cv->validation_data->input_dimension], model_data);
        double d = cv->validation_data->y[i] - yh;
        val_mse += d * d;
    }
    cv->validation_rmse = sqrt(val_mse / cv->validation_data->n_points);
    cv->overfitting_ratio = cv->validation_rmse / (cv->calibration_rmse + 1e-15);
    cv->validated = cv->overfitting_ratio < 2.0;
    (void)n_params;
}

double uq_extrapolation_index(double* x_new, UQMatrix* X_train) {
    /* Mahalanobis distance-based extrapolation measure */
    int n = X_train->rows, p = X_train->cols;
    double* x_center = (double*)calloc(p, sizeof(double));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < p; j++)
            x_center[j] += uq_matrix_get(X_train, i, j);
    for (int j = 0; j < p; j++) x_center[j] /= n;

    UQMatrix* cov = uq_matrix_create(p, p);
    for (int i = 0; i < p; i++)
        for (int j = 0; j < p; j++) {
            double s = 0.0;
            for (int k = 0; k < n; k++)
                s += (uq_matrix_get(X_train, k, i) - x_center[i])
                     * (uq_matrix_get(X_train, k, j) - x_center[j]);
            uq_matrix_set(cov, i, j, s / (n - 1));
        }

    double* diff = (double*)malloc(p * sizeof(double));
    for (int j = 0; j < p; j++) diff[j] = x_new[j] - x_center[j];
    double md = uq_stats_mahalanobis(diff, NULL, cov, p);
    free(x_center); free(diff);
    uq_matrix_free(cov);

    /* Normalize: md / sqrt(p) scaling */
    return 1.0 - exp(-md / sqrt((double)p));
}

/* ============================================================================
 * Decision Analysis under Uncertainty
 * ============================================================================ */

UQDecisionAnalysis* uq_decision_create(int n_alternatives, char** names) {
    UQDecisionAnalysis* da = (UQDecisionAnalysis*)calloc(1, sizeof(UQDecisionAnalysis));
    da->n_alternatives = n_alternatives;
    if (names) {
        da->alternative_names = (char**)malloc(n_alternatives * sizeof(char*));
        for (int i = 0; i < n_alternatives; i++)
            da->alternative_names[i] = names[i] ? strdup(names[i]) : NULL;
    }
    da->expected_utilities = (double*)calloc(n_alternatives, sizeof(double));
    da->utility_variance = (double*)calloc(n_alternatives, sizeof(double));
    da->worst_case_utility = (double*)calloc(n_alternatives, sizeof(double));
    return da;
}

void uq_decision_free(UQDecisionAnalysis* da) {
    if (!da) return;
    if (da->alternative_names) {
        for (int i = 0; i < da->n_alternatives; i++)
            free(da->alternative_names[i]);
        free(da->alternative_names);
    }
    free(da->expected_utilities);
    free(da->utility_variance);
    free(da->worst_case_utility);
    free(da);
}

void uq_decision_compute_expected(UQDecisionAnalysis* da,
    double** utility_matrix, double* state_probs, int n_states) {
    for (int a = 0; a < da->n_alternatives; a++) {
        double eu = 0.0, ev = 0.0;
        for (int s = 0; s < n_states; s++) {
            eu += state_probs[s] * utility_matrix[a][s];
            ev += state_probs[s] * utility_matrix[a][s] * utility_matrix[a][s];
        }
        da->expected_utilities[a] = eu;
        da->utility_variance[a] = ev - eu * eu;
    }
    /* Best alternative */
    da->best_alternative = 0;
    for (int a = 1; a < da->n_alternatives; a++)
        if (da->expected_utilities[a] > da->expected_utilities[da->best_alternative])
            da->best_alternative = a;
}

void uq_decision_minimax(UQDecisionAnalysis* da,
    double** utility_matrix, int n_alternatives, int n_states) {
    double best_worst = INFINITY;
    for (int a = 0; a < n_alternatives; a++) {
        double worst = utility_matrix[a][0];
        for (int s = 1; s < n_states; s++)
            if (utility_matrix[a][s] < worst) worst = utility_matrix[a][s];
        da->worst_case_utility[a] = worst;
        if (worst < best_worst) {
            best_worst = worst;
            da->best_alternative = a;
        }
    }
    da->criterion = UQ_DECISION_MINIMAX;
}

void uq_decision_bayes_risk(UQDecisionAnalysis* da,
    double** utility_matrix, double* posterior_probs,
    int n_alternatives, int n_states) {
    uq_decision_compute_expected(da, utility_matrix, posterior_probs, n_states);
    da->criterion = UQ_DECISION_BAYES_RISK;
}

void uq_decision_hurwicz(UQDecisionAnalysis* da,
    double** utility_matrix, int n_alternatives, int n_states,
    double alpha) {
    da->hurwicz_alpha = alpha;
    for (int a = 0; a < n_alternatives; a++) {
        double worst = utility_matrix[a][0], best = utility_matrix[a][0];
        for (int s = 1; s < n_states; s++) {
            if (utility_matrix[a][s] < worst) worst = utility_matrix[a][s];
            if (utility_matrix[a][s] > best) best = utility_matrix[a][s];
        }
        da->expected_utilities[a] = alpha * best + (1.0 - alpha) * worst;
    }
    da->best_alternative = 0;
    for (int a = 1; a < n_alternatives; a++)
        if (da->expected_utilities[a] > da->expected_utilities[da->best_alternative])
            da->best_alternative = a;
    da->criterion = UQ_DECISION_HURWICZ;
}

double uq_evpi(double** utility_matrix, double* prior_probs,
               int n_alternatives, int n_states) {
    /* Expected Value of Perfect Information */
    double max_eu = -INFINITY;
    for (int a = 0; a < n_alternatives; a++) {
        double eu = 0.0;
        for (int s = 0; s < n_states; s++)
            eu += prior_probs[s] * utility_matrix[a][s];
        if (eu > max_eu) max_eu = eu;
    }

    double ev_ppi = 0.0;
    for (int s = 0; s < n_states; s++) {
        double max_u = -INFINITY;
        for (int a = 0; a < n_alternatives; a++)
            if (utility_matrix[a][s] > max_u) max_u = utility_matrix[a][s];
        ev_ppi += prior_probs[s] * max_u;
    }

    return ev_ppi - max_eu;
}

double uq_evsi(double** utility_matrix, double* prior_probs,
               double** likelihood, int n_alt, int n_states, int n_signals) {
    double max_eu_no_info = -INFINITY;
    for (int a = 0; a < n_alt; a++) {
        double eu = 0.0;
        for (int s = 0; s < n_states; s++)
            eu += prior_probs[s] * utility_matrix[a][s];
        if (eu > max_eu_no_info) max_eu_no_info = eu;
    }

    double ev_with_info = 0.0;
    for (int sig = 0; sig < n_signals; sig++) {
        double p_signal = 0.0;
        for (int s = 0; s < n_states; s++)
            p_signal += prior_probs[s] * likelihood[s][sig];

        double max_eu_sig = -INFINITY;
        for (int a = 0; a < n_alt; a++) {
            double eu = 0.0;
            for (int s = 0; s < n_states; s++)
                eu += prior_probs[s] * likelihood[s][sig] * utility_matrix[a][s];
            eu /= (p_signal + 1e-15);
            if (eu > max_eu_sig) max_eu_sig = eu;
        }
        ev_with_info += p_signal * max_eu_sig;
    }

    return ev_with_info - max_eu_no_info;
}
