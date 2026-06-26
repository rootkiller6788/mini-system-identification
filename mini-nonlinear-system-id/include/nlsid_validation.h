#ifndef NLSID_VALIDATION_H
#define NLSID_VALIDATION_H

#include "nlsid_core.h"
#include "nlsid_models.h"

/* ============================================================================
 * mini-nonlinear-system-id: Model Validation and Selection
 *
 * Statistical tests and criteria for validating identified nonlinear
 * models and selecting between candidate model structures.
 *
 * The validation framework addresses three questions:
 *   1. Is the model adequate? (residual whiteness, independence)
 *   2. Which model is best? (AIC, BIC, cross-validation, F-test)
 *   3. Is the system actually nonlinear? (nonlinearity detection)
 *
 * Reference:
 *   Ljung (1999) Chapter 16: Model Validation
 *   Billings & Zhu (1995) "Model validation tests..."
 *   Burnham & Anderson (2002) "Model Selection and Multimodel Inference"
 * ============================================================================ */

/* ============================================================================
 * Part 1: Residual Analysis
 * ============================================================================ */

/** Compute residual statistics: mean, variance, skewness, kurtosis.
 *  For a good model, residuals should be zero-mean white noise. */
void nlsid_residual_statistics(const double* residuals, int n,
                                double* mean, double* variance,
                                double* skewness, double* kurtosis);

/** Auto-correlation test for residual whiteness.
 *  Computes r_ee(τ) for τ = 0..max_lag.
 *  For white residuals, r_ee(τ) ≈ 0 for τ > 0.
 *  Reference: Ljung (1999) §16.5 */
void nlsid_autocorrelation_test(const double* e, int n, int max_lag,
                                 double* r_ee, double* confidence_band,
                                 bool* is_white, double* q_statistic);

/** Cross-correlation test: r_eu(τ) between residuals e(t) and input u(t).
 *  A good model should have r_eu(τ) ≈ 0 for all τ, indicating no
 *  unmodeled linear dynamics remain.
 *  Reference: Billings & Voon (1986) */
void nlsid_crosscorrelation_test(const double* e, const double* u,
                                  int n, int max_lag,
                                  double* r_eu, double* confidence_band,
                                  bool* is_independent);

/** Ljung-Box Q test for residual whiteness.
 *  Q = n*(n+2) Σ_{k=1}^{h} r_k^2 / (n-k)
 *  Under H0 (white noise), Q ~ χ²(h). Returns approximate p-value.
 *  Reference: Ljung & Box (1978) */
double nlsid_ljung_box_test(const double* residuals, int n, int max_lag,
                             double* q_stat);

/** Compute normalized ACF and PACF of residuals to check model adequacy.
 *  PACF computed via Durbin-Levinson recursion. */
void nlsid_residual_acf_pacf(const double* e, int n, int max_lag,
                              double* acf, double* pacf);

/* ============================================================================
 * Part 2: Information Criteria for Model Selection  [L4 Theorem]
 * ============================================================================ */

/** Akaike Information Criterion: AIC = N*ln(V_N) + 2*d
 *  where d = number of parameters, V_N = MSE.
 *  Smaller AIC indicates better model (trading fit vs complexity).
 *  Theorem (Akaike, 1974): AIC is an asymptotically unbiased estimator
 *  of the expected Kullback-Leibler divergence. */
double nlsid_aic(double prediction_error_variance, int n_data, int n_params);

/** Corrected AIC for small sample sizes:
 *  AICc = AIC + 2d(d+1)/(N-d-1)
 *  Recommended when N/d < 40.
 *  Reference: Hurvich & Tsai (1989) */
double nlsid_aicc(double prediction_error_variance, int n_data, int n_params);

/** Bayesian Information Criterion: BIC = N*ln(V_N) + d*ln(N)
 *  BIC imposes a heavier penalty on model complexity than AIC.
 *  Theorem: Under regularity conditions, BIC is consistent (selects the
 *  true model with probability → 1 as N → ∞).
 *  Reference: Schwarz (1978) */
double nlsid_bic(double prediction_error_variance, int n_data, int n_params);

/** Minimum Description Length: MDL = -log P(data|model) + (d/2)log(N)
 *  Based on Rissanen's principle of shortest description.
 *  Reference: Rissanen (1978) */
double nlsid_mdl(double prediction_error_variance, int n_data, int n_params);

/** Final Prediction Error: FPE = V_N * (1 + d/N) / (1 - d/N)
 *  Estimates the expected prediction error on fresh data.
 *  Reference: Akaike (1970) */
double nlsid_fpe(double prediction_error_variance, int n_data, int n_params);

/** Hannan-Quinn criterion: HQ = N*ln(V_N) + 2*d*ln(ln(N))
 *  Intermediate penalty between AIC and BIC.
 *  Reference: Hannan & Quinn (1979) */
double nlsid_hannan_quinn(double prediction_error_variance, int n_data,
                           int n_params);

/* ============================================================================
 * Part 3: Model Structure Selection
 * ============================================================================ */

/** F-test for comparing nested nonlinear models.
 *  F = ((V_reduced - V_full) / (d_full - d_reduced)) / (V_full / (N - d_full))
 *  Under H0 (reduced model is adequate), F ~ F(df1, df2).
 *  Returns the computed F-statistic. */
double nlsid_f_test(double v_reduced, double v_full,
                     int d_reduced, int d_full, int n_data,
                     double* p_value);

/** Select optimal NARX model orders (ny, nu, nk) by searching over
 *  a grid and returning the combination with minimum criterion.
 *  Returns 0 on success. */
int nlsid_select_narx_order(const NLSIDDataset* ds,
                             int ny_max, int nu_max, int nk_max,
                             BasisExpansion* template_expansion,
                             NLSIDConfig* config,
                             int criterion, /* 0=AIC, 1=BIC, 2=AICc */
                             int* best_ny, int* best_nu, int* best_nk,
                             double* best_criterion_value);

/** Structural risk minimization: select model complexity to minimize
 *  the generalization error bound.
 *  R(θ) ≤ R_emp(θ) + sqrt( h*(ln(2N/h) + 1) - ln(η/4) / N )
 *  where h = VC dimension (approximated by n_params).
 *  Reference: Vapnik (1998) */
double nlsid_structural_risk(double empirical_risk, int n_params,
                              int n_data, double confidence);

/* ============================================================================
 * Part 4: Nonlinear Model Adequacy Tests
 * ============================================================================ */

/** Model validity tests for NARMAX models (Billings & Zhu, 1995):
 *  1. φ_ee(τ) ≈ δ(τ)     (residual whiteness)
 *  2. φ_eu(τ) ≈ 0 ∀τ     (residual-input independence)
 *  3. φ_e(ue)(τ) ≈ 0 ∀τ ≥ 0  (nonlinear independence)
 *  4. φ_u2e2(τ) ≈ 0 ∀τ   (nonlinear cross-correlation)
 *  5. φ_e(eu)(τ) ≈ 0 ∀τ  (nonlinear dynamic independence)
 *
 *  Returns true if ALL tests pass at 95% confidence level. */
bool nlsid_narmax_validation_tests(const double* e, const double* u,
                                    int n, int max_lag, bool* test_results);

/** Compute φ_ee(τ): auto-correlation of residuals (Test 1) */
void nlsid_test_phi_ee(const double* e, int n, int max_lag,
                        double* phi, double* conf, bool* passed);

/** Compute φ_eu(τ): cross-correlation residuals-input (Test 2) */
void nlsid_test_phi_eu(const double* e, const double* u, int n, int max_lag,
                        double* phi, double* conf, bool* passed);

/** Compute φ_e(ue)(τ): nonlinear cross-correlation (Test 3)
 *  where (ue)(t) = u(t) * e(t) */
void nlsid_test_phi_e_ue(const double* e, const double* u, int n, int max_lag,
                          double* phi, double* conf, bool* passed);

/** Compute φ_(u2'e2)(τ): nonlinear cross-correlation (Test 4)
 *  where u2'(t) = u^2(t) - mean(u^2) and e2'(t) = e^2(t) - mean(e^2) */
void nlsid_test_phi_u2_e2(const double* e, const double* u, int n, int max_lag,
                           double* phi, double* conf, bool* passed);

/** Compute φ_e(eu)(τ): dynamic nonlinear test (Test 5) */
void nlsid_test_phi_e_eu(const double* e, const double* u, int n, int max_lag,
                          double* phi, double* conf, bool* passed);

/* ============================================================================
 * Part 5: Simulation-Based Validation
 * ============================================================================ */

/** Compare k-step-ahead predictions with measured output.
 *  For k=1: one-step-ahead (best case)
 *  For k→∞: pure simulation (worst case, no output correction)
 *  Computes NRMSE fit for each k. */
void nlsid_k_step_ahead_validation(const NLSIDModel* model,
                                    const NLSIDDataset* ds,
                                    int max_k, double* fit_vs_k);

/** Stability analysis of the identified model:
 *  Run simulation from multiple initial conditions and check if
 *  outputs remain bounded.
 *  Returns the percentage of stable trajectories. */
double nlsid_simulation_stability_test(const NLSIDModel* model,
                                        const double* u_test, int n_test,
                                        int n_random_ics, unsigned int* seed);

/** Compute the output error (simulation error) as opposed to
 *  one-step-ahead prediction error. */
double nlsid_output_error(const NLSIDModel* model,
                           const NLSIDDataset* ds, double* y_sim);

/* ============================================================================
 * Part 6: Linear vs Nonlinear Comparison
 * ============================================================================ */

/** Fit a linear ARX model for comparison with nonlinear models.
 *  Uses ordinary least squares.
 *  Returns the linear ARX parameters and fit percentage. */
int nlsid_fit_linear_arx(const NLSIDDataset* ds, int na, int nb, int nk,
                          double* a_params, double* b_params,
                          double* fit_percent, double* mse);

/** Statistical test: is the nonlinear model significantly better?
 *  Uses the F-test comparing linear ARX vs nonlinear NARX.
 *  H0: linear model is adequate
 *  H1: nonlinear model is needed
 *  Returns true if nonlinear model is significantly better (p < 0.05). */
bool nlsid_test_nonlinear_significance(const NLSIDDataset* ds,
                                        int na, int nb, int nk,
                                        const NLSIDModel* nonlinear_model,
                                        double* p_value);

/** Compute the nonlinearity contribution ratio:
 *  η = (MSE_linear - MSE_nonlinear) / MSE_linear
 *  η ≈ 0 → system is linear; η ≈ 1 → system is strongly nonlinear. */
double nlsid_nonlinearity_contribution_ratio(const NLSIDDataset* ds,
                                              int na, int nb, int nk,
                                              const NLSIDModel* nln_model);

/* ============================================================================
 * Part 7: Confidence Intervals
 * ============================================================================ */

/** Compute parameter standard errors from the inverse Hessian.
 *  σ_i = sqrt( σ_e^2 * [H^{-1}]_{ii} )
 *  where σ_e^2 = V_N is the estimated noise variance.
 *  Theorem (asymptotic normality): √N (θ̂ - θ₀) → N(0, P) where
 *  P = σ_e^2 * H^{-1} (under regularity conditions).
 *  Reference: Ljung (1999) §9.4 */
void nlsid_parameter_standard_errors(const NLSIDModel* model,
                                      const NLSIDDataset* ds,
                                      double residual_variance,
                                      double* std_errors);

/** 95% confidence interval for each parameter:
 *  θ_i ∈ [θ̂_i - 1.96 σ_i, θ̂_i + 1.96 σ_i] */
void nlsid_parameter_confidence_intervals(const double* theta,
                                           const double* std_errors,
                                           int n_params,
                                           double* ci_lower, double* ci_upper);

/** t-test for parameter significance. H0: θ_i = 0.
 *  t = θ̂_i / σ_i. Returns array of p-values. */
void nlsid_parameter_t_test(const double* theta, const double* std_errors,
                             int n_params, int n_data, double* p_values);

/* ============================================================================
 * Part 8: Comprehensive Validation Report
 * ============================================================================ */

/** Complete model validation report structure */
typedef struct {
    /* Residual tests */
    double residual_mean;
    double residual_variance;
    double residual_whiteness_q;
    double residual_whiteness_pvalue;
    bool residual_is_white;
    double crosscorr_max;
    bool crosscorr_independent;

    /* Information criteria */
    double aic;
    double bic;
    double aicc;
    double mdl;
    double fpe;
    double hq;

    /* Fit statistics */
    double fit_estimation;
    double fit_validation;
    double output_error_fit;

    /* NARMAX validation */
    bool test1_residual_white;
    bool test2_eu_independent;
    bool test3_e_ue;
    bool test4_u2_e2;
    bool test5_e_eu;
    bool all_narmax_passed;

    /* Stability */
    double simulation_stability_percent;

    /* Nonlinear vs linear */
    double nonlinear_contribution_ratio;
    bool nonlinear_significant;

    /* Overall verdict */
    bool model_accepted;
    double overall_score;  /* 0=bad, 1=perfect */
} NLSIDValidationReport;

/** Run comprehensive validation and generate report */
NLSIDValidationReport* nlsid_validate(const NLSIDModel* model,
                                       const NLSIDDataset* estimation_data,
                                       const NLSIDDataset* validation_data,
                                       int max_lag);
void nlsid_validation_report_free(NLSIDValidationReport* report);
void nlsid_validation_report_print(const NLSIDValidationReport* report);

#endif /* NLSID_VALIDATION_H */