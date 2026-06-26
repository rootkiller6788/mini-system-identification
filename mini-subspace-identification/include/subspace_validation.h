#ifndef SUBSPACE_VALIDATION_H
#define SUBSPACE_VALIDATION_H
#include "subspace_core.h"

/* ============================================================================
 * Subspace Identification -- Model Validation
 *
 * Comprehensive validation framework for identified state-space models.
 * Validates:
 *   1. Prediction accuracy (simulation and k-step prediction)
 *   2. Residual analysis (whiteness, independence from input)
 *   3. Stability (eigenvalues of A inside unit circle)
 *   4. Model uncertainty (parameter covariance, confidence bounds)
 *   5. Frequency-domain validation (Bode plot comparison)
 *
 * The chi-squared test for residual whiteness:
 *   Q = N * (N + 2) * sum_{k=1}^h r_k^2 / (N - k)  ~  chi^2(h)
 * where r_k is the k-th lag autocorrelation of residuals.
 * Reject whiteness if Q > chi^2_{alpha}(h).
 *
 * Cross-correlation test:
 *   r_{ue}(tau) = E[u(t) * e(t+tau)]
 * Should be zero for all tau if model captures input-output dynamics.
 *
 * References:
 *   Ljung, L. (1999) -- System Identification: Theory for the User, Ch. 16
 *   Van Overschee & De Moor (1996) -- Subspace Identification, Ch. 5
 *   Box, Jenkins, Reinsel & Ljung (2015) -- Time Series Analysis, Ch. 8
 * ============================================================================ */

/* --- Prediction Accuracy Metrics --- */

/* Normalized Root Mean Square Error (NRMSE) fit percentage.
 * Fit = 100 * (1 - ||y - y_hat|| / ||y - mean(y)||) */
double subspace_validation_nrmse(const double *y, const double *y_hat, int N);

/* Variance Accounted For (VAF).
 * VAF = 1 - var(y - y_hat) / var(y) */
double subspace_validation_vaf(const double *y, const double *y_hat, int N);

/* Mean Absolute Percentage Error (MAPE) */
double subspace_validation_mape(const double *y, const double *y_hat, int N);

/* Maximum Absolute Error */
double subspace_validation_maxae(const double *y, const double *y_hat, int N);

/* --- Residual Analysis --- */

/* Autocorrelation of residuals at lags 0..max_lag.
 * r_k = sum_{t=1}^{N-k} e(t) * e(t+k) / sum_{t=1}^N e(t)^2 */
void subspace_residual_autocorrelation(const double *e, int N,
                                        int max_lag, double *acf);

/* Ljung-Box Q test for residual whiteness.
 * Returns the p-value under chi^2(max_lag) null hypothesis.
 * p < 0.05 suggests residuals are not white. */
double subspace_ljung_box_test(const double *e, int N, int max_lag,
                                double *Q_stat);

/* Cross-correlation between input u and residual e at lags -max_lag..+max_lag.
 * Tests for feedback effects and missing input dynamics. */
void subspace_cross_correlation_ue(const double *u, const double *e,
                                     int N, int max_lag, double *ccf);

/* Cross-correlation significance test.
 * Returns true if cross-correlation is within 95% confidence bounds
 * (+- 1.96/sqrt(N)) at all lags. */
bool subspace_cross_correlation_test(const double *ccf, int N, int max_lag);

/* --- Stability Validation --- */

/* Check if all eigenvalues of A are inside the unit circle.
 * Returns the spectral radius max(|lambda_i|). */
double subspace_stability_check(const SubspaceModel *model);

/* Compute poles and zeros of the identified system (SISO).
 * For MIMO, computes transmission zeros. */
int subspace_model_poles_zeros(const SubspaceModel *model,
                                double *poles_real, double *poles_imag,
                                double *zeros_real, double *zeros_imag,
                                int max_pz);

/* --- Model Uncertainty --- */

/* Compute parameter covariance matrix via the delta method.
 * Based on the asymptotic distribution of subspace estimates. */
int subspace_parameter_covariance(const SubspaceData *data,
                                   const SubspaceModel *model,
                                   double *cov_matrix);

/* Compute 95% confidence intervals for frequency response.
 * Uses linearization of the mapping from parameters to frequency response. */
int subspace_bode_confidence(const SubspaceModel *model,
                              const double *parameter_cov,
                              int n_freq, const double *omega,
                              double *mag_lower, double *mag_upper,
                              double *phase_lower, double *phase_upper);

/* --- Comparative Validation --- */

/* Compare two models on the same validation data.
 * Returns: -1 if model1 is better (higher NRMSE), +1 if model2 is better, 0 if tie */
int subspace_compare_models(const SubspaceModel *model1,
                             const SubspaceModel *model2,
                             const SubspaceData *validation_data);

/* --- Validation Report --- */

/* Print a comprehensive validation report to stdout */
void subspace_validation_report(const SubspaceModel *model,
                                 const SubspaceData *validation_data);

/* Generate validation metrics summary structure */
typedef struct {
    double  nrmse_fit;        /* NRMSE fit percentage */
    double  vaf;              /* Variance accounted for */
    double  mape;             /* Mean absolute percentage error */
    double  max_absolute_error;
    double  residual_whiteness_pvalue;
    double  cross_correlation_max;
    double  spectral_radius;
    bool    is_stable;
    double  condition_number;
    double  aic;
    double  bic;
    int     n_parameters;     /* Number of free parameters in model */
    char    recommendation[256];
} SubspaceValidationReport;

SubspaceValidationReport subspace_validate_model(const SubspaceModel *model,
                                                   const SubspaceData *data);
void subspace_validation_report_free(SubspaceValidationReport *report);

#endif /* SUBSPACE_VALIDATION_H */
