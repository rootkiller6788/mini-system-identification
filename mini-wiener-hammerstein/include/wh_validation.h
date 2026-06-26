/**
 * wh_validation.h ? Model Validation for Wiener-Hammerstein Identification
 *
 * Validation methods to assess the quality of an identified WH model:
 *
 *   1. FIT metric: Normalized RMS error (standard in system identification)
 *   2. Residual analysis: Whiteness and independence tests on residuals
 *   3. Cross-validation: Train/test split to detect overfitting
 *   4. Correlation analysis: Higher-order correlation tests for nonlinear
 *      model validation (Billings & Voon, 1986)
 *   5. Frequency-domain validation: Compare BLA of model vs data
 *   6. Pole-zero analysis: Stability and physical realizability checks
 *
 * Key theoretical insight (Ljung, 1999):
 *   A good model should have residuals that are:
 *     - White (uncorrelated in time): R_??(?) = 0 for ? ? 0
 *     - Independent of past inputs: R_?u(?) = 0 for ? ? 0
 *     - For nonlinear models: R_{??'}{u?'}(?) = 0 (higher-order test)
 *
 * References:
 *   - Ljung, L. (1999). System Identification: Theory for the User. 2nd ed.
 *   - Billings, S.A. & Voon, W.S.F. (1986). "Correlation based model validity
 *     tests for non-linear models." Int. J. Control, 44(1), 235-244.
 *   - Soderstrom, T. & Stoica, P. (1989). System Identification.
 *
 * Knowledge Level: L4 (Fundamental Laws), L6 (Canonical Problems)
 */

#ifndef WH_VALIDATION_H
#define WH_VALIDATION_H

#include "wh_model.h"
#include "wh_simulation.h"
#include "wh_identification.h"

/* ??? Validation metrics ????????????????????????????????????????????????? */

/**
 * wh_validate_fit ? Compute FIT metric between model output and reference.
 *
 * FIT = 100 * (1 - ||y_ref - y_model|| / ||y_ref - mean(y_ref)||)
 *
 * The FIT metric measures the fraction of output variation explained by
 * the model. FIT = 100% is perfect; FIT = 0% means the model is no better
 * than predicting the mean; negative FIT means worse than constant prediction.
 *
 * @param model    Identified WH model.
 * @param u        Input data used for validation.
 * @param y_ref    Reference (measured) output.
 * @param n        Number of validation samples.
 * @param config   Simulation configuration (e.g., transient handling).
 * @return         FIT percentage.
 */
double wh_validate_fit(const WH_Model* model,
                        const double* u, const double* y_ref,
                        int n, const WH_SimConfig* config);

/**
 * wh_validate_multi_fit ? Compute FIT on multiple validation datasets.
 *
 * Returns the average FIT across all datasets. Used for k-fold
 * cross-validation.
 *
 * @param model       WH model.
 * @param u_sets      Array of pointers to input datasets (k sets).
 * @param y_sets      Array of pointers to output datasets (k sets).
 * @param n_per_set   Number of samples per dataset (all same size).
 * @param k           Number of datasets.
 * @param config      Simulation config.
 * @param individual_fits Output: FIT for each set (pre-allocated, length k).
 * @return            Average FIT across all k sets.
 */
double wh_validate_multi_fit(const WH_Model* model,
                              const double** u_sets,
                              const double** y_sets,
                              int n_per_set, int k,
                              const WH_SimConfig* config,
                              double* individual_fits);

/* ??? Residual analysis ?????????????????????????????????????????????????? */

/**
 * WH_ResidualAnalysis ? Results of residual analysis.
 */
typedef struct {
    double*  auto_corr;         /* Autocorrelation R_??(?) for ?=0..max_lag */
    double*  cross_corr_eu;     /* Cross-corr R_?u(?) for ?=?max_lag..max_lag */
    double*  cross_corr_e2u2;   /* Higher-order: R_{??'}{u?'}(?)            */
    int      max_lag;           /* Maximum lag analyzed                      */
    double   whiteness_pvalue;  /* Ljung-Box test p-value for whiteness      */
    int      is_white_95;       /* Flag: residuals pass whiteness at 95%     */
    int      is_independent_95; /* Flag: residuals independent of u at 95%   */
    int      passes_nl_test;    /* Flag: passes nonlinear correlation test   */
    double   variance;          /* Residual variance ??_?                    */
    double   mean;              /* Residual mean (should be ?0)              */
} WH_ResidualAnalysis;

/**
 * wh_validate_residuals ? Perform full residual analysis.
 *
 * Steps:
 *   1. Simulate model to get predicted output y_hat.
 *   2. Compute residuals: ?[k] = y[k] - y_hat[k].
 *   3. Compute autocorrelation R_??(?).
 *   4. Compute cross-correlation R_?u(?).
 *   5. Compute nonlinear correlation R_{??'}{u?'}(?).
 *   6. Ljung-Box test for residual whiteness.
 *   7. Apply 95% confidence bounds (?1.96/?N).
 *
 * @param model    WH model.
 * @param u        Input data.
 * @param y_ref    Measured output.
 * @param n        Number of samples.
 * @param max_lag  Maximum lag for correlation analysis.
 * @param config   Simulation config.
 * @param result   Pre-allocated result structure.
 * @return         0 on success.
 */
int wh_validate_residuals(const WH_Model* model,
                           const double* u, const double* y_ref,
                           int n, int max_lag,
                           const WH_SimConfig* config,
                           WH_ResidualAnalysis* result);

/**
 * wh_validate_residuals_free ? Free resources in residual analysis results.
 */
void wh_validate_residuals_free(WH_ResidualAnalysis* result);

/* ??? Cross-validation ??????????????????????????????????????????????????? */

/**
 * WH_CrossValidation ? Results of k-fold cross-validation.
 */
typedef struct {
    int      k_folds;            /* Number of folds                         */
    double   mean_fit;           /* Mean FIT across folds                   */
    double   std_fit;            /* Std deviation of FIT across folds       */
    double   min_fit;            /* Minimum FIT across folds                */
    double   max_fit;            /* Maximum FIT across folds                */
    double*  fold_fits;          /* FIT for each fold (length k_folds)      */
    double   mean_mse;           /* Mean MSE across folds                   */
    int      is_reliable;        /* Flag: model is reliable (std_fit < threshold) */
} WH_CrossValidation;

/**
 * wh_validate_crossval ? Perform k-fold cross-validation.
 *
 * Splits data into k folds. For each fold i:
 *   - Training set: all folds except i
 *   - Validation set: fold i
 *   - Identify model on training set, evaluate on validation set.
 *
 * This provides an unbiased estimate of generalization performance.
 *
 * @param u            Complete input data.
 * @param y            Complete output data.
 * @param n            Total number of samples.
 * @param k_folds      Number of folds (typically 5 or 10).
 * @param id_config    Identification configuration template.
 * @param sim_config   Simulation configuration.
 * @param result       Pre-allocated result structure.
 * @return             0 on success.
 */
int wh_validate_crossval(const double* u, const double* y, int n,
                          int k_folds,
                          const WH_IdentConfig* id_config,
                          const WH_SimConfig* sim_config,
                          WH_CrossValidation* result);

/**
 * wh_validate_crossval_free ? Free cross-validation resources.
 */
void wh_validate_crossval_free(WH_CrossValidation* result);

/* ??? Stability and pole-zero validation ????????????????????????????????? */

/**
 * wh_validate_stability ? Verify BIBO stability of both linear blocks.
 *
 * Checks that all poles of L1 and L2 are strictly inside the unit circle.
 * For FIR blocks, always passes.
 *
 * @param model  WH model.
 * @return       1 if stable, 0 otherwise.
 */
int wh_validate_stability(const WH_Model* model);

/**
 * wh_validate_delay ? Verify that the total delay is physically reasonable.
 *
 * @param model       WH model.
 * @param max_delay   Maximum acceptable delay (samples).
 * @return            1 if delay ? max_delay, 0 otherwise.
 */
int wh_validate_delay(const WH_Model* model, int max_delay);

/**
 * wh_validate_monotonic ? Check if the nonlinearity is monotonic.
 *
 * Non-monotonic nonlinearities can cause identifiability issues.
 *
 * @param model  WH model.
 * @return       1 if monotonic, 0 otherwise.
 */
int wh_validate_monotonic(const WH_Model* model);

/* ??? Frequency-domain validation ???????????????????????????????????????? */

/**
 * wh_validate_frequency ? Compare model and data BLA in frequency domain.
 *
 * Estimates the Best Linear Approximation (BLA) of both the identified
 * model and the validation data via multisine experiments, then compares
 * magnitude and phase at each excited frequency.
 *
 * @param model          WH model.
 * @param u              Multisine input (one period).
 * @param y              Measured output (one period).
 * @param n_period       Number of samples per period.
 * @param max_freq_error Output: maximum magnitude error (dB) across frequencies.
 * @param mean_freq_error Output: mean magnitude error (dB).
 * @return               0 on success.
 */
int wh_validate_frequency(const WH_Model* model,
                           const double* u, const double* y,
                           int n_period,
                           double* max_freq_error,
                           double* mean_freq_error);

/* ??? Comprehensive validation report ???????????????????????????????????? */

/**
 * WH_ValidationReport ? Complete model validation report.
 */
typedef struct {
    double           fit_percent;        /* FIT on validation data          */
    double           mse;                /* MSE on validation data          */
    double           rmse;               /* RMSE on validation data         */
    int              is_stable;          /* BIBO stability check            */
    int              passes_residuals;   /* Residual whiteness & independence */
    double           crossval_mean_fit;  /* Cross-validation mean FIT       */
    double           crossval_std_fit;   /* Cross-validation FIT std        */
    int              is_monotonic;       /* Nonlinearity monotonicity       */
    double           max_freq_error_db;  /* Max frequency-domain error (dB) */
    double           mean_freq_error_db; /* Mean frequency-domain error (dB) */
    int              n_tests_passed;     /* Number of validation tests passed */
    int              n_tests_total;      /* Total number of validation tests */
    int              model_is_valid;     /* Overall validity verdict        */
} WH_ValidationReport;

/**
 * wh_validate_comprehensive ? Run all validation tests and produce report.
 *
 * Runs: FIT, residual analysis, cross-validation, stability check,
 * monotonicity check, and frequency-domain comparison.
 *
 * @param model       WH model to validate.
 * @param u           Validation input data.
 * @param y           Validation output data.
 * @param n           Number of validation samples.
 * @param k_folds     Number of cross-validation folds.
 * @param report      Pre-allocated report structure.
 * @return            0 on success.
 */
int wh_validate_comprehensive(const WH_Model* model,
                               const double* u, const double* y,
                               int n, int k_folds,
                               WH_ValidationReport* report);

/**
 * wh_validate_report_print ? Print validation report to stdout.
 */
void wh_validate_report_print(const WH_ValidationReport* report);

#endif /* WH_VALIDATION_H */
