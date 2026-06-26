#ifndef UQ_VALIDATION_H
#define UQ_VALIDATION_H

#include "uq_core.h"
#include <stdbool.h>

/* ============================================================================
 * Model Validation and Predictive Assessment under Uncertainty
 *
 * Key References:
 *   - Oberkampf, W.L. & Roy, C.J. (2010). "Verification and Validation in
 *     Scientific Computing." Cambridge University Press.
 *   - ASME V&V 20-2009. "Standard for Verification and Validation in
 *     Computational Fluid Dynamics and Heat Transfer."
 *   - Roache, P.J. (1998). "Verification and Validation in Computational
 *     Science and Engineering." Hermosa Publishers.
 * ============================================================================ */

/* --- Validation Metrics --- */

typedef enum {
    UQ_VAL_RESIDUAL = 0,
    UQ_VAL_RELATIVE = 1,
    UQ_VAL_RMSE = 2,
    UQ_VAL_NRMSE = 3,             /* Normalized RMSE */
    UQ_VAL_MAE = 4,
    UQ_VAL_MAPE = 5,              /* Mean Absolute Percentage Error */
    UQ_VAL_R_SQUARED = 6,
    UQ_VAL_Q_SQUARED = 7,         /* Predictive R² (leave-one-out) */
    UQ_VAL_CONCORDANCE = 8,       /* Lin's concordance correlation */
    UQ_VAL_THEIL_U = 9,           /* Theil's U statistic */
    UQ_VAL_MASE = 10,             /* Mean Absolute Scaled Error */
    UQ_VAL_WILKS = 11,            /* Wilks' theorem likelihood ratio */
    UQ_VAL_BF = 12                /* Bayes Factor vs null */
} UQValidationMetric;

/* --- Validation Result --- */

typedef struct {
    UQValidationMetric metric;
    double value;
    double p_value;               /* For hypothesis tests */
    double confidence_lower;
    double confidence_upper;
    bool is_significant;          /* p < alpha */
    double threshold;             /* Acceptance threshold */
    bool passed;
} UQValidationResult;

/* --- Predictive Assessment --- */

typedef struct {
    UQLinearModel* model;
    double* predictions;
    double* prediction_variance;
    double* residuals;
    int n_predictions;

    /* Prediction intervals */
    double* pi_lower;             /* [n] */
    double* pi_upper;             /* [n] */
    double pi_coverage;           /* Fraction of observations within PI */
    double pi_average_width;

    /* Cross-validation */
    double* cv_predictions;       /* Leave-one-out predictions */
    double cv_rmse;
    double cv_mae;
    double cv_q_squared;

    /* Residual diagnostics */
    double dw_statistic;          /* Durbin-Watson */
    double dw_p_value;
    double shapiro_wilk_stat;     /* Normality test */
    double shapiro_wilk_p;
    bool residuals_normal;
    bool residuals_independent;
    bool homoskedastic;           /* Breusch-Pagan test */
    double bp_statistic;
    double bp_p_value;

    /* Outlier detection */
    int n_outliers;               /* Studentized residual > threshold */
    int* outlier_indices;
} UQPredictiveAssessment;

/* --- Verification (Code/Algorithm correctness) --- */

typedef struct {
    int n_test_cases;
    char** case_names;
    double* exact_solutions;
    double* computed_solutions;
    double* absolute_errors;
    double* relative_errors;
    double* observed_order;        /* Convergence order */
    double* expected_order;
    double grid_convergence_index; /* GCI (Roache, 1994) */
    bool* passed;
    int n_passed;
} UQVerificationSuite;

/* --- Calibration-Validation split assessment --- */

typedef struct {
    UQDataset* calibration_data;
    UQDataset* validation_data;
    double calibration_rmse;
    double validation_rmse;
    double overfitting_ratio;      /* val_rmse / cal_rmse */
    double aic;
    double bic;

    /* Reality check: model vs. reality gap */
    double model_reality_gap;
    double model_reality_gap_ci;

    /* Extrapolation assessment */
    double extrapolation_index;    /* 0 (interpolation) → 1 (far extrapolation) */
    UQMatrix* extrapolation_leverage;

    bool validated;
} UQCalValAssessment;

/* --- Uncertainty-Aware Decision --- */

typedef enum {
    UQ_DECISION_MINIMAX = 0,       /* Minimize maximum regret */
    UQ_DECISION_BAYES_RISK = 1,    /* Minimize posterior expected loss */
    UQ_DECISION_OPTIMISTIC = 2,    /* Best-case decision */
    UQ_DECISION_PESSIMISTIC = 3,   /* Worst-case robust */
    UQ_DECISION_HURWICZ = 4,       /* Weighted optimism/pessimism */
    UQ_DECISION_INFO_GAP = 5       /* Info-gap decision (Ben-Haim, 2006) */
} UQDecisionCriterion;

typedef struct {
    UQDecisionCriterion criterion;
    int n_alternatives;
    char** alternative_names;
    double* expected_utilities;
    double* utility_variance;
    double* worst_case_utility;
    int best_alternative;
    double value_of_information;   /* EVPI */
    double hurwicz_alpha;          /* 0=pessimistic, 1=optimistic */
} UQDecisionAnalysis;

/* --- API: Validation Metrics --- */

UQValidationResult uq_validate_compute(double* observed, double* predicted,
    int n, UQValidationMetric metric);
UQValidationResult uq_validate_residual(double* observed, double* predicted,
    int n);
UQValidationResult uq_validate_rmse(double* observed, double* predicted,
    int n);
UQValidationResult uq_validate_mae(double* observed, double* predicted, int n);
UQValidationResult uq_validate_mape(double* observed, double* predicted, int n);
UQValidationResult uq_validate_r_squared(double* observed, double* predicted,
    int n, int n_params);
UQValidationResult uq_validate_concordance(double* observed, double* predicted,
    int n);
UQValidationResult uq_validate_theil_u(double* observed, double* predicted,
    int n);
UQValidationResult uq_validate_mase(double* observed, double* predicted,
    int n, int seasonal_period);

/* --- API: Predictive Assessment --- */

UQPredictiveAssessment* uq_predictive_create(UQLinearModel* lm,
    double* X_pred, int n_pred, int p);
void uq_predictive_free(UQPredictiveAssessment* pa);
void uq_predictive_compute(UQPredictiveAssessment* pa, double* y_obs);
void uq_prediction_intervals(UQPredictiveAssessment* pa, double confidence);
void uq_residual_diagnostics(UQPredictiveAssessment* pa, double* y_obs, int n);
void uq_cross_validate(UQPredictiveAssessment* pa, double* X, double* y,
                       int n, int p);
void uq_durbin_watson(UQPredictiveAssessment* pa);
void uq_shapiro_wilk_test(double* residuals, int n, double* W, double* p_value);
void uq_breusch_pagan_test(UQPredictiveAssessment* pa, double* X, int n, int p);
void uq_outlier_detect(UQPredictiveAssessment* pa, double threshold);

/* --- API: Verification --- */

UQVerificationSuite* uq_verify_create(int n_cases);
void uq_verify_free(UQVerificationSuite* suite);
void uq_verify_add_case(UQVerificationSuite* suite, int idx, const char* name,
                        double exact, double computed);
void uq_verify_compute_order(UQVerificationSuite* suite, double* resolutions,
                             int n_resolutions);
double uq_verify_gci(double f_fine, double f_coarse, double r, double p_order);
void uq_verify_print(UQVerificationSuite* suite);

/* --- API: Calibration-Validation --- */

UQCalValAssessment* uq_calval_create(UQDataset* cal, UQDataset* val);
void uq_calval_free(UQCalValAssessment* cv);
void uq_calval_assess(UQCalValAssessment* cv,
    double (*model)(double*, double*, void*), void* model_data,
    double* params, int n_params);
double uq_extrapolation_index(double* x_new, UQMatrix* X_train);

/* --- API: Uncertainty-Aware Decision --- */

UQDecisionAnalysis* uq_decision_create(int n_alternatives, char** names);
void uq_decision_free(UQDecisionAnalysis* da);
void uq_decision_compute_expected(UQDecisionAnalysis* da,
    double** utility_matrix,   /* [alt][state] */
    double* state_probs, int n_states);
void uq_decision_minimax(UQDecisionAnalysis* da,
    double** utility_matrix, int n_alternatives, int n_states);
void uq_decision_bayes_risk(UQDecisionAnalysis* da,
    double** utility_matrix, double* posterior_probs,
    int n_alternatives, int n_states);
void uq_decision_hurwicz(UQDecisionAnalysis* da,
    double** utility_matrix, int n_alternatives, int n_states,
    double alpha);
double uq_evpi(double** utility_matrix, double* prior_probs,
               int n_alternatives, int n_states);
double uq_evsi(double** utility_matrix, double* prior_probs,
               double** likelihood, int n_alt, int n_states, int n_signals);

#endif /* UQ_VALIDATION_H */
