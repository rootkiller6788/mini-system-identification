#ifndef UQ_CORE_H
#define UQ_CORE_H

#include <stdbool.h>
#include <stddef.h>

/* ============================================================================
 * Uncertainty Quantification (UQ) Core Types and Utilities
 *
 * Key References:
 *   - Smith, R.C. (2013). "Uncertainty Quantification: Theory, Implementation,
 *     and Applications." SIAM.
 *   - Sullivan, T.J. (2015). "Introduction to Uncertainty Quantification."
 *     Springer.
 *   - Kennedy, M.C. & O'Hagan, A. (2001). "Bayesian calibration of computer
 *     models." JRSS B, 63(3), 425-464.
 *   - Sudret, B. (2007). "Global sensitivity analysis using polynomial chaos
 *     expansions." Reliability Engineering & System Safety, 93(7), 964-979.
 * ============================================================================ */

/* --- Uncertainty Classification --- */

typedef enum {
    UQ_ALEATORY = 0,         /* Inherent/irreducible randomness */
    UQ_EPISTEMIC = 1,        /* Knowledge/reducible uncertainty */
    UQ_MEASUREMENT = 2,      /* Sensor/observation noise */
    UQ_MODEL_FORM = 3,       /* Model structure error */
    UQ_PARAMETER = 4,        /* Parameter estimation error */
    UQ_NUMERICAL = 5,        /* Numerical discretization error */
    UQ_INTERPOLATION = 6,    /* Surrogate/interpolation error */
    UQ_DATA_SPARSITY = 7     /* Insufficient data uncertainty */
} UQUncertaintyType;

typedef enum {
    UQ_DIST_NORMAL = 0,
    UQ_DIST_UNIFORM = 1,
    UQ_DIST_STUDENT_T = 2,
    UQ_DIST_CHI2 = 3,
    UQ_DIST_F = 4,
    UQ_DIST_LOG_NORMAL = 5,
    UQ_DIST_BETA = 6,
    UQ_DIST_GAMMA = 7,
    UQ_DIST_EXPONENTIAL = 8,
    UQ_DIST_WEIBULL = 9,
    UQ_DIST_CAUCHY = 10,
    UQ_DIST_MULTIVARIATE_NORMAL = 11,
    UQ_DIST_DIRICHLET = 12,
    UQ_DIST_WISHART = 13,
    UQ_DIST_EMPIRICAL = 14,
    UQ_DIST_GAUSSIAN_PROCESS = 15,
    UQ_DIST_KERNEL_DENSITY = 16
} UQDistributionType;

typedef enum {
    UQ_CONFIDENCE_INTERVAL = 0,
    UQ_PREDICTION_INTERVAL = 1,
    UQ_CREDIBLE_INTERVAL = 2,     /* Bayesian */
    UQ_TOLERANCE_INTERVAL = 3,
    UQ_SIMULTANEOUS_BAND = 4      /* Scheffe / Bonferroni */
} UQIntervalType;

typedef enum {
    UQ_FIRST_ORDER = 0,           /* Linear sensitivity */
    UQ_TOTAL_EFFECT = 1,          /* Sobol total index */
    UQ_SOBOL_MAIN = 2,            /* Sobol main effect */
    UQ_MORRIS = 3,                /* Morris elementary effects */
    UQ_SHAPLEY = 4,               /* Shapley value decomposition */
    UQ_DELTA = 5,                 /* Moment-independent delta */
    UQ_FAST = 6                   /* Fourier amplitude sensitivity */
} UQSensitivityMethod;

/* --- Fundamental Numeric Structures --- */

typedef struct {
    double* data;
    int rows;
    int cols;
    bool is_owner;                /* Owns the memory */
} UQMatrix;

typedef struct {
    double* components;
    int dimension;
} UQVector;

typedef struct {
    double mean;
    double variance;
    double skewness;
    double kurtosis;
    double* quantiles;            /* e.g. [0.025, 0.25, 0.5, 0.75, 0.975] */
    int n_quantiles;
} UQSummaryStats;

/* --- Core Distribution Model --- */

typedef struct {
    UQDistributionType type;
    int n_params;                  /* Number of distribution parameters */
    double* params;               /* Parameter vector (e.g., [μ, σ] for normal) */

    /* Support bounds */
    double lower_bound;
    double upper_bound;
    bool is_bounded;

    /* Derived quantities (cached/computed) */
    double mean;
    double variance;
    double median;
    double mode;

    /* Entropy (if computable analytically) */
    double entropy;

    /* For multivariate: dimension */
    int dimension;

    /* For empirical/kernel: stored samples */
    double* samples;
    int n_samples;
    int sample_capacity;

    /* For Gaussian Process: kernel parameters */
    double gp_length_scale;
    double gp_signal_variance;
    double gp_noise_variance;
    double* gp_training_x;
    double* gp_training_y;
    int gp_n_train;
} UQDistribution;

/* --- Confidence Region --- */

typedef struct {
    UQIntervalType type;
    double confidence_level;       /* e.g., 0.95 */
    double lower_bound;
    double upper_bound;
    int dimension;                /* 1 for scalar, >1 for joint region */
    double* center;               /* Center point (d-dimensional) */
    UQMatrix* covariance;         /* Covariance matrix (d x d) for joint regions */
    double n_samples;             /* Sample size used to compute region */
    double dof;                   /* Degrees of freedom */
    bool is_asymmetric;           /* True if not symmetric about center */
} UQConfidenceRegion;

/* --- Parameter Uncertainty --- */

typedef struct {
    char* name;
    double nominal_value;
    double standard_error;
    double t_statistic;
    double p_value;
    UQConfidenceRegion ci;
    double* correlation_row;       /* Correlation with other parameters */
    int n_correlated;
    bool is_identifiable;          /* Fisher information rank test */
    double relative_error;         /* |se / nominal| */
} UQParameterEstimate;

typedef struct {
    char* model_name;
    UQParameterEstimate* params;
    int n_params;
    double residual_variance;
    double r_squared;
    double adjusted_r_squared;
    double akaike_ic;              /* AIC */
    double bayesian_ic;            /* BIC */
    UQMatrix* parameter_covariance;
    UQMatrix* fisher_information;
    double log_likelihood;
    int n_observations;
    int n_effective_params;        /* For regularized/penalized */
    double condition_number;       /* Of Fisher matrix */
} UQParameterEnsemble;

/* --- Data Structures for UQ --- */

typedef struct {
    double* x;
    double* y;
    double* y_std;                /* Known measurement std (if available) */
    int n_points;
    int input_dimension;
    char** input_names;
} UQDataset;

typedef struct {
    UQMatrix* design_matrix;       /* X (n x p) */
    UQVector* response;            /* y (n x 1) */
    UQVector* coefficients;        /* β (p x 1) */
    UQMatrix* covariance_beta;     /* Cov(β) (p x p) */
    double sigma_squared;          /* σ² estimate */
    double* residuals;
    double* leverages;             /* Hat matrix diagonal */
    double* studentized_residuals;
    int n;
    int p;
    int rank;
    double* cook_distance;
    double press_statistic;        /* PRESS */
    double* variance_inflation_factors;
} UQLinearModel;

/* --- Convergence Diagnostics --- */

typedef struct {
    double geweke_z;              /* Geweke convergence diagnostic */
    double gelman_rubin_rhat;     /* Gelman-Rubin R-hat statistic */
    double raftery_lewis_I;       /* Raftery-Lewis dependence factor */
    int effective_sample_size;
    int burn_in;
    double* autocorrelation;
    int n_lags;
    bool converged;
} UQConvergenceDiagnostic;

/* --- Core API: Distribution Operations --- */

UQDistribution* uq_dist_create_normal(double mean, double std);
UQDistribution* uq_dist_create_uniform(double a, double b);
UQDistribution* uq_dist_create_student_t(double location, double scale, double df);
UQDistribution* uq_dist_create_chi2(double df);
UQDistribution* uq_dist_create_lognormal(double mu, double sigma);
UQDistribution* uq_dist_create_beta(double alpha, double beta);
UQDistribution* uq_dist_create_gamma(double shape, double scale);
UQDistribution* uq_dist_create_wishart(int dim, double df);
UQDistribution* uq_dist_create_dirichlet(int dim, double* alpha);
UQDistribution* uq_dist_create_multivariate_normal(int dim, double* mean, UQMatrix* cov);
UQDistribution* uq_dist_create_empirical(double* samples, int n);
UQDistribution* uq_dist_create_kde(double* samples, int n, double bandwidth);
void uq_dist_free(UQDistribution* dist);
double uq_dist_pdf(UQDistribution* dist, double x);
double uq_dist_cdf(UQDistribution* dist, double x);
double uq_dist_quantile(UQDistribution* dist, double p);
double uq_dist_log_pdf(UQDistribution* dist, double x);
double uq_dist_sample(UQDistribution* dist);
void uq_dist_sample_n(UQDistribution* dist, int n, double* out);
double uq_dist_entropy_analytical(UQDistribution* dist);
double uq_dist_kl_divergence(UQDistribution* p, UQDistribution* q, int n_mc);
UQSummaryStats uq_dist_summary_stats(UQDistribution* dist);

/* --- Core API: Confidence Regions --- */

UQConfidenceRegion* uq_ci_create(double confidence, int dimension);
void uq_ci_free(UQConfidenceRegion* ci);
void uq_ci_from_normal(UQConfidenceRegion* ci, double mean, double std, int n);
void uq_ci_from_student(UQConfidenceRegion* ci, double mean, double se, double df);
void uq_ci_from_bootstrap(double* samples, int n, double confidence,
                          UQConfidenceRegion* ci);
void uq_ci_from_quantiles(double* data, int n, double confidence,
                          UQConfidenceRegion* ci);
bool uq_ci_contains(UQConfidenceRegion* ci, double* point);
double uq_ci_half_width(UQConfidenceRegion* ci);
void uq_ci_print(UQConfidenceRegion* ci);

/* --- Core API: Parameter Estimation --- */

UQParameterEnsemble* uq_ensemble_create(const char* model_name, int n_params);
void uq_ensemble_free(UQParameterEnsemble* ens);
void uq_ensemble_set_param(UQParameterEnsemble* ens, int idx, const char* name,
                           double estimate, double std_err);
void uq_ensemble_compute_statistics(UQParameterEnsemble* ens);
void uq_ensemble_compute_correlations(UQParameterEnsemble* ens);
void uq_ensemble_print(UQParameterEnsemble* ens);
double uq_ensemble_mahalanobis_distance(UQParameterEnsemble* ens, double* point);

/* --- Core API: Linear Model UQ --- */

UQLinearModel* uq_lm_create(UQMatrix* X, UQVector* y);
void uq_lm_free(UQLinearModel* lm);
void uq_lm_fit(UQLinearModel* lm);
void uq_lm_fit_weighted(UQLinearModel* lm, double* weights);
void uq_lm_predict(UQLinearModel* lm, double* x_new, double* y_hat,
                   double* se_fit, double* se_pred);
void uq_lm_anova(UQLinearModel* lm, double* ss_reg, double* ss_res,
                 double* ss_tot, double* f_stat, double* p_val);
void uq_lm_influence_diagnostics(UQLinearModel* lm);
void uq_lm_vif(UQLinearModel* lm);
double uq_lm_press(UQLinearModel* lm);

/* --- Core API: Convergence Diagnostics --- */

UQConvergenceDiagnostic* uq_conv_create(int n_lags);
void uq_conv_free(UQConvergenceDiagnostic* diag);
void uq_conv_geweke(UQConvergenceDiagnostic* diag, double* chain, int n,
                    double frac_first, double frac_last);
void uq_conv_gelman_rubin(UQConvergenceDiagnostic* diag, double** chains,
                          int n_chains, int n_per_chain);
void uq_conv_autocorrelation(UQConvergenceDiagnostic* diag, double* chain, int n);
int uq_conv_effective_sample_size(UQConvergenceDiagnostic* diag, double* chain,
                                   int n);

/* --- Core API: Matrix/Vector Utilities --- */

UQMatrix* uq_matrix_create(int rows, int cols);
void uq_matrix_free(UQMatrix* mat);
UQMatrix* uq_matrix_copy(UQMatrix* src);
void uq_matrix_set(UQMatrix* mat, int i, int j, double val);
double uq_matrix_get(UQMatrix* mat, int i, int j);
void uq_matrix_multiply(UQMatrix* C, UQMatrix* A, UQMatrix* B);
void uq_matrix_transpose(UQMatrix* At, UQMatrix* A);
void uq_matrix_invert(UQMatrix* A_inv, UQMatrix* A);
double uq_matrix_determinant(UQMatrix* A);
double uq_matrix_trace(UQMatrix* A);
void uq_matrix_cholesky(UQMatrix* L, UQMatrix* A);
void uq_matrix_eigen_sym(UQMatrix* A, double* eigenvalues, UQMatrix* eigenvectors);
void uq_matrix_svd(UQMatrix* U, double* S, UQMatrix* Vt, UQMatrix* A);
double uq_matrix_condition_number(UQMatrix* A);
int uq_matrix_rank(UQMatrix* A, double tol);

UQVector* uq_vector_create(int dim);
void uq_vector_free(UQVector* v);
double uq_vector_norm(UQVector* v);
double uq_vector_dot(UQVector* a, UQVector* b);
void uq_vector_scale(UQVector* v, double s);

/* --- Core API: Statistical Functions --- */

double uq_stats_lgamma(double x);
double uq_stats_digamma(double x);
double uq_stats_erfinv(double x);
double uq_stats_normal_quantile(double p);
double uq_stats_student_t_quantile(double p, double df);
double uq_stats_chi2_quantile(double p, double df);
double uq_stats_f_quantile(double p, double df1, double df2);
double uq_stats_beta_regularized(double a, double b, double x);
double uq_stats_correlation(double* x, double* y, int n);
void uq_stats_covariance_matrix(UQMatrix* cov, UQMatrix* data);
double uq_stats_mahalanobis(double* x, double* mu, UQMatrix* sigma, int dim);

/* --- Core API: Data Operations --- */

UQDataset* uq_dataset_create(int n, int input_dim);
void uq_dataset_free(UQDataset* ds);
void uq_dataset_split(UQDataset* ds, double train_frac,
                      UQDataset** train, UQDataset** test);
void uq_dataset_standardize(UQDataset* ds);
void uq_dataset_summary(UQDataset* ds, UQSummaryStats** x_stats,
                         UQSummaryStats* y_stats);

#endif /* UQ_CORE_H */
