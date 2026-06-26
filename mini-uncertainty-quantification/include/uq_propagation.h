#ifndef UQ_PROPAGATION_H
#define UQ_PROPAGATION_H

#include "uq_core.h"
#include "uq_sampling.h"
#include <stdbool.h>

/* ============================================================================
 * Uncertainty Propagation Methods
 *
 * Forward propagation: given input distributions, characterize output distribution.
 * Inverse propagation: given output constraints, characterize admissible inputs.
 *
 * Key References:
 *   - Ghanem, R.G. & Spanos, P.D. (1991). "Stochastic Finite Elements:
 *     A Spectral Approach." Springer.
 *   - Xiu, D. (2010). "Numerical Methods for Stochastic Computations:
 *     A Spectral Method Approach." Princeton University Press.
 *   - Julier, S.J. & Uhlmann, J.K. (2004). "Unscented filtering and
 *     nonlinear estimation." Proc. IEEE, 92(3), 401-422.
 *   - Le Maître, O.P. & Knio, O.M. (2010). "Spectral Methods for
 *     Uncertainty Quantification." Springer.
 * ============================================================================ */

/* --- Propagation Methods --- */

typedef enum {
    UQ_PROP_LINEARIZATION = 0,       /* First-order Taylor expansion */
    UQ_PROP_QUADRATURE = 1,          /* Gauss-Hermite quadrature */
    UQ_PROP_MC_SAMPLING = 2,         /* Monte Carlo sampling */
    UQ_PROP_UNSCENTED = 3,           /* Unscented Transform */
    UQ_PROP_PCE_NON_INTRUSIVE = 4,   /* Non-intrusive PCE (regression) */
    UQ_PROP_PCE_GALERKIN = 5,        /* Intrusive PCE (Galerkin) */
    UQ_PROP_SG_COLLOCATION = 6,      /* Sparse grid stochastic collocation */
    UQ_PROP_KL_EXPANSION = 7,        /* Karhunen-Loève expansion */
    UQ_PROP_ROSENBLUETH = 8,         /* Rosenblueth point estimate method */
    UQ_PROP_SIGMA_2N = 9,            /* 2N+1 Sigma points */
    UQ_PROP_FOSM = 10,               /* First-Order Second-Moment */
    UQ_PROP_RBD = 11,                /* Reliability-Based Design point */
    UQ_PROP_SUBSET_SIM = 12,         /* Subset simulation */
    UQ_PROP_LINE_SAMPLING = 13       /* Line sampling */
} UQPropagationMethod;

/* --- Polynomial Chaos Expansion (PCE) --- */

typedef enum {
    UQ_PCE_HERMITE = 0,       /* Gaussian measure */
    UQ_PCE_LEGENDRE = 1,      /* Uniform measure */
    UQ_PCE_LAGUERRE = 2,      /* Exponential/Gamma measure */
    UQ_PCE_JACOBI = 3,        /* Beta measure */
    UQ_PCE_GEGENBAUER = 4,    /* Specific Jacobi case */
    UQ_PCE_CHEBYSHEV = 5      /* Chebyshev polynomials */
} UQPCEType;

typedef struct {
    UQPCEType basis_type;
    int n_inputs;               /* Stochastic dimension */
    int p_order;                /* Polynomial order */
    int n_basis_functions;      /* Total PCE terms = (p + d)! / (p! d!) */
    int** multi_indices;        /* [n_basis][n_inputs] — multi-index set */

    /* Coefficients */
    double* coefficients;       /* [n_basis] */

    /* One-dimensional basis recurrence */
    double (*alpha_recur)(int k, double param);   /* Recurrence alpha */
    double (*beta_recur)(int k, double param);    /* Recurrence beta */

    /* Quadrature */
    double* quad_nodes;         /* [n_quad × n_inputs] */
    double* quad_weights;       /* [n_quad] */
    int n_quadrature;

    /* Sobol indices from PCE coefficients */
    double* sobol_main_indices;
    double* sobol_total_indices;

    /* Training data (non-intrusive) */
    double* X_train;
    double* y_train;
    int n_train;

    /* Validation */
    double loo_error;
    double r_squared;
    double q_squared;           /* Predictive squared correlation */

    bool converged;
} UQPCE;

/* --- Sparse Grid / Stochastic Collocation --- */

typedef struct {
    int n_inputs;
    int level;                   /* Sparse grid level */
    int n_nodes;
    double* nodes;              /* [n_nodes × n_inputs] */
    double* weights;            /* [n_nodes] */

    /* Interpolation */
    double* function_values;    /* Model evaluations at nodes */
    int (*basis_index)(int level, int dim);  /* Level-to-index mapping */
} UQSparseGrid;

/* --- Karhunen-Loève Expansion --- */

typedef struct {
    int n_spatial_points;
    int n_retained_modes;
    double* spatial_grid;
    double* mean_function;
    double* eigenvalues;        /* [n_modes] — decreasing */
    double** eigenfunctions;    /* [n_modes][n_spatial] — L2-orthonormal */
    double variance_explained;  /* Fraction of total variance captured */
    bool is_separable;          /* Kernel is separable */
} UQKarhunenLoeve;

/* --- Unscented Transform --- */

typedef struct {
    int n_dim;                    /* Input dimension */
    int n_sigma;                  /* Number of sigma points (2n+1 typically) */
    double* sigma_points;         /* [n_sigma × n_dim] */
    double* weights_mean;         /* [n_sigma] */
    double* weights_cov;          /* [n_sigma] */
    double alpha;                 /* Spread parameter */
    double beta;                  /* Prior knowledge (2 for Gaussian) */
    double kappa;                 /* Secondary scaling (3-n for Gaussian) */
    double lambda;                /* α²(n + κ) - n */
} UQUnscentedTransform;

/* --- Reliability Analysis --- */

typedef struct {
    double beta_hasofer_lind;    /* Reliability index (FORM) */
    double failure_probability;
    double* design_point;        /* Most Probable Point (MPP) in standard normal */
    int n_iterations;
    bool converged;
    double* importance_direction; /* For line sampling */
} UQReliability;

/* --- API: Linearization & Moment Propagation --- */

void uq_fosm_propagate(double (*f)(double*, void*), void* f_data,
    double* x_mean, UQMatrix* x_cov, int dim,
    double* y_mean, double* y_variance);
void uq_linearization_gradient(double (*f)(double*, void*), void* f_data,
    double* x, int dim, double* gradient, double eps);
void uq_rosenblueth_2p(double (*f)(double*, void*), void* f_data,
    double* x_mean, double* x_std, int dim,
    double* y_mean, double* y_std);
void uq_rosenblueth_3p(double (*f)(double*, void*), void* f_data,
    double* x_mean, double* x_std, int dim,
    double* y_mean, double* y_std, double* y_skewness);

/* --- API: Unscented Transform --- */

UQUnscentedTransform* uq_ut_create(int n_dim, double alpha, double beta,
                                   double kappa);
void uq_ut_free(UQUnscentedTransform* ut);
void uq_ut_compute_sigma_points(UQUnscentedTransform* ut, double* mean,
                                UQMatrix* covariance);
void uq_ut_propagate(UQUnscentedTransform* ut,
    void (*f)(double*, double*, void*), void* f_data,
    int output_dim, double* y_mean, UQMatrix* y_cov);
void uq_ut_cross_covariance(UQUnscentedTransform* ut, double* input_sigma_y,
    int output_dim, UQMatrix* cross_cov);

/* --- API: Polynomial Chaos Expansion --- */

UQPCE* uq_pce_create(UQPCEType basis, int n_inputs, int p_order);
void uq_pce_free(UQPCE* pce);
void uq_pce_build_basis(UQPCE* pce);
void uq_pce_fit_regression(UQPCE* pce, double* X, double* y, int n_train);
void uq_pce_fit_quadrature(UQPCE* pce,
    double (*model)(double*, void*), void* model_data);
double uq_pce_evaluate(UQPCE* pce, double* xi);
void uq_pce_mean_variance(UQPCE* pce, double* mean, double* variance);
void uq_pce_sobol_indices(UQPCE* pce);
double uq_pce_compute_sobol_main(UQPCE* pce, int variable_idx);
double uq_pce_compute_sobol_total(UQPCE* pce, int variable_idx);
void uq_pce_pdf_approximation(UQPCE* pce, int n_points,
                              double* x_grid, double* pdf_vals);

/* --- API: Sparse Grid Collocation --- */

UQSparseGrid* uq_sg_create(int n_inputs, int level);
void uq_sg_free(UQSparseGrid* sg);
void uq_sg_build_nodes(UQSparseGrid* sg);
void uq_sg_evaluate_model(UQSparseGrid* sg,
    double (*model)(double*, void*), void* model_data);
double uq_sg_interpolate(UQSparseGrid* sg, double* point);
void uq_sg_mean_variance(UQSparseGrid* sg, double* mean, double* variance);

/* --- API: Karhunen-Loève --- */

UQKarhunenLoeve* uq_kl_create(double* grid, int n_points, double* kernel_params);
void uq_kl_free(UQKarhunenLoeve* kl);
void uq_kl_decompose(UQKarhunenLoeve* kl,
    double (*covariance_kernel)(double, double, double*), double* kernel_params);
void uq_kl_truncate(UQKarhunenLoeve* kl, double energy_threshold);
double uq_kl_reconstruct(UQKarhunenLoeve* kl, double* xi, double x);
double uq_kl_realization(UQKarhunenLoeve* kl, double* xi, double* out);

/* --- API: Reliability Analysis --- */

UQReliability* uq_relia_create(int dim);
void uq_relia_free(UQReliability* relia);
void uq_form_analysis(UQReliability* relia,
    double (*limit_state)(double*, void*), void* ls_data,
    double* x_mean, UQMatrix* x_cov);
void uq_subset_simulation(double (*limit_state)(double*, void*), void* ls_data,
    int dim, int n_samples_per_level, double p0,
    double* failure_prob, double* cov);
void uq_line_sampling(double (*limit_state)(double*, void*), void* ls_data,
    int dim, double* importance_dir, int n_lines,
    double* failure_prob, double* cov);

/* --- API: Propagation Helpers --- */

void uq_mc_propagate(int n_samples,
    double (*model)(double*, void*), void* model_data,
    UQDistribution** input_dists, int n_inputs,
    double* samples_out, double* mean, double* variance,
    double* skewness, double* kurtosis);
double uq_probability_of_failure(double* model_outputs, int n,
                                 double threshold);

#endif /* UQ_PROPAGATION_H */
