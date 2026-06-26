#ifndef PEM_CORE_H
#define PEM_CORE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ============================================================================
 * Prediction Error Method (PEM) — Core Types and Utilities
 *
 * Based on foundational works:
 *   Lennart Ljung — System Identification: Theory for the User (1987, 1999)
 *   Torsten Soderstrom & Petre Stoica — System Identification (1989)
 *   Karl J. Astrom & Peter Eykhoff — System Identification: A Survey (1971)
 *   Hirotugu Akaike — A New Look at the Statistical Model Identification (1974)
 *
 * The Prediction Error Method (PEM) is a unifying framework for system
 * identification where model parameters are estimated by minimizing the
 * prediction error between measured output and one-step-ahead prediction.
 *
 * Mathematical Foundation:
 *   Given data Z^N = {u(1), y(1), ..., u(N), y(N)}
 *   and model M(theta): y(t) = G(q,theta)*u(t) + H(q,theta)*e(t)
 *   Predictor: y_hat(t|theta) = H^{-1}(q)*G(q)*u(t) + (1-H^{-1}(q))*y(t)
 *   Prediction error: eps(t,theta) = y(t) - y_hat(t|theta)
 *   Criterion: V_N(theta) = (1/N) * sum_{t=1}^N l(eps(t,theta))
 *   Estimate: theta_hat_N = argmin_theta V_N(theta)
 * ============================================================================ */

/* --- Model Structure Families --- */
typedef enum {
    PEM_ARX   = 0,  /* AutoRegressive with eXogenous input */
    PEM_ARMAX = 1,  /* ARMA with eXogenous input */
    PEM_OE    = 2,  /* Output Error */
    PEM_BJ    = 3,  /* Box-Jenkins */
    PEM_SS    = 4,  /* State-Space (innovation form) */
    PEM_FIR   = 5,  /* Finite Impulse Response */
    PEM_ARARX = 6,  /* Generalized ARARX */
    PEM_USER  = 99  /* User-defined model structure */
} PEMModelStructure;

/* --- Optimization Algorithms --- */
typedef enum {
    PEM_OPT_GN     = 0,  /* Gauss-Newton */
    PEM_OPT_LM     = 1,  /* Levenberg-Marquardt */
    PEM_OPT_SGD    = 2,  /* Stochastic Gradient Descent (Robbins-Monro) */
    PEM_OPT_BFGS   = 3,  /* BFGS Quasi-Newton */
    PEM_OPT_NR     = 4   /* Newton-Raphson (full Hessian) */
} PEMOptimizationAlgorithm;

/* --- Convergence Status --- */
typedef enum {
    PEM_CONVERGED         = 0,
    PEM_MAX_ITER          = 1,
    PEM_DIVERGED          = 2,
    PEM_SINGULAR_HESSIAN  = 3,
    PEM_LINE_SEARCH_FAIL  = 4,
    PEM_NOT_STARTED       = 5,
    PEM_GRADIENT_TOL      = 6
} PEMConvergenceStatus;

/* --- Polynomial Representation ---
 * P(q) = p_0 + p_1*q^{-1} + ... + p_{n-1}*q^{-(n-1)}
 * Stored as coeff[0..order-1]; coeff[0] = p_0.
 */
typedef struct {
    int     order;
    double *coeff;
} PEMPolynomial;

/* --- Transfer Function ---
 * G(q) = N(q)/D(q), D(0) = 1 (monic denominator)
 */
typedef struct {
    PEMPolynomial numerator;
    PEMPolynomial denominator;
} PEMTransferFunction;

/* --- Dynamic Data Object ---
 * Z^N = {u(1), y(1), ..., u(N), y(N)}
 */
typedef struct {
    int      N;
    double   Ts;
    double  *u;
    double  *y;
    double  *t;
    char    *name;
} PEMData;

/* --- PEM Optimization Options --- */
typedef struct {
    int                         max_iterations;
    double                      tol_param;
    double                      tol_gradient;
    double                      tol_cost;
    double                      lambda_init;
    double                      lambda_factor;
    double                      lambda_max;
    double                      lambda_min;
    PEMOptimizationAlgorithm    algorithm;
    bool                        verbose;
    bool                        compute_covariance;
    double                      line_search_c1;
    double                      line_search_rho;
    int                         max_line_search;
} PEMOptions;

/* --- PEM Estimation Result --- */
typedef struct {
    int                     npar;
    double                 *theta_hat;
    double                 *covariance;
    double                 *gradient;
    double                  loss_final;
    double                  loss_init;
    int                     iterations;
    double                  elapsed_sec;
    PEMConvergenceStatus    status;
    double                  condition_number;
    double                 *information_matrix;
} PEMResult;

/* --- Model Validation Statistics --- */
typedef struct {
    int      N;
    double   loss;
    double   aic;
    double   aicc;
    double   bic;
    double   fpe;
    double   fit_percent;
    double   residual_whiteness;
    double   crosscorr_max;
    int      npar;
    double   r_squared;
    double   adjusted_r_squared;
} PEMValidation;

/* ============================================================================
 * Core API Functions
 * ============================================================================ */

/* --- Memory Management --- */
PEMData* pem_data_alloc(int N);
void pem_data_free(PEMData *data);
PEMPolynomial pem_poly_alloc(int order);
void pem_poly_free(PEMPolynomial *p);
PEMResult* pem_result_alloc(int npar);
void pem_result_free(PEMResult *result);
PEMOptions pem_options_default(void);
PEMValidation* pem_validation_alloc(void);
void pem_validation_free(PEMValidation *v);

/* --- Polynomial Operations --- */
double pem_poly_apply(const PEMPolynomial *p, const double *u, int t);
PEMPolynomial pem_poly_add(const PEMPolynomial *a, const PEMPolynomial *b);
PEMPolynomial pem_poly_mul(const PEMPolynomial *a, const PEMPolynomial *b);
int pem_poly_long_division(const PEMPolynomial *num, const PEMPolynomial *den,
                           double *quotient, int max_terms,
                           double *remainder, int max_rem);
void pem_tf_simulate(const PEMTransferFunction *G, const double *u,
                     double *y, int N, const double *y0);

/* --- Utility Functions --- */
double pem_mean(const double *x, int n);
double pem_variance(const double *x, int n, double mean);
double pem_norm2(const double *v, int n);
double pem_dot(const double *a, const double *b, int n);
double pem_rms_error(const double *y, const double *y_hat, int N);
double pem_nrmse_fit(const double *y, const double *y_hat, int N);
void pem_result_print(const PEMResult *r);
void pem_validation_print(const PEMValidation *v);

static inline double pem_clamp(double x, double lo, double hi) {
    return (x < lo) ? lo : ((x > hi) ? hi : x);
}

static inline double pem_safe_div(double num, double den, double fallback) {
    return (den > -1e-15 && den < 1e-15) ? fallback : (num / den);
}

#endif /* PEM_CORE_H */
