/**
 * wh_identification.h ? Wiener-Hammerstein System Identification Algorithms
 *
 * Identification methods for estimating the parameters of a WH model
 * from input-output data {u(t), y(t)}_{t=1}^{N}.
 *
 * Three main approaches are implemented:
 *
 * 1. Best Linear Approximation (BLA):
 *    Uses multisine excitations to estimate the linear dynamics of L1 and L2
 *    via frequency-domain analysis. The nonlinear distortion is separated
 *    from the noise through averaging over multiple realizations.
 *
 * 2. Iterative Method:
 *    Alternating least-squares: fix two blocks, estimate the third.
 *    Cycle through L1, N, L2 until convergence.
 *
 * 3. Over-Parameterization:
 *    Represent the full WH model as a single (over-parameterized) linear
 *    regression, then project the solution back to the WH structure via
 *    singular value decomposition (SVD).
 *
 * References:
 *   - Schoukens, J., Vaes, M., & Pintelon, R. (2016). "Linear System
 *     Identification in a Nonlinear Setting." IEEE Control Systems, 36(3), 38-69.
 *   - Wills, A. et al. (2013). "Identification of Hammerstein-Wiener Models."
 *     Automatica, 49(1), 70-81.
 *   - Bai, E.W. (2002). "A blind approach to the Hammerstein-Wiener model
 *     identification." Automatica, 38(6), 967-979.
 *
 * Knowledge Level: L5 (Algorithms/Methods)
 */

#ifndef WH_IDENTIFICATION_H
#define WH_IDENTIFICATION_H

#include "wh_model.h"

/* ??? Identification configuration ??????????????????????????????????????? */

/**
 * WH_IdentConfig ? Configuration for identification algorithm.
 */
typedef struct {
    wh_id_method_t  method;         /* Identification method to use          */
    int             max_iterations; /* Maximum iterations for iterative method*/
    double          tolerance;      /* Convergence tolerance for parameters   */
    int             order_L1;       /* Order of first linear block (0=auto)  */
    int             order_L2;       /* Order of second linear block (0=auto) */
    int             nl_degree;      /* Polynomial degree for nonlinearity     */
    int             nl_type;        /* wh_nl_type_t for nonlinearity model    */
    double          lambda;         /* Regularization parameter (?0)         */
    int             use_arma_noise; /* Flag: use ARMA noise model (1=yes)    */
    int             verbosity;      /* 0=silent, 1=progress, 2=debug         */
} WH_IdentConfig;

/* ??? Identification results ????????????????????????????????????????????? */

/**
 * WH_IdentResult ? Results of identification procedure.
 */
typedef struct {
    WH_Model*       model;          /* Identified WH model                    */
    int             iterations;     /* Number of iterations performed         */
    double          final_loss;     /* Final loss function value              */
    double          fit_percent;    /* FIT metric on identification data      */
    double          aic;            /* Akaike Information Criterion           */
    double          bic;            /* Bayesian Information Criterion         */
    double          convergence_measure; /* ||?_{k+1} - ?_k|| / ||?_k||       */
    int             converged;      /* Flag: algorithm converged              */
    int             n_parameters;   /* Number of estimated parameters         */
    double*         loss_history;   /* Loss value per iteration (malloc'd)    */
    int             loss_history_len;/* Length of loss history                */
} WH_IdentResult;

/* ??? Core identification API ???????????????????????????????????????????? */

/**
 * wh_ident_config_default ? Get default identification configuration.
 *
 * Returns a WH_IdentConfig with sensible defaults:
 *   method = WH_ID_ITERATIVE, max_iterations = 50, tolerance = 1e-6,
 *   order_L1 = 2, order_L2 = 2, nl_degree = 3, nl_type = WH_NL_POLYNOMIAL,
 *   lambda = 0.0, verbosity = 0.
 */
WH_IdentConfig wh_ident_config_default(void);

/**
 * wh_identify ? Identify a WH model from input-output data.
 *
 * @param u          Input data (length N).
 * @param y          Output data (length N).
 * @param N          Number of data points.
 * @param config     Identification configuration.
 * @param result     Output: identification results (caller allocates).
 * @return           WH_STATUS_OK on success, error code otherwise.
 */
int wh_identify(const double* u, const double* y, int N,
                const WH_IdentConfig* config, WH_IdentResult* result);

/**
 * wh_ident_result_free ? Free resources associated with identification results.
 *
 * @param result  Results to free.
 */
void wh_ident_result_free(WH_IdentResult* result);

/* ??? BLA-based identification ??????????????????????????????????????????? */

/**
 * wh_ident_bla ? Identify WH model using Best Linear Approximation approach.
 *
 * Algorithm:
 *   1. Estimate BLA G_BLA from multisines at different excitation levels.
 *   2. Separate L1 and L2 from BLA using pole/zero partitioning.
 *   3. Estimate nonlinearity from residual (y - y_linear).
 *
 * Required: input data from multisine experiment with multiple realizations.
 *
 * @param u          Input data (multisine).
 * @param y          Output data (multiple realizations concatenated).
 * @param n_periods  Number of periods per realization.
 * @param n_realizations Number of independent realizations.
 * @param config     Configuration.
 * @param result     Output results.
 * @return           0 on success.
 */
int wh_ident_bla(const double* u, const double* y,
                  int n_periods, int n_realizations,
                  const WH_IdentConfig* config, WH_IdentResult* result);

/* ??? Iterative identification ??????????????????????????????????????????? */

/**
 * wh_ident_iterative ? Identify WH model using iterative alternating estimation.
 *
 * Algorithm:
 *   1. Initialize L1, L2, N (L1 & L2 as identity, N as linear).
 *   2. Fix L2, N ? estimate L1 via linear regression on filtered data.
 *   3. Fix L1, L2 ? estimate N via linear regression on intermediate signals.
 *   4. Fix L1, N ? estimate L2 via linear regression on filtered data.
 *   5. Check convergence; if not converged, goto 2.
 *
 * @param u         Input data.
 * @param y         Output data.
 * @param N         Number of samples.
 * @param config    Configuration.
 * @param result    Output results.
 * @return          0 on success.
 */
int wh_ident_iterative(const double* u, const double* y, int N,
                        const WH_IdentConfig* config, WH_IdentResult* result);

/* ??? Over-parameterization identification ??????????????????????????????? */

/**
 * wh_ident_overparam ? Identify WH model via over-parameterization + SVD.
 *
 * Algorithm:
 *   1. Expand WH model into a single high-dimensional linear-in-parameters form.
 *   2. Solve linear least-squares to get over-parameterized estimate.
 *   3. Apply SVD to the parameter matrix to extract low-rank WH structure.
 *   4. Refine via gradient descent if needed.
 *
 * This method avoids the local minima issues of iterative methods.
 *
 * @param u         Input data.
 * @param y         Output data.
 * @param N         Number of samples.
 * @param config    Configuration.
 * @param result    Output results.
 * @return          0 on success.
 */
int wh_ident_overparam(const double* u, const double* y, int N,
                        const WH_IdentConfig* config, WH_IdentResult* result);

/* ??? PEM gradient-based identification ?????????????????????????????????? */

/**
 * wh_ident_pem_gradient ? Identify WH model via Prediction Error Method
 * with gradient-based optimization.
 *
 * Minimizes: V(?) = 1/(2N) * ?_{t=1}^{N} ?(t,?)?
 * where ?(t,?) = y(t) - ?(t|?) is the prediction error.
 *
 * Uses Levenberg-Marquardt for robust convergence.
 *
 * @param u         Input data.
 * @param y         Output data.
 * @param N         Number of samples.
 * @param config    Configuration.
 * @param result    Output results.
 * @return          0 on success.
 */
int wh_ident_pem_gradient(const double* u, const double* y, int N,
                           const WH_IdentConfig* config, WH_IdentResult* result);

/* ??? Model order selection ?????????????????????????????????????????????? */

/**
 * wh_ident_order_selection ? Select optimal model orders via grid search
 * using AIC or BIC criterion.
 *
 * @param u           Input data.
 * @param y           Output data.
 * @param N           Number of samples.
 * @param order_L1_min Minimum L1 order.
 * @param order_L1_max Maximum L1 order.
 * @param order_L2_min Minimum L2 order.
 * @param order_L2_max Maximum L2 order.
 * @param nl_degree_min Minimum polynomial degree.
 * @param nl_degree_max Maximum polynomial degree.
 * @param use_bic     Use BIC (1) or AIC (0) for selection.
 * @param best_order_L1 Output: optimal L1 order.
 * @param best_order_L2 Output: optimal L2 order.
 * @param best_nl_deg  Output: optimal nonlinearity degree.
 * @return            0 on success.
 */
int wh_ident_order_selection(const double* u, const double* y, int N,
                              int order_L1_min, int order_L1_max,
                              int order_L2_min, int order_L2_max,
                              int nl_degree_min, int nl_degree_max,
                              int use_bic,
                              int* best_order_L1, int* best_order_L2,
                              int* best_nl_deg);

/* ??? Information criteria ??????????????????????????????????????????????? */

/**
 * wh_ident_compute_aic ? Compute Akaike Information Criterion.
 *
 * AIC = N * ln(MSE) + 2*k
 * where N is number of data points, MSE is mean squared error, k is number
 * of estimated parameters.
 *
 * @param mse           Mean squared error.
 * @param n_params      Number of estimated parameters.
 * @param n_data        Number of data points.
 * @return              AIC value.
 */
double wh_ident_compute_aic(double mse, int n_params, int n_data);

/**
 * wh_ident_compute_bic ? Compute Bayesian Information Criterion.
 *
 * BIC = N * ln(MSE) + k * ln(N)
 *
 * @param mse           Mean squared error.
 * @param n_params      Number of estimated parameters.
 * @param n_data        Number of data points.
 * @return              BIC value.
 */
double wh_ident_compute_bic(double mse, int n_params, int n_data);

#endif /* WH_IDENTIFICATION_H */
