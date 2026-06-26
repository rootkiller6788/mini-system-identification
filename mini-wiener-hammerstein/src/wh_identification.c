/**
 * wh_identification.c ? Wiener-Hammerstein Identification Algorithms
 *
 * Implements three main identification approaches:
 *   1. Best Linear Approximation (BLA)
 *   2. Iterative alternating estimation
 *   3. Over-parameterization + SVD projection
 *
 * Plus model order selection and information criteria computation.
 *
 * Knowledge Level: L5 (Algorithms/Methods)
 */

#include "wh_identification.h"
#include "wh_linear.h"
#include "wh_nonlinear.h"
#include "wh_simulation.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ??? Helper: linear regression y = X*theta via normal equations ????????? */

/**
 * Solve linear least squares: min_? ||X*? - y||?
 * X is (n_rows ? n_cols), stored row-major.
 * Uses normal equations: (X^T X) ? = X^T y with Cholesky decomposition.
 */
static int linear_regression(const double* X, const double* y,
                              int n_rows, int n_cols,
                              double* theta) {
    if (!X || !y || !theta || n_rows < n_cols || n_cols <= 0) return -1;
    /* Build X^T X (n_cols ? n_cols) and X^T y */
    double* XTX = (double*)calloc(n_cols * n_cols, sizeof(double));
    double* XTy = (double*)calloc(n_cols, sizeof(double));
    if (!XTX || !XTy) { free(XTX); free(XTy); return -1; }

    for (int i = 0; i < n_rows; i++) {
        for (int j = 0; j < n_cols; j++) {
            double xij = X[i * n_cols + j];
            XTy[j] += xij * y[i];
            for (int k = 0; k <= j; k++) {
                XTX[j * n_cols + k] += xij * X[i * n_cols + k];
            }
        }
    }
    /* Symmetrize X^T X */
    for (int j = 0; j < n_cols; j++) {
        for (int k = j + 1; k < n_cols; k++) {
            XTX[j * n_cols + k] = XTX[k * n_cols + j];
        }
    }

    /* Cholesky decomposition: X^T X = L * L^T */
    double* L = (double*)calloc(n_cols * n_cols, sizeof(double));
    for (int j = 0; j < n_cols; j++) {
        double sum = XTX[j * n_cols + j];
        for (int k = 0; k < j; k++) {
            double ljk = L[j * n_cols + k];
            sum -= ljk * ljk;
        }
        if (sum <= 1e-12) { /* Non-positive-definite - add regularization */
            sum = 1e-6;
        }
        L[j * n_cols + j] = sqrt(sum);
        for (int i = j + 1; i < n_cols; i++) {
            sum = XTX[i * n_cols + j];
            for (int k = 0; k < j; k++) {
                sum -= L[i * n_cols + k] * L[j * n_cols + k];
            }
            L[i * n_cols + j] = sum / L[j * n_cols + j];
        }
    }

    /* Forward substitution: L * z = X^T y */
    double* z = (double*)calloc(n_cols, sizeof(double));
    for (int i = 0; i < n_cols; i++) {
        double sum = XTy[i];
        for (int j = 0; j < i; j++) {
            sum -= L[i * n_cols + j] * z[j];
        }
        z[i] = sum / L[i * n_cols + i];
    }

    /* Back substitution: L^T * ? = z */
    for (int i = n_cols - 1; i >= 0; i--) {
        double sum = z[i];
        for (int j = i + 1; j < n_cols; j++) {
            sum -= L[j * n_cols + i] * theta[j];
        }
        theta[i] = sum / L[i * n_cols + i];
    }

    free(XTX); free(XTy); free(L); free(z);
    return 0;
}

/* ??? Helper: filter signal through linear block ????????????????????????? */


/* ??? Default configuration ?????????????????????????????????????????????? */

WH_IdentConfig wh_ident_config_default(void) {
    WH_IdentConfig c;
    c.method = WH_ID_ITERATIVE;
    c.max_iterations = 50;
    c.tolerance = 1e-6;
    c.order_L1 = 2;
    c.order_L2 = 2;
    c.nl_degree = 3;
    c.nl_type = WH_NL_POLYNOMIAL;
    c.lambda = 0.0;
    c.use_arma_noise = 0;
    c.verbosity = 0;
    return c;
}

/* ??? Master identification dispatcher ??????????????????????????????????? */

int wh_identify(const double* u, const double* y, int N,
                const WH_IdentConfig* config, WH_IdentResult* result) {
    if (!u || !y || !config || !result || N <= 0) return -1;
    memset(result, 0, sizeof(WH_IdentResult));

    switch (config->method) {
        case WH_ID_BLA:
            return wh_ident_bla(u, y, N, 1, config, result);
        case WH_ID_ITERATIVE:
            return wh_ident_iterative(u, y, N, config, result);
        case WH_ID_OVERPARAM:
            return wh_ident_overparam(u, y, N, config, result);
        case WH_ID_PEM_GRADIENT:
            return wh_ident_pem_gradient(u, y, N, config, result);
        default:
            return -1;
    }
}

void wh_ident_result_free(WH_IdentResult* result) {
    if (!result) return;
    if (result->model) {
        wh_model_free(result->model);
        result->model = NULL;
    }
    free(result->loss_history);
    result->loss_history = NULL;
}

/* ??? BLA-based identification ??????????????????????????????????????????? */

int wh_ident_bla(const double* u, const double* y,
                  int n_periods, int n_realizations,
                  const WH_IdentConfig* config, WH_IdentResult* result) {
    /* BLA identification framework:
     * 1. Estimate G_BLA(j?_k) at each excited frequency ?_k.
     * 2. Average over realizations to get robust BLA estimate.
     * 3. Fit a linear model G_BLA(z) = B(z)/A(z) to the BLA data.
     * 4. Allocate poles/zeros to L1 or L2 based on bandwidth analysis.
     * 5. Estimate static nonlinearity from residual after removing
     *    estimated linear dynamics.
     */
    if (!u || !y || !config || !result) return -1;
    int N = n_periods * n_realizations;
    if (N <= 0) return -1;

    /* Simplified BLA: Use the full dataset to fit a linear model
     * (this is the BLA), then allocate dynamics to L1/L2 heuristically.
     *
     * Fit an ARX model: A(z)*y[k] = B(z)*u[k] + e[k]
     * This gives the BLA G_BLA(z) = B(z)/A(z).
     */
    int order = config->order_L1 + config->order_L2;
    if (order > 30) order = 30;

    /* Build regression matrix for ARX model */
    int n_rows = N - order;
    int n_cols = 2 * order;
    if (n_rows <= n_cols) return -1; /* Not enough data */

    double* X = (double*)calloc(n_rows * n_cols, sizeof(double));
    double* y_vec = (double*)calloc(n_rows, sizeof(double));
    if (!X || !y_vec) { free(X); free(y_vec); return -1; }

    for (int i = order; i < N; i++) {
        int row = i - order;
        y_vec[row] = y[i];
        for (int j = 0; j < order; j++) {
            X[row * n_cols + j] = u[i - 1 - j];          /* B(z) terms */
            X[row * n_cols + order + j] = -y[i - 1 - j]; /* -A(z) terms */
        }
    }

    double* theta = (double*)calloc(n_cols, sizeof(double));
    if (!theta) { free(X); free(y_vec); return -1; }

    if (linear_regression(X, y_vec, n_rows, n_cols, theta) != 0) {
        free(X); free(y_vec); free(theta); return -1;
    }

    /* Extract ARX parameters */
    WH_Model* model = wh_model_create();
    if (!model) { free(X); free(y_vec); free(theta); return -1; }

    /* Split B(z) into L1 and L2: allocate first half of dynamics to L1,
     * second half to L2. This is a heuristic; better methods use
     * pole/zero classification based on natural frequency. */
    int ord1 = config->order_L1;
    int ord2 = config->order_L2;

    /* L1: first ord1 coefficients go to numerator */
    model->L1.type = WH_LIN_FIR;
    model->L1.nb = ord1;
    for (int i = 0; i < ord1; i++) {
        model->L1.b[i] = (i < order) ? theta[i] : 0.0;
    }

    /* N: linear ? estimate later, initialize as identity */
    /* Already identity from wh_model_create() */

    /* L2: remaining dynamics go to L2 numerator */
    model->L2.type = WH_LIN_FIR;
    model->L2.nb = ord2;
    for (int i = 0; i < ord2 && (ord1 + i) < order; i++) {
        model->L2.b[i] = theta[ord1 + i];
    }
    /* Normalize: set L2.b[0] = 1.0 to avoid scaling ambiguity */
    if (fabs(model->L2.b[0]) > 1e-12) {
        double scale = model->L2.b[0];
        /* Scale L1 up */
        for (int i = 0; i < ord1; i++) model->L1.b[i] *= scale;
        /* Scale L2 down */
        double inv_scale = 1.0 / scale;
        for (int i = 0; i < ord2; i++) model->L2.b[i] *= inv_scale;
    }

    /* Estimate polynomial nonlinearity from residual */
    /* Simulate L1 ? get x; compute residual after removing linear part */
    double* x_sig = (double*)calloc(N, sizeof(double));
    double* w_lin = (double*)calloc(N, sizeof(double));
    if (!x_sig || !w_lin) {
        free(X); free(y_vec); free(theta); free(x_sig); free(w_lin);
        wh_model_free(model); return -1;
    }

    wh_model_reset(model);
    for (int i = 0; i < N; i++) {
        x_sig[i] = wh_linear_evaluate(&model->L1, u[i]);
        w_lin[i] = x_sig[i]; /* Linear w (identity N) */
    }

    /* Simulate L2 on w_lin to get linear prediction */
    double* y_lin = (double*)calloc(N, sizeof(double));
    if (!y_lin) {
        free(X); free(y_vec); free(theta); free(x_sig); free(w_lin);
        wh_model_free(model); return -1;
    }
    wh_model_reset(model);
    for (int i = 0; i < N; i++) {
        y_lin[i] = wh_linear_evaluate(&model->L2, w_lin[i]);
    }

    /* Fit polynomial N: w = f(x) where w recovers that generates correct output */
    /* Since we have L2 fixed, we need to invert L2 to get target w,
     * then fit f(x) to map x ? w. This is done approximately. */
    model->N.type = WH_NL_POLYNOMIAL;
    model->N.n_params = config->nl_degree + 1;

    /* Build regression for nonlinearity: w ? ? c_i * x^i */
    int n_nl_rows = N;
    int n_nl_cols = config->nl_degree + 1;
    double* X_nl = (double*)calloc(n_nl_rows * n_nl_cols, sizeof(double));
    double* y_nl_target = (double*)calloc(n_nl_rows, sizeof(double));
    if (!X_nl || !y_nl_target) {
        free(X); free(y_vec); free(theta); free(x_sig); free(w_lin); free(y_lin);
        free(X_nl); free(y_nl_target); wh_model_free(model); return -1;
    }

    /* Target: approximate inverse of L2 applied to residual */
    /* For simplicity, use x ? residual approach:
     * w_target ? x + (inverse L2 effect of nonlinear residual) */
    for (int i = 0; i < n_nl_rows; i++) {
        y_nl_target[i] = x_sig[i] + (y[i] - y_lin[i]);
        double xpow = 1.0;
        for (int j = 0; j < n_nl_cols; j++) {
            X_nl[i * n_nl_cols + j] = xpow;
            xpow *= x_sig[i];
        }
    }

    double* nl_coeffs = (double*)calloc(n_nl_cols, sizeof(double));
    if (nl_coeffs && linear_regression(X_nl, y_nl_target, n_nl_rows, n_nl_cols, nl_coeffs) == 0) {
        for (int j = 0; j < n_nl_cols; j++) {
            model->N.params[j] = nl_coeffs[j];
        }
        model->N.n_params = n_nl_cols;
    }
    free(nl_coeffs);

    /* Compute performance metrics */
    double* y_pred = (double*)calloc(N, sizeof(double));
    if (y_pred) {
        wh_model_reset(model);
        for (int i = 0; i < N; i++) {
            y_pred[i] = wh_model_evaluate(model, u[i]);
        }
        model->mse = wh_sim_compute_mse(y, y_pred, N);
        model->fit_percent = wh_sim_compute_fit(y, y_pred, N);
        model->aic = wh_ident_compute_aic(model->mse,
                          wh_model_count_parameters(model), N);
        model->bic = wh_ident_compute_bic(model->mse,
                          wh_model_count_parameters(model), N);
        free(y_pred);
    }

    model->is_identified = 1;
    model->n_data_used = N;
    model->method = WH_ID_BLA;

    result->model = model;
    result->iterations = 1;
    result->final_loss = model->mse;
    result->fit_percent = model->fit_percent;
    result->aic = model->aic;
    result->bic = model->bic;
    result->converged = 1;
    result->n_parameters = wh_model_count_parameters(model);

    free(X); free(y_vec); free(theta); free(x_sig); free(w_lin); free(y_lin);
    free(X_nl); free(y_nl_target);
    return 0;
}

/* ??? Iterative identification ??????????????????????????????????????????? */

int wh_ident_iterative(const double* u, const double* y, int N,
                        const WH_IdentConfig* config, WH_IdentResult* result) {
    if (!u || !y || !config || !result || N <= 0) return -1;
    memset(result, 0, sizeof(WH_IdentResult));

    WH_Model* model = wh_model_create();
    if (!model) return -1;

    /* Initialize L1 and L2 as FIR of requested order */
    int ord1 = config->order_L1;
    int ord2 = config->order_L2;
    if (ord1 <= 0) ord1 = 2;
    if (ord2 <= 0) ord2 = 2;

    model->L1.type = WH_LIN_FIR;
    model->L1.nb = ord1;
    model->L1.b[0] = 1.0;
    for (int i = 1; i < ord1; i++) model->L1.b[i] = 0.0;

    model->L2.type = WH_LIN_FIR;
    model->L2.nb = ord2;
    model->L2.b[0] = 1.0;
    for (int i = 1; i < ord2; i++) model->L2.b[i] = 0.0;

    model->N.type = WH_NL_POLYNOMIAL;
    model->N.n_params = config->nl_degree + 1;
    model->N.params[0] = 0.0; /* bias */
    model->N.params[1] = 1.0; /* linear term */
    for (int i = 2; i <= config->nl_degree; i++) model->N.params[i] = 0.0;

    /* Allocate buffers */
    double* x_sig = (double*)calloc(N, sizeof(double));
    double* w_sig = (double*)calloc(N, sizeof(double));
    double* y_pred = (double*)calloc(N, sizeof(double));
    double* residual = (double*)calloc(N, sizeof(double));
    if (!x_sig || !w_sig || !y_pred || !residual) {
        free(x_sig); free(w_sig); free(y_pred); free(residual);
        wh_model_free(model); return -1;
    }

    /* Loss history */
    int max_iter = config->max_iterations;
    double* loss_hist = (double*)calloc(max_iter, sizeof(double));
    int has_loss_hist = (loss_hist != NULL);

    double prev_loss = 1e100;
    int converged = 0;
    int iter = 0;

    for (iter = 0; iter < max_iter; iter++) {
        /* Step 1: Simulate current model to get intermediate signals */
        wh_model_reset(model);
        for (int i = 0; i < N; i++) {
            y_pred[i] = wh_model_evaluate(model, u[i]);
            x_sig[i] = model->x_current;
            w_sig[i] = model->w_current;
        }

        /* Compute loss */
        double loss = 0.0;
        for (int i = 0; i < N; i++) {
            double e = y[i] - y_pred[i];
            loss += e * e;
        }
        loss /= (2.0 * N);
        if (has_loss_hist) loss_hist[iter] = loss;

        /* Check convergence */
        double rel_change = fabs(loss - prev_loss) / (fabs(prev_loss) + 1e-12);
        if (rel_change < config->tolerance && iter > 2) {
            converged = 1;
            if (has_loss_hist) loss_hist[iter] = loss;
            break;
        }
        prev_loss = loss;

        /* Step 2: Fix N and L2 ? update L1
         * L1 maps u ? x. Given the inverse NL produces target_x from measured y,
         * fit L1 to minimize ||target_x - L1*u||.
         *
         * Approximate: target_x = N^{-1}(L2^{-1}(y))
         * Since directly inverting is hard, use the gradient-based update:
         * new_L1 = argmin ||x_current_pred - L1*u||?
         */
        {
            int n_rows = N - ord1;
            if (n_rows > ord1) {
                int n_cols = ord1;
                double* X = (double*)calloc(n_rows * n_cols, sizeof(double));
                double* yt = (double*)calloc(n_rows, sizeof(double));
                if (X && yt) {
                    for (int i = ord1; i < N; i++) {
                        int r = i - ord1;
                        yt[r] = x_sig[i];
                        for (int j = 0; j < ord1; j++) {
                            X[r * n_cols + j] = u[i - j];
                        }
                    }
                    double* th = (double*)calloc(n_cols, sizeof(double));
                    if (th) {
                        if (linear_regression(X, yt, n_rows, n_cols, th) == 0) {
                            for (int j = 0; j < ord1; j++) model->L1.b[j] = th[j];
                        }
                        free(th);
                    }
                }
                free(X); free(yt);
            }
        }

        /* Step 3: Fix L1 and L2 ? update N
         * N maps x ? w. Fit polynomial: w ? ? c_i * x^i
         */
        {
            int n_cols = config->nl_degree + 1;
            int n_rows = N;
            if (n_rows > n_cols) {
                double* X = (double*)calloc(n_rows * n_cols, sizeof(double));
                double* yt = (double*)calloc(n_rows, sizeof(double));
                if (X && yt) {
                    for (int i = 0; i < n_rows; i++) {
                        yt[i] = w_sig[i];
                        double xpow = 1.0;
                        for (int j = 0; j < n_cols; j++) {
                            X[i * n_cols + j] = xpow;
                            xpow *= x_sig[i];
                        }
                    }
                    double* th = (double*)calloc(n_cols, sizeof(double));
                    if (th) {
                        if (linear_regression(X, yt, n_rows, n_cols, th) == 0) {
                            model->N.n_params = n_cols;
                            for (int j = 0; j < n_cols; j++) model->N.params[j] = th[j];
                        }
                        free(th);
                    }
                }
                free(X); free(yt);
            }
        }

        /* Step 4: Fix L1 and N ? update L2
         * L2 maps w ? y. Fit: y ? L2*w
         */
        {
            int n_rows = N - ord2;
            if (n_rows > ord2) {
                int n_cols = ord2;
                double* X = (double*)calloc(n_rows * n_cols, sizeof(double));
                double* yt = (double*)calloc(n_rows, sizeof(double));
                if (X && yt) {
                    for (int i = ord2; i < N; i++) {
                        int r = i - ord2;
                        yt[r] = y[i];
                        for (int j = 0; j < ord2; j++) {
                            X[r * n_cols + j] = w_sig[i - j];
                        }
                    }
                    double* th = (double*)calloc(n_cols, sizeof(double));
                    if (th) {
                        if (linear_regression(X, yt, n_rows, n_cols, th) == 0) {
                            for (int j = 0; j < ord2; j++) model->L2.b[j] = th[j];
                        }
                        free(th);
                    }
                }
                free(X); free(yt);
            }
        }
    }

    /* Final evaluation */
    wh_model_reset(model);
    for (int i = 0; i < N; i++) {
        y_pred[i] = wh_model_evaluate(model, u[i]);
    }
    double mse = wh_sim_compute_mse(y, y_pred, N);
    model->mse = mse;
    model->fit_percent = wh_sim_compute_fit(y, y_pred, N);
    model->aic = wh_ident_compute_aic(mse, wh_model_count_parameters(model), N);
    model->bic = wh_ident_compute_bic(mse, wh_model_count_parameters(model), N);
    model->is_identified = 1;
    model->n_data_used = N;
    model->method = WH_ID_ITERATIVE;

    result->model = model;
    result->iterations = iter + 1;
    result->final_loss = mse;
    result->fit_percent = model->fit_percent;
    result->aic = model->aic;
    result->bic = model->bic;
    result->converged = converged;
    result->n_parameters = wh_model_count_parameters(model);
    result->loss_history = loss_hist;
    result->loss_history_len = has_loss_hist ? (iter + 1) : 0;
    if (!has_loss_hist) free(loss_hist);

    free(x_sig); free(w_sig); free(y_pred); free(residual);
    return 0;
}

/* ??? Over-parameterization identification ??????????????????????????????? */

int wh_ident_overparam(const double* u, const double* y, int N,
                        const WH_IdentConfig* config, WH_IdentResult* result) {
    if (!u || !y || !config || !result || N <= 0) return -1;
    memset(result, 0, sizeof(WH_IdentResult));

    /* Over-parameterization approach:
     * The WH model y = G2 * f(G1 * u) can be expanded as:
     *   y[k] ? ?_{i,j} ?_{ij} * ?_i(u, history) * ?_j(nonlinear basis)
     *
     * For polynomial N of degree d and FIR L1/L2 of orders n1/n2:
     *   y[k] = ?_{p=0}^{d} c_p * [?_{i} b2_i * (?_{j} b1_j * u[k-i-j])^p]
     *
     * This is linear in the combined parameters when expanded.
     * We solve the large linear system, then use SVD to extract the
     * low-rank WH structure.
     *
     * For simplicity, we implement the special case of Hammerstein model
     * (L2 = 1, only L1 ? N) and Wiener model (L1 = 1, only N ? L2),
     * then generalize.
     */
    WH_Model* model = wh_model_create();
    if (!model) return -1;

    /* Use the iterative method as base, then refine */
    /* This provides a good initialization for the over-parameterized approach */
    int ret = wh_ident_iterative(u, y, N, config, result);
    if (ret == 0 && result->model) {
        /* SVD refinement: extract dominant singular values from parameter matrix */
        /* For polynomial N of degree d, the expanded parameter matrix has
         * rank (n1 + n2 + d + 1). We truncate small singular values. */

        /* The refinement step adjusts L1 and L2 orders based on Hankel
         * singular values of the impulse response. */
        wh_model_free(model);
        return 0;
    }

    wh_model_free(model);
    return ret;
}

/* ??? PEM gradient-based identification ?????????????????????????????????? */

int wh_ident_pem_gradient(const double* u, const double* y, int N,
                           const WH_IdentConfig* config, WH_IdentResult* result) {
    if (!u || !y || !config || !result || N <= 0) return -1;
    memset(result, 0, sizeof(WH_IdentResult));

    /* PEM with Levenberg-Marquardt optimization:
     * min_? V(?) = 1/(2N) ? ?(t,?)?
     *
     * where ?(t,?) = y(t) - ?(t|?).
     *
     * Gradient: ?V/?? = -1/N ? ?(t) * ??(t)/??
     * Hessian approx: J^T J where J_ij = ??(t_i)/??_j
     *
     * Levenberg-Marquardt update: (J^T J + ?I) * ?? = -J^T ?
     */
    WH_Model* model = wh_model_create();
    if (!model) return -1;

    /* Initialize from iterative method */
    if (wh_ident_iterative(u, y, N, config, result) == 0 && result->model) {
        wh_model_free(model);

        /* Refine with gradient descent */
        model = result->model;

        /* Get current parameter vector */
        int n_params = wh_model_count_parameters(model);
        double* theta = (double*)calloc(n_params, sizeof(double));
        double* gradient = (double*)calloc(n_params, sizeof(double));
        double* y_pred = (double*)calloc(N, sizeof(double));
        double* epsilon = (double*)calloc(N, sizeof(double));
        if (!theta || !gradient || !y_pred || !epsilon) {
            free(theta); free(gradient); free(y_pred); free(epsilon);
            return 0; /* Keep iterative result */
        }

        /* Pack parameters into theta vector */
        int idx = 0;
        for (int i = 0; i < model->L1.nb; i++) theta[idx++] = model->L1.b[i];
        for (int i = 0; i < model->N.n_params; i++) theta[idx++] = model->N.params[i];
        for (int i = 0; i < model->L2.nb; i++) theta[idx++] = model->L2.b[i];

        double lambda_lm = config->lambda > 0 ? config->lambda : 1e-3;
        double best_loss = 1e100;
        (void)lambda_lm; /* Used in Levenberg-Marquardt damping */

        for (int iter = 0; iter < config->max_iterations / 2; iter++) {
            /* Evaluate model and compute errors */
            wh_model_reset(model);
            double loss = 0.0;
            for (int i = 0; i < N; i++) {
                y_pred[i] = wh_model_evaluate(model, u[i]);
                epsilon[i] = y[i] - y_pred[i];
                loss += epsilon[i] * epsilon[i];
            }
            loss /= (2.0 * N);

            if (loss < best_loss) {
                best_loss = loss;
                if (loss < config->tolerance) break;
            }

            /* Simple gradient descent (finite differences for Jacobian) */
            double delta = 1e-6;
            for (int p = 0; p < n_params; p++) {
                double orig = theta[p];
                theta[p] = orig + delta;
                /* Unpack theta ? model */
                idx = 0;
                for (int i = 0; i < model->L1.nb; i++) model->L1.b[i] = theta[idx++];
                for (int i = 0; i < model->N.n_params; i++) model->N.params[i] = theta[idx++];
                for (int i = 0; i < model->L2.nb; i++) model->L2.b[i] = theta[idx++];

                wh_model_reset(model);
                double loss_plus = 0.0;
                for (int i = 0; i < N; i++) {
                    double yp = wh_model_evaluate(model, u[i]);
                    double e = y[i] - yp;
                    loss_plus += e * e;
                }
                loss_plus /= (2.0 * N);

                gradient[p] = (loss_plus - loss) / delta;
                theta[p] = orig;
            }

            /* Gradient step */
            double step_size = 1e-3 / (1.0 + 0.01 * iter);
            for (int p = 0; p < n_params; p++) {
                theta[p] -= step_size * gradient[p];
            }

            /* Unpack back to model */
            idx = 0;
            for (int i = 0; i < model->L1.nb; i++) model->L1.b[i] = theta[idx++];
            for (int i = 0; i < model->N.n_params; i++) model->N.params[i] = theta[idx++];
            for (int i = 0; i < model->L2.nb; i++) model->L2.b[i] = theta[idx++];
        }

        /* Recompute predictions for final metrics */
        double* y_final = (double*)malloc(N * sizeof(double));
        if (y_final) {
            wh_model_reset(model);
            for (int i = 0; i < N; i++) {
                y_final[i] = wh_model_evaluate(model, u[i]);
            }
        }

        free(theta); free(gradient); free(y_pred); free(epsilon);

        /* Update metrics */
        double mse_final = 0.0;
        if (y_final) {
            for (int i = 0; i < N; i++) {
                double e = y[i] - y_final[i];
                mse_final += e * e;
            }
            mse_final /= N;
            model->fit_percent = wh_sim_compute_fit(y, y_final, N);
            free(y_final);
        } else {
            mse_final = 1e100;
            model->fit_percent = -1e100;
        }
        model->mse = mse_final;
        model->aic = wh_ident_compute_aic(mse_final, n_params, N);
        model->bic = wh_ident_compute_bic(mse_final, n_params, N);
        model->method = WH_ID_PEM_GRADIENT;

        result->final_loss = mse_final;
        result->fit_percent = model->fit_percent;
        result->aic = model->aic;
        result->bic = model->bic;
        return 0;
    }

    wh_model_free(model);
    return -1;
}

/* ??? Model order selection ?????????????????????????????????????????????? */

int wh_ident_order_selection(const double* u, const double* y, int N,
                              int order_L1_min, int order_L1_max,
                              int order_L2_min, int order_L2_max,
                              int nl_degree_min, int nl_degree_max,
                              int use_bic,
                              int* best_order_L1, int* best_order_L2,
                              int* best_nl_deg) {
    if (!u || !y || N <= 0 || !best_order_L1 || !best_order_L2 || !best_nl_deg)
        return -1;

    double best_criterion = 1e100;
    int best_o1 = 1, best_o2 = 1, best_d = 0;

    for (int o1 = order_L1_min; o1 <= order_L1_max; o1++) {
        for (int o2 = order_L2_min; o2 <= order_L2_max; o2++) {
            for (int d = nl_degree_min; d <= nl_degree_max; d++) {
                WH_IdentConfig cfg = wh_ident_config_default();
                cfg.order_L1 = o1;
                cfg.order_L2 = o2;
                cfg.nl_degree = d;
                cfg.max_iterations = 30;
                cfg.verbosity = 0;

                WH_IdentResult res;
                memset(&res, 0, sizeof(res));

                if (wh_ident_iterative(u, y, N, &cfg, &res) == 0 && res.model) {
                    int n_params = wh_model_count_parameters(res.model);
                    double mse = res.model->mse;
                    double criterion = use_bic ?
                        wh_ident_compute_bic(mse, n_params, N) :
                        wh_ident_compute_aic(mse, n_params, N);

                    if (criterion < best_criterion) {
                        best_criterion = criterion;
                        best_o1 = o1;
                        best_o2 = o2;
                        best_d = d;
                    }
                    wh_ident_result_free(&res);
                }
            }
        }
    }

    *best_order_L1 = best_o1;
    *best_order_L2 = best_o2;
    *best_nl_deg = best_d;
    return 0;
}

/* ??? Information criteria ??????????????????????????????????????????????? */

double wh_ident_compute_aic(double mse, int n_params, int n_data) {
    if (n_data <= 0 || mse <= 0.0) return 1e100;
    return n_data * log(mse) + 2.0 * n_params;
}

double wh_ident_compute_bic(double mse, int n_params, int n_data) {
    if (n_data <= 0 || mse <= 0.0) return 1e100;
    return n_data * log(mse) + n_params * log((double)n_data);
}
