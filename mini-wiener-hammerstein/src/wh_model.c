/**
 * wh_model.c ? Wiener-Hammerstein Model Core Implementation
 *
 * Implements core WH model operations: creation, evaluation, simulation,
 * stability checking, and parameter counting.
 *
 * Knowledge Level: L1 (Definitions), L3 (Mathematical Structures)
 */

#include "wh_model.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>



/* ??? Model creation and destruction ????????????????????????????????????? */

WH_Model* wh_model_create(void) {
    WH_Model* m = (WH_Model*)calloc(1, sizeof(WH_Model));
    if (!m) return NULL;

    /* L1: identity FIR (b[0]=1, a[0]=1) */
    m->L1.type = WH_LIN_FIR;
    m->L1.nb = 1;
    m->L1.na = 0;
    m->L1.b[0] = 1.0;
    m->L1.order = 0;
    m->L1.D = 1.0;
    m->L1.Ts = 1.0;
    m->L1.state_dim = 0;

    /* Nonlinearity: linear (f(x)=x, polynomial degree 1) */
    m->N.type = WH_NL_POLYNOMIAL;
    m->N.n_params = 2;
    m->N.params[0] = 0.0;  /* c0 = 0 */
    m->N.params[1] = 1.0;  /* c1 = 1 */

    /* L2: identity FIR */
    m->L2.type = WH_LIN_FIR;
    m->L2.nb = 1;
    m->L2.na = 0;
    m->L2.b[0] = 1.0;
    m->L2.order = 0;
    m->L2.D = 1.0;
    m->L2.Ts = 1.0;
    m->L2.state_dim = 0;

    /* Noise model: OE (H(q)=1) */
    m->noise.order_C = 0;
    m->noise.order_D = 0;
    m->noise.D[0] = 1.0;
    m->noise.C[0] = 1.0;
    m->noise.noise_variance = 0.0;

    /* Metadata */
    m->method = WH_ID_ITERATIVE;
    m->status = WH_STATUS_OK;
    m->is_identified = 0;
    m->fit_percent = 0.0;
    m->mse = 0.0;
    m->aic = 0.0;
    m->bic = 0.0;
    m->n_params_total = 2;  /* L1.nb + N.n_params - redundant structures */
    m->n_data_used = 0;

    return m;
}

void wh_model_free(WH_Model* model) {
    free(model);
}

WH_Model* wh_model_copy(const WH_Model* src) {
    if (!src) return NULL;
    WH_Model* dst = (WH_Model*)malloc(sizeof(WH_Model));
    if (!dst) return NULL;
    memcpy(dst, src, sizeof(WH_Model));
    return dst;
}

/* ??? Model evaluation ??????????????????????????????????????????????????? */

double wh_model_evaluate(WH_Model* model, double u) {
    if (!model) return 0.0;

    /* L1: Filter input through first linear block */
    double x = 0.0;

    if (model->L1.type == WH_LIN_FIR) {
        /* Shift state buffer right */
        for (int i = model->L1.nb - 1; i > 0; i--) {
            model->L1_state[i] = model->L1_state[i - 1];
        }
        model->L1_state[0] = u;

        /* FIR convolution */
        x = 0.0;
        for (int i = 0; i < model->L1.nb; i++) {
            x += model->L1.b[i] * model->L1_state[i];
        }
    } else if (model->L1.type == WH_LIN_IIR_TF) {
        /* Shift input and output buffers */
        for (int i = model->L1.nb - 1; i > 0; i--) {
            model->L1_state[i] = model->L1_state[i - 1];
        }
        model->L1_state[0] = u;

        /* IIR: y[k] = ? b_i?u[k-i] - ? a_i?y[k-i] */
        x = 0.0;
        for (int i = 0; i < model->L1.nb; i++) {
            x += model->L1.b[i] * model->L1_state[i];
        }
        for (int i = 1; i < model->L1.na; i++) {
            x -= model->L1.a[i] * model->L1_state[model->L1.nb + i - 1];
        }
        /* Shift output state */
        for (int i = model->L1.na - 1; i > 0; i--) {
            model->L1_state[model->L1.nb + i] =
                model->L1_state[model->L1.nb + i - 1];
        }
        model->L1_state[model->L1.nb] = x;
    } else {
        /* State-space: x[k+1] = A*x[k] + B*u, y = C*x + D*u */
        int n = model->L1.order;
        double new_state[64];
        for (int i = 0; i < n; i++) {
            new_state[i] = 0.0;
            for (int j = 0; j < n; j++) {
                new_state[i] += model->L1.A[i * n + j] * model->L1_state[j];
            }
            new_state[i] += model->L1.B[i] * u;
        }
        x = model->L1.D * u;
        for (int i = 0; i < n; i++) {
            x += model->L1.C[i] * model->L1_state[i];
        }
        memcpy(model->L1_state, new_state, n * sizeof(double));
    }
    model->x_current = x;

    /* N: Static nonlinearity */
    double w = 0.0;
    const WH_Nonlinearity* nl = &model->N;
    switch (nl->type) {
        case WH_NL_POLYNOMIAL: {
            double xpow = 1.0;
            for (int i = 0; i < nl->n_params; i++) {
                w += nl->params[i] * xpow;
                xpow *= x;
            }
            break;
        }
        case WH_NL_SATURATION: {
            double K = nl->params[0];
            double L = nl->params[1];
            if (x > L) w = K * L;
            else if (x < -L) w = -K * L;
            else w = K * x;
            break;
        }
        case WH_NL_DEADZONE: {
            double dz = nl->params[0];
            if (fabs(x) <= dz) w = 0.0;
            else w = (x > 0 ? 1.0 : -1.0) * (fabs(x) - dz);
            break;
        }
        case WH_NL_SIGMOID: {
            double a = nl->params[0], b = nl->params[1], c = nl->params[2];
            w = a / (1.0 + exp(-b * (x - c)));
            break;
        }
        case WH_NL_TANH: {
            double a = nl->params[0], b = nl->params[1], c = nl->params[2];
            w = a * tanh(b * (x - c));
            break;
        }
        case WH_NL_GAUSSIAN_RBF: {
            for (int i = 0; i < nl->n_centers; i++) {
                double dx = x - nl->centers[i];
                double sigma2 = nl->rbf_widths[i] * nl->rbf_widths[i];
                w += nl->rbf_weights[i] * exp(-0.5 * dx * dx / sigma2);
            }
            break;
        }
        case WH_NL_SPLINE: {
            /* Find the interval containing x */
            int k = 0;
            for (int i = 0; i < nl->n_knots - 1; i++) {
                if (x >= nl->knots[i] && x <= nl->knots[i + 1]) {
                    k = i;
                    break;
                }
            }
            if (x < nl->knots[0]) k = 0;
            if (x > nl->knots[nl->n_knots - 1]) k = nl->n_knots - 2;
            double dx = x - nl->knots[k];
            const double* c = &nl->spline_coeffs[4 * k];
            w = c[0] + c[1] * dx + c[2] * dx * dx + c[3] * dx * dx * dx;
            break;
        }
        case WH_NL_PIECEWISE_LINEAR: {
            int idx = 0;
            for (int i = 0; i < nl->lut_size - 1; i++) {
                if (x >= nl->lut_x[i]) idx = i;
            }
            if (idx >= nl->lut_size - 1) idx = nl->lut_size - 2;
            double alpha = (x - nl->lut_x[idx]) /
                           (nl->lut_x[idx + 1] - nl->lut_x[idx]);
            if (alpha < 0.0) alpha = 0.0;
            if (alpha > 1.0) alpha = 1.0;
            w = nl->lut_y[idx] + alpha * (nl->lut_y[idx + 1] - nl->lut_y[idx]);
            break;
        }
        case WH_NL_LOOKUP_TABLE: {
            /* Binary search for interval */
            int lo = 0, hi = nl->lut_size - 1;
            while (hi - lo > 1) {
                int mid = (lo + hi) / 2;
                if (x < nl->lut_x[mid]) hi = mid;
                else lo = mid;
            }
            double alpha = (x - nl->lut_x[lo]) /
                           (nl->lut_x[hi] - nl->lut_x[lo]);
            if (alpha < 0.0) alpha = 0.0;
            if (alpha > 1.0) alpha = 1.0;
            w = nl->lut_y[lo] + alpha * (nl->lut_y[hi] - nl->lut_y[lo]);
            break;
        }
        default:
            w = x;
            break;
    }
    model->w_current = w;

    /* L2: Filter nonlinearity output through second linear block */
    double y = 0.0;

    if (model->L2.type == WH_LIN_FIR) {
        for (int i = model->L2.nb - 1; i > 0; i--) {
            model->L2_state[i] = model->L2_state[i - 1];
        }
        model->L2_state[0] = w;
        y = 0.0;
        for (int i = 0; i < model->L2.nb; i++) {
            y += model->L2.b[i] * model->L2_state[i];
        }
    } else if (model->L2.type == WH_LIN_IIR_TF) {
        for (int i = model->L2.nb - 1; i > 0; i--) {
            model->L2_state[i] = model->L2_state[i - 1];
        }
        model->L2_state[0] = w;
        y = 0.0;
        for (int i = 0; i < model->L2.nb; i++) {
            y += model->L2.b[i] * model->L2_state[i];
        }
        for (int i = 1; i < model->L2.na; i++) {
            y -= model->L2.a[i] * model->L2_state[model->L2.nb + i - 1];
        }
        for (int i = model->L2.na - 1; i > 0; i--) {
            model->L2_state[model->L2.nb + i] =
                model->L2_state[model->L2.nb + i - 1];
        }
        model->L2_state[model->L2.nb] = y;
    } else {
        int n = model->L2.order;
        double new_state[64];
        for (int i = 0; i < n; i++) {
            new_state[i] = 0.0;
            for (int j = 0; j < n; j++) {
                new_state[i] += model->L2.A[i * n + j] * model->L2_state[j];
            }
            new_state[i] += model->L2.B[i] * w;
        }
        y = model->L2.D * w;
        for (int i = 0; i < n; i++) {
            y += model->L2.C[i] * model->L2_state[i];
        }
        memcpy(model->L2_state, new_state, n * sizeof(double));
    }

    return y;
}

/* ??? Batch simulation ??????????????????????????????????????????????????? */

int wh_model_simulate(WH_Model* model,
                       const double* u, double* y, int n_samples) {
    if (!model || !u || !y || n_samples <= 0) return WH_STATUS_PARAM_VIOLATION;
    /* Reset internal states before simulation */
    wh_model_reset(model);
    for (int i = 0; i < n_samples; i++) {
        y[i] = wh_model_evaluate(model, u[i]);
    }
    return WH_STATUS_OK;
}

/* ??? Model state management ????????????????????????????????????????????? */

void wh_model_reset(WH_Model* model) {
    if (!model) return;
    memset(model->L1_state, 0, sizeof(model->L1_state));
    memset(model->L2_state, 0, sizeof(model->L2_state));
    model->x_current = 0.0;
    model->w_current = 0.0;
}

/* ??? Delay computation ?????????????????????????????????????????????????? */

int wh_model_get_delay(const WH_Model* model) {
    if (!model) return 0;
    int d1 = 0, d2 = 0;

    /* Find index of first non-zero coefficient in L1 */
    for (int i = 0; i < model->L1.nb; i++) {
        if (fabs(model->L1.b[i]) > 1e-12) {
            d1 = i;
            break;
        }
    }

    /* Find index of first non-zero coefficient in L2 */
    for (int i = 0; i < model->L2.nb; i++) {
        if (fabs(model->L2.b[i]) > 1e-12) {
            d2 = i;
            break;
        }
    }

    return d1 + d2;
}

/* ??? Stability check ???????????????????????????????????????????????????? */

/**
 * wh_model_is_stable ? Check BIBO stability.
 *
 * For discrete-time LTI systems, BIBO stability is equivalent to all poles
 * being strictly inside the unit circle (|p_i| < 1).
 *
 * We compute poles using the companion matrix eigenvalue method for the
 * denominator polynomial A(z) = 1 + a_1*z^{-1} + ... + a_{na-1}*z^{-(na-1)}.
 *
 * For degree-1 and degree-2 polynomials, we use analytical formulas.
 * For higher degrees, we use a simple implementation of the iterative
 * companion matrix power method.
 */
static int has_pole_outside_unit_circle(const double* a, int na) {
    if (na <= 0) return 0; /* No denominator ? FIR, always stable */
    if (na == 1) {
        double p = -a[1] / a[0]; /* Root of a0 + a1*z^{-1} = 0 ? z = -a1/a0 */
        return fabs(p) >= 1.0;
    }
    if (na == 2) {
        /* a0 + a1*z^{-1} + a2*z^{-2} = 0 ? a0*z^2 + a1*z + a2 = 0 */
        double disc = a[1] * a[1] - 4.0 * a[0] * a[2];
        if (disc >= 0) {
            double p1 = (-a[1] + sqrt(disc)) / (2.0 * a[0]);
            double p2 = (-a[1] - sqrt(disc)) / (2.0 * a[0]);
            return (fabs(p1) >= 1.0 || fabs(p2) >= 1.0);
        } else {
            /* Complex conjugate: |p| = sqrt(re? + im?) = sqrt(a2/a0) */
            return (fabs(a[2] / a[0]) >= 1.0);
        }
    }
    /* For higher order, use Jury stability test (simplified) */
    /* Check necessary condition: |a_{na}| < |a_0| */
    if (fabs(a[na - 1]) >= fabs(a[0])) return 1;

    /* Apply Jury's test recursively */
    double a_copy[64];
    for (int i = 0; i < na; i++) {
        a_copy[i] = a[na - 1 - i]; /* Reverse for z-domain: A(z) coefficients */
    }

    /* Simplified Jury criterion: check that A(1) > 0, (-1)^n * A(-1) > 0 */
    double A_1 = 0.0;
    for (int i = 0; i < na; i++) A_1 += a_copy[i];
    if (A_1 <= 0.0) return 1;

    double A_m1 = 0.0;
    for (int i = 0; i < na; i++) {
        A_m1 += (i % 2 == 0 ? a_copy[i] : -a_copy[i]);
    }
    /* na is the degree: use actual polynomial degree for sign check */
    int poly_degree = na - 1;  /* a[0] is leading, a[na-1] is constant */
    double sign = (poly_degree % 2 == 0) ? 1.0 : -1.0;
    if (sign * A_m1 <= 0.0) return 1;

    return 0; /* Jury conditions satisfied ? stable */
}

int wh_model_is_stable(const WH_Model* model) {
    if (!model) return 0;

    /* FIR is always stable */
    if (model->L1.type == WH_LIN_FIR && model->L2.type == WH_LIN_FIR) {
        return 1;
    }

    /* Check L1 */
    if (model->L1.type == WH_LIN_IIR_TF && model->L1.na > 0) {
        if (has_pole_outside_unit_circle(model->L1.a, model->L1.na + 1)) {
            return 0;
        }
    }

    /* Check L2 */
    if (model->L2.type == WH_LIN_IIR_TF && model->L2.na > 0) {
        if (has_pole_outside_unit_circle(model->L2.a, model->L2.na + 1)) {
            return 0;
        }
    }

    return 1;
}

/* ??? Printing ??????????????????????????????????????????????????????????? */

void wh_model_print(const WH_Model* model) {
    if (!model) {
        printf("WH_Model: NULL\n");
        return;
    }
    printf("??? Wiener-Hammerstein Model ????????????????????????????????\n");
    printf("?  L1 type: %s, order: %d, nb=%d, na=%d\n",
           wh_model_get_lin_type_name(model->L1.type),
           model->L1.order, model->L1.nb, model->L1.na);
    printf("?  N  type: %s, n_params=%d\n",
           wh_model_get_nl_type_name(model->N.type), model->N.n_params);
    printf("?  L2 type: %s, order: %d, nb=%d, na=%d\n",
           wh_model_get_lin_type_name(model->L2.type),
           model->L2.order, model->L2.nb, model->L2.na);
    printf("?  Status: %d, Identified: %s\n",
           model->status, model->is_identified ? "Yes" : "No");
    printf("?  FIT: %.2f%%, MSE: %.6f\n",
           model->fit_percent, model->mse);
    printf("?  AIC: %.3f, BIC: %.3f\n", model->aic, model->bic);
    printf("?  Total delay: %d samples, Stable: %s\n",
           wh_model_get_delay(model),
           wh_model_is_stable(model) ? "Yes" : "No");
    printf("?  N params: %d, N data: %d\n",
           wh_model_count_parameters(model), model->n_data_used);
    printf("????????????????????????????????????????????????????????????????\n");
}

const char* wh_model_get_nl_type_name(wh_nl_type_t type) {
    static const char* names[] = {
        "POLYNOMIAL", "SPLINE", "SATURATION", "DEADZONE",
        "PIECEWISE_LINEAR", "SIGMOID", "TANH", "GAUSSIAN_RBF",
        "LOOKUP_TABLE"
    };
    if (type < 0 || type >= WH_NL_COUNT) return "UNKNOWN";
    return names[type];
}

const char* wh_model_get_lin_type_name(wh_lin_type_t type) {
    static const char* names[] = {
        "FIR", "IIR_TF", "STATE_SPACE"
    };
    if (type < 0 || type >= WH_LIN_COUNT) return "UNKNOWN";
    return names[type];
}

/* ??? Parameter counting ????????????????????????????????????????????????? */

int wh_model_count_parameters(const WH_Model* model) {
    if (!model) return 0;
    int count = 0;

    /* L1 parameters: b coefficients + a coefficients (excluding a[0]=1) */
    count += model->L1.nb;
    count += (model->L1.na > 0 ? model->L1.na - 1 : 0);

    /* N parameters: depends on type */
    switch (model->N.type) {
        case WH_NL_POLYNOMIAL:
            count += model->N.n_params;
            break;
        case WH_NL_SPLINE:
            count += model->N.n_knots * 4; /* 4 coefficients per segment */
            break;
        case WH_NL_SATURATION:
            count += 2; /* K, L */
            break;
        case WH_NL_DEADZONE:
            count += 1; /* dz */
            break;
        case WH_NL_SIGMOID: case WH_NL_TANH:
            count += 3; /* a, b, c */
            break;
        case WH_NL_GAUSSIAN_RBF:
            count += 3 * model->N.n_centers; /* 3 params per center */
            break;
        case WH_NL_PIECEWISE_LINEAR: case WH_NL_LOOKUP_TABLE:
            count += 2 * model->N.lut_size; /* x and y breakpoints */
            break;
        default:
            break;
    }

    /* L2 parameters */
    count += model->L2.nb;
    count += (model->L2.na > 0 ? model->L2.na - 1 : 0);

    /* Noise model parameters */
    count += model->noise.order_C;
    count += model->noise.order_D;
    count += 1; /* noise variance */

    return count;
}
