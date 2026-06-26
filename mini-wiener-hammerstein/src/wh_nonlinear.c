/**
 * wh_nonlinear.c ? Static Nonlinearity Implementations
 *
 * Implements polynomial, spline, saturation, dead-zone, sigmoid, tanh,
 * Gaussian RBF, piecewise linear, and lookup table nonlinearities.
 * Each nonlinearity maps x ? f(x) with optional derivative computation.
 *
 * Knowledge Level: L3 (Mathematical Structures), L5 (Algorithms)
 */

#include "wh_nonlinear.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/* ??? Initialization functions ??????????????????????????????????????????? */

int wh_nl_init_polynomial(WH_Nonlinearity* nl,
                           const double* coeffs, int degree) {
    if (!nl || !coeffs || degree < 0 || degree + 1 > 32) return -1;
    memset(nl, 0, sizeof(WH_Nonlinearity));
    nl->type = WH_NL_POLYNOMIAL;
    nl->n_params = degree + 1;
    for (int i = 0; i <= degree; i++) {
        nl->params[i] = coeffs[i];
    }
    return 0;
}

int wh_nl_init_spline(WH_Nonlinearity* nl,
                       const double* knots, const double* values,
                       int n_knots) {
    if (!nl || !knots || !values || n_knots < 2 || n_knots > 32) return -1;
    /* Verify monotonic knots */
    for (int i = 1; i < n_knots; i++) {
        if (knots[i] <= knots[i - 1]) return -1;
    }
    memset(nl, 0, sizeof(WH_Nonlinearity));
    nl->type = WH_NL_SPLINE;
    nl->n_knots = n_knots;
    for (int i = 0; i < n_knots; i++) {
        nl->knots[i] = knots[i];
        nl->params[i] = values[i]; /* Store values in params temporarily */
    }

    /* Construct natural cubic spline coefficients.
     * For each segment i (between knot i and i+1):
     *   S_i(x) = a_i + b_i*(x-x_i) + c_i*(x-x_i)^2 + d_i*(x-x_i)^3
     *   where a_i = y_i
     *   b_i = (y_{i+1}-y_i)/h_i - h_i*(2*c_i + c_{i+1})/3
     *   d_i = (c_{i+1} - c_i) / (3*h_i)
     *
     * Solve tridiagonal system for c_i (second derivatives at knots).
     * Natural boundary: c_0 = c_{n-1} = 0.
     */
    int m = n_knots - 1; /* Number of segments */
    double h[32], alpha[32];
    for (int i = 0; i < m; i++) {
        h[i] = knots[i + 1] - knots[i];
    }
    for (int i = 1; i < m; i++) {
        alpha[i] = (3.0 / h[i]) * (values[i + 1] - values[i])
                 - (3.0 / h[i - 1]) * (values[i] - values[i - 1]);
    }

    /* Tridiagonal solver for c[] */
    double l[32], mu[32], z[32], c[32];
    for (int i = 0; i < 32; i++) c[i] = 0.0;
    l[0] = 1.0;
    mu[0] = 0.0;
    z[0] = 0.0;
    for (int i = 1; i < m; i++) {
        l[i] = 2.0 * (knots[i + 1] - knots[i - 1]) - h[i - 1] * mu[i - 1];
        mu[i] = h[i] / l[i];
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
    }
    l[m] = 1.0;
    z[m] = 0.0;
    c[m] = 0.0;
    for (int j = m - 1; j >= 0; j--) {
        c[j] = z[j] - mu[j] * c[j + 1];
    }

    /* Store segment coefficients: [a, b, c, d] for each segment */
    for (int i = 0; i < m; i++) {
        double a_i = values[i];
        double c_i = c[i];
        double c_ip1 = c[i + 1];
        double b_i = (values[i + 1] - values[i]) / h[i]
                     - h[i] * (2.0 * c_i + c_ip1) / 3.0;
        double d_i = (c_ip1 - c_i) / (3.0 * h[i]);

        nl->spline_coeffs[4 * i + 0] = a_i;
        nl->spline_coeffs[4 * i + 1] = b_i;
        nl->spline_coeffs[4 * i + 2] = c_i;
        nl->spline_coeffs[4 * i + 3] = d_i;
    }

    return 0;
}

int wh_nl_init_saturation(WH_Nonlinearity* nl, double K, double L) {
    if (!nl || L < 0.0) return -1;
    memset(nl, 0, sizeof(WH_Nonlinearity));
    nl->type = WH_NL_SATURATION;
    nl->n_params = 2;
    nl->params[0] = K;
    nl->params[1] = L;
    return 0;
}

int wh_nl_init_deadzone(WH_Nonlinearity* nl, double dz) {
    if (!nl || dz < 0.0) return -1;
    memset(nl, 0, sizeof(WH_Nonlinearity));
    nl->type = WH_NL_DEADZONE;
    nl->n_params = 1;
    nl->params[0] = dz;
    return 0;
}

int wh_nl_init_sigmoid(WH_Nonlinearity* nl, double a, double b, double c) {
    if (!nl) return -1;
    memset(nl, 0, sizeof(WH_Nonlinearity));
    nl->type = WH_NL_SIGMOID;
    nl->n_params = 3;
    nl->params[0] = a;
    nl->params[1] = b;
    nl->params[2] = c;
    return 0;
}

int wh_nl_init_tanh(WH_Nonlinearity* nl, double a, double b, double c) {
    if (!nl) return -1;
    memset(nl, 0, sizeof(WH_Nonlinearity));
    nl->type = WH_NL_TANH;
    nl->n_params = 3;
    nl->params[0] = a;
    nl->params[1] = b;
    nl->params[2] = c;
    return 0;
}

int wh_nl_init_rbf(WH_Nonlinearity* nl,
                    const double* centers, const double* widths,
                    const double* weights, int n) {
    if (!nl || !centers || !widths || !weights || n <= 0 || n > 32) return -1;
    memset(nl, 0, sizeof(WH_Nonlinearity));
    nl->type = WH_NL_GAUSSIAN_RBF;
    nl->n_centers = n;
    nl->n_params = 3 * n;
    for (int i = 0; i < n; i++) {
        nl->centers[i] = centers[i];
        nl->rbf_widths[i] = widths[i];
        nl->rbf_weights[i] = weights[i];
    }
    return 0;
}

int wh_nl_init_lookup(WH_Nonlinearity* nl,
                       const double* x_vals, const double* y_vals, int n) {
    if (!nl || !x_vals || !y_vals || n < 2 || n > 32) return -1;
    memset(nl, 0, sizeof(WH_Nonlinearity));
    nl->type = WH_NL_LOOKUP_TABLE;
    nl->lut_size = n;
    for (int i = 0; i < n; i++) {
        nl->lut_x[i] = x_vals[i];
        nl->lut_y[i] = y_vals[i];
    }
    return 0;
}

/* ??? Evaluation ????????????????????????????????????????????????????????? */

double wh_nl_evaluate(const WH_Nonlinearity* nl, double x) {
    if (!nl) return x;

    switch (nl->type) {
        case WH_NL_POLYNOMIAL: {
            double y = 0.0, xpow = 1.0;
            for (int i = 0; i < nl->n_params; i++) {
                y += nl->params[i] * xpow;
                xpow *= x;
            }
            return y;
        }
        case WH_NL_SPLINE: {
            /* Find segment */
            int k = 0;
            if (x <= nl->knots[0]) k = 0;
            else if (x >= nl->knots[nl->n_knots - 1]) k = nl->n_knots - 2;
            else {
                for (int i = 0; i < nl->n_knots - 1; i++) {
                    if (x >= nl->knots[i] && x <= nl->knots[i + 1]) {
                        k = i;
                        break;
                    }
                }
            }
            double dx = x - nl->knots[k];
            double* c = (double*)&nl->spline_coeffs[4 * k];
            return c[0] + c[1] * dx + c[2] * dx * dx + c[3] * dx * dx * dx;
        }
        case WH_NL_SATURATION: {
            double K = nl->params[0], L = nl->params[1];
            if (x > L) return K * L;
            if (x < -L) return -K * L;
            return K * x;
        }
        case WH_NL_DEADZONE: {
            double dz = nl->params[0];
            if (fabs(x) <= dz) return 0.0;
            return (x > 0.0 ? 1.0 : -1.0) * (fabs(x) - dz);
        }
        case WH_NL_SIGMOID: {
            double a = nl->params[0], b = nl->params[1], c = nl->params[2];
            double exp_arg = -b * (x - c);
            // Prevent overflow
            if (exp_arg > 100.0) return a;
            if (exp_arg < -100.0) return 0.0;
            return a / (1.0 + exp(exp_arg));
        }
        case WH_NL_TANH: {
            double a = nl->params[0], b = nl->params[1], c = nl->params[2];
            return a * tanh(b * (x - c));
        }
        case WH_NL_GAUSSIAN_RBF: {
            double y = 0.0;
            for (int i = 0; i < nl->n_centers; i++) {
                double dx = x - nl->centers[i];
                double sigma2 = nl->rbf_widths[i] * nl->rbf_widths[i];
                if (sigma2 < 1e-12) continue;
                y += nl->rbf_weights[i] * exp(-0.5 * dx * dx / sigma2);
            }
            return y;
        }
        case WH_NL_PIECEWISE_LINEAR: {
            if (x <= nl->lut_x[0]) return nl->lut_y[0];
            if (x >= nl->lut_x[nl->lut_size - 1]) return nl->lut_y[nl->lut_size - 1];
            int idx = 0;
            for (int i = 1; i < nl->lut_size; i++) {
                if (x >= nl->lut_x[i]) idx = i;
            }
            if (idx >= nl->lut_size - 1) idx = nl->lut_size - 2;
            double alpha = (x - nl->lut_x[idx]) /
                           (nl->lut_x[idx + 1] - nl->lut_x[idx]);
            return nl->lut_y[idx] + alpha * (nl->lut_y[idx + 1] - nl->lut_y[idx]);
        }
        case WH_NL_LOOKUP_TABLE: {
            /* Binary search */
            if (x <= nl->lut_x[0]) return nl->lut_y[0];
            if (x >= nl->lut_x[nl->lut_size - 1]) return nl->lut_y[nl->lut_size - 1];
            int lo = 0, hi = nl->lut_size - 1;
            while (hi - lo > 1) {
                int mid = (lo + hi) / 2;
                if (x < nl->lut_x[mid]) hi = mid;
                else lo = mid;
            }
            double alpha = (x - nl->lut_x[lo]) / (nl->lut_x[hi] - nl->lut_x[lo]);
            return nl->lut_y[lo] + alpha * (nl->lut_y[hi] - nl->lut_y[lo]);
        }
        default:
            return x;
    }
}

void wh_nl_evaluate_batch(const WH_Nonlinearity* nl,
                           const double* x, double* y, int n) {
    if (!nl || !x || !y) return;
    for (int i = 0; i < n; i++) {
        y[i] = wh_nl_evaluate(nl, x[i]);
    }
}

/* ??? Derivative ????????????????????????????????????????????????????????? */

double wh_nl_derivative(const WH_Nonlinearity* nl, double x) {
    if (!nl) return 1.0;

    switch (nl->type) {
        case WH_NL_POLYNOMIAL: {
            if (nl->n_params < 2) return 0.0;
            double dy = 0.0, xpow = 1.0;
            for (int i = 1; i < nl->n_params; i++) {
                dy += i * nl->params[i] * xpow;
                xpow *= x;
            }
            return dy;
        }
        case WH_NL_SPLINE: {
            int k = 0;
            if (x <= nl->knots[0]) k = 0;
            else if (x >= nl->knots[nl->n_knots - 1]) k = nl->n_knots - 2;
            else {
                for (int i = 0; i < nl->n_knots - 1; i++) {
                    if (x >= nl->knots[i] && x <= nl->knots[i + 1]) {
                        k = i; break;
                    }
                }
            }
            double dx = x - nl->knots[k];
            double* c = (double*)&nl->spline_coeffs[4 * k];
            return c[1] + 2.0 * c[2] * dx + 3.0 * c[3] * dx * dx;
        }
        case WH_NL_SATURATION: {
            double L = nl->params[1], K = nl->params[0];
            if (fabs(x) >= L) return 0.0;
            return K;
        }
        case WH_NL_DEADZONE: {
            double dz = nl->params[0];
            if (fabs(x) <= dz) return 0.0;
            return 1.0;
        }
        case WH_NL_SIGMOID: {
            double a = nl->params[0], b = nl->params[1], c = nl->params[2];
            double exp_arg = -b * (x - c);
            if (exp_arg > 100.0 || exp_arg < -100.0) return 0.0;
            double sig = 1.0 / (1.0 + exp(exp_arg));
            return a * b * sig * (1.0 - sig);
        }
        case WH_NL_TANH: {
            double a = nl->params[0], b = nl->params[1], c = nl->params[2];
            double t = tanh(b * (x - c));
            return a * b * (1.0 - t * t);
        }
        case WH_NL_GAUSSIAN_RBF: {
            double dy = 0.0;
            for (int i = 0; i < nl->n_centers; i++) {
                double dx = x - nl->centers[i];
                double sigma2 = nl->rbf_widths[i] * nl->rbf_widths[i];
                if (sigma2 < 1e-12) continue;
                double g = exp(-0.5 * dx * dx / sigma2);
                dy -= nl->rbf_weights[i] * dx * g / sigma2;
            }
            return dy;
        }
        case WH_NL_PIECEWISE_LINEAR: {
            if (x <= nl->lut_x[0] || x >= nl->lut_x[nl->lut_size - 1]) return 0.0;
            int idx = 0;
            for (int i = 1; i < nl->lut_size; i++) {
                if (x >= nl->lut_x[i]) idx = i;
            }
            if (idx >= nl->lut_size - 1) idx = nl->lut_size - 2;
            return (nl->lut_y[idx + 1] - nl->lut_y[idx]) /
                   (nl->lut_x[idx + 1] - nl->lut_x[idx]);
        }
        case WH_NL_LOOKUP_TABLE: {
            if (x <= nl->lut_x[0] || x >= nl->lut_x[nl->lut_size - 1]) return 0.0;
            int lo = 0, hi = nl->lut_size - 1;
            while (hi - lo > 1) {
                int mid = (lo + hi) / 2;
                if (x < nl->lut_x[mid]) hi = mid;
                else lo = mid;
            }
            return (nl->lut_y[hi] - nl->lut_y[lo]) /
                   (nl->lut_x[hi] - nl->lut_x[lo]);
        }
        default:
            return 1.0;
    }
}

/* ??? Range computation ?????????????????????????????????????????????????? */

void wh_nl_get_range(const WH_Nonlinearity* nl,
                      double x_min, double x_max,
                      double* f_min, double* f_max) {
    if (!nl || !f_min || !f_max || x_min > x_max) return;
    /* Sample at 200 points and at potential extrema */
    int n_samples = 200;
    double f = wh_nl_evaluate(nl, x_min);
    *f_min = f;
    *f_max = f;
    for (int i = 1; i <= n_samples; i++) {
        double x = x_min + (x_max - x_min) * i / (double)n_samples;
        f = wh_nl_evaluate(nl, x);
        if (f < *f_min) *f_min = f;
        if (f > *f_max) *f_max = f;
    }
}

/* ??? Monotonicity check ????????????????????????????????????????????????? */

int wh_nl_is_monotonic(const WH_Nonlinearity* nl) {
    if (!nl) return 1; /* Identity is monotonic */
    /* Sample at many points, check derivative sign */
    const double domain = 10.0;
    int n_checks = 500;
    int sign = 0;
    for (int i = 0; i < n_checks; i++) {
        double x = -domain + 2.0 * domain * i / (double)(n_checks - 1);
        double dx = wh_nl_derivative(nl, x);
        if (dx > 1e-8) {
            if (sign == 0) sign = 1;
            else if (sign < 0) return 0;
        } else if (dx < -1e-8) {
            if (sign == 0) sign = -1;
            else if (sign > 0) return 0;
        }
    }
    return 1;
}

/* ??? Odd symmetry check ????????????????????????????????????????????????? */

int wh_nl_is_odd(const WH_Nonlinearity* nl) {
    if (!nl) return 1;
    int n_checks = 100;
    double tol = 1e-6;
    for (int i = 0; i < n_checks; i++) {
        double x = 0.1 + 9.9 * i / (double)(n_checks - 1);
        double f1 = wh_nl_evaluate(nl, x);
        double f2 = wh_nl_evaluate(nl, -x);
        if (fabs(f1 + f2) > tol * (1.0 + fabs(f1))) return 0;
    }
    return 1;
}

/* ??? Root finding ??????????????????????????????????????????????????????? */

double wh_nl_find_root(const WH_Nonlinearity* nl, double target,
                        double a, double b, double tol) {
    if (!nl || a >= b) return NAN;
    double fa = wh_nl_evaluate(nl, a) - target;
    double fb = wh_nl_evaluate(nl, b) - target;
    if (fa * fb > 0.0) return NAN; /* No sign change ? no root guaranteed */
    if (fabs(fa) < tol) return a;
    if (fabs(fb) < tol) return b;
    for (int iter = 0; iter < 100; iter++) {
        double c = (a + b) / 2.0;
        double fc = wh_nl_evaluate(nl, c) - target;
        if (fabs(fc) < tol || fabs(b - a) < tol) return c;
        if (fa * fc < 0.0) { b = c; fb = fc; }
        else { a = c; fa = fc; }
    }
    return (a + b) / 2.0;
}

/* ??? Printing ??????????????????????????????????????????????????????????? */

void wh_nl_print(const WH_Nonlinearity* nl) {
    if (!nl) { printf("Nonlinearity: NULL\n"); return; }
    printf("Nonlinearity [%s], n_params=%d\n",
           nl->type == WH_NL_POLYNOMIAL ? "POLY" :
           nl->type == WH_NL_SPLINE ? "SPLINE" :
           nl->type == WH_NL_SATURATION ? "SAT" :
           nl->type == WH_NL_DEADZONE ? "DZ" :
           nl->type == WH_NL_SIGMOID ? "SIGMOID" :
           nl->type == WH_NL_TANH ? "TANH" :
           nl->type == WH_NL_GAUSSIAN_RBF ? "RBF" :
           nl->type == WH_NL_PIECEWISE_LINEAR ? "PWL" :
           nl->type == WH_NL_LOOKUP_TABLE ? "LUT" : "?", nl->n_params);
    printf("  Monotonic: %s, Odd: %s\n",
           wh_nl_is_monotonic(nl) ? "Yes" : "No",
           wh_nl_is_odd(nl) ? "Yes" : "No");
    if (nl->n_params > 0) {
        printf("  Params: [");
        for (int i = 0; i < nl->n_params && i < 8; i++)
            printf("%.3f%s", nl->params[i], i < nl->n_params - 1 ? ", " : "");
        if (nl->n_params > 8) printf("...");
        printf("]\n");
    }
}

void wh_nl_copy(WH_Nonlinearity* dest, const WH_Nonlinearity* src) {
    if (!dest || !src) return;
    memcpy(dest, src, sizeof(WH_Nonlinearity));
}
