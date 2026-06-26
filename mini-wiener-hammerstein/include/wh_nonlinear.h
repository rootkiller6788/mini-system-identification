/**
 * wh_nonlinear.h ? Static Nonlinearity Models
 *
 * Implements various static nonlinear functions for the nonlinear block N
 * in the Wiener-Hammerstein model: x(t) ? [ N ] ? w(t) = f(x(t)).
 *
 * Supported nonlinearities:
 *   Polynomial:  f(x) = c0 + c1*x + c2*x^2 + ... + cd*x^d
 *   Spline:      f(x) = piecewise cubic with C? continuity at knots
 *   Saturation:  f(x) = K * sat(x/L) with smooth or hard clipping
 *   Dead-zone:   f(x) = max(0, |x|-dz) * sign(x)
 *   Sigmoid:     f(x) = a / (1 + exp(-b*(x-c)))
 *   Tanh:        f(x) = a * tanh(b*(x-c))
 *   RBF Network: f(x) = ? w_i * exp(-(x-c_i)?/(2?_i?))
 *   Lookup:      Linear interpolation between breakpoints
 *
 * The key theoretical insight: any continuous static nonlinearity on a
 * compact domain can be uniformly approximated by polynomials (Weierstrass
 * approximation theorem), splines, or RBF networks.
 *
 * References:
 *   - Nelles, O. (2001). Nonlinear System Identification. Springer.
 *   - de Boor, C. (1978). A Practical Guide to Splines. Springer.
 *
 * Knowledge Level: L3 (Mathematical Structures), L5 (Algorithms)
 */

#ifndef WH_NONLINEAR_H
#define WH_NONLINEAR_H

#include "wh_model.h"

/* ??? Nonlinearity life cycle ???????????????????????????????????????????? */

/**
 * wh_nl_init_polynomial ? Initialize as polynomial nonlinearity.
 *
 * f(x) = ?_{i=0}^{degree} coeffs[i] * x^i
 *
 * @param nl      Nonlinearity to initialize.
 * @param coeffs  Polynomial coefficients [c0, c1, ..., c_d].
 * @param degree  Polynomial degree (?0).
 * @return        0 on success, -1 on error.
 */
int wh_nl_init_polynomial(WH_Nonlinearity* nl,
                           const double* coeffs, int degree);

/**
 * wh_nl_init_spline ? Initialize as cubic spline.
 *
 * Given knots {k0 < k1 < ... < k_m} and values {v0, v1, ..., v_m},
 * constructs a C? cubic spline with natural boundary conditions
 * (f''(k0) = f''(k_m) = 0).
 *
 * @param nl       Nonlinearity to initialize.
 * @param knots    Knot positions (monotonically increasing, length n_knots).
 * @param values   Function values at knots (length n_knots).
 * @param n_knots  Number of knots (? 2).
 * @return         0 on success, -1 on error.
 */
int wh_nl_init_spline(WH_Nonlinearity* nl,
                       const double* knots, const double* values,
                       int n_knots);

/**
 * wh_nl_init_saturation ? Initialize saturation nonlinearity.
 *
 * f(x) = K * x            for |x| < L
 *      = K * sign(x) * L  for |x| ? L
 *
 * @param nl   Nonlinearity.
 * @param K    Linear gain in the unsaturated region.
 * @param L    Saturation limit.
 * @return     0 on success.
 */
int wh_nl_init_saturation(WH_Nonlinearity* nl, double K, double L);

/**
 * wh_nl_init_deadzone ? Initialize dead-zone nonlinearity.
 *
 * f(x) = 0              for |x| ? dz
 *      = x - dz*sign(x) for |x| > dz
 *
 * @param nl  Nonlinearity.
 * @param dz  Dead-zone half-width (? 0).
 * @return    0 on success.
 */
int wh_nl_init_deadzone(WH_Nonlinearity* nl, double dz);

/**
 * wh_nl_init_sigmoid ? Initialize sigmoid nonlinearity.
 *
 * f(x) = a / (1 + exp(-b*(x-c)))
 *
 * @param nl  Nonlinearity.
 * @param a   Amplitude (maximum value).
 * @param b   Steepness parameter (> 0 for increasing).
 * @param c   Center offset.
 * @return    0 on success.
 */
int wh_nl_init_sigmoid(WH_Nonlinearity* nl, double a, double b, double c);

/**
 * wh_nl_init_tanh ? Initialize hyperbolic tangent nonlinearity.
 *
 * f(x) = a * tanh(b*(x-c))
 *
 * @param nl  Nonlinearity.
 * @param a   Amplitude.
 * @param b   Steepness parameter (> 0 for increasing).
 * @param c   Center offset.
 * @return    0 on success.
 */
int wh_nl_init_tanh(WH_Nonlinearity* nl, double a, double b, double c);

/**
 * wh_nl_init_rbf ? Initialize Gaussian RBF network.
 *
 * f(x) = ?_{i=0}^{n-1} w_i * exp(-(x-c_i)?/(2*?_i?))
 *
 * @param nl      Nonlinearity.
 * @param centers RBF center positions (length n).
 * @param widths  RBF width parameters ?_i (length n).
 * @param weights RBF output weights (length n).
 * @param n       Number of RBF units.
 * @return        0 on success, -1 on error.
 */
int wh_nl_init_rbf(WH_Nonlinearity* nl,
                    const double* centers, const double* widths,
                    const double* weights, int n);

/**
 * wh_nl_init_lookup ? Initialize lookup table with linear interpolation.
 *
 * @param nl      Nonlinearity.
 * @param x_vals  Input breakpoints (monotonically increasing, length n).
 * @param y_vals  Output values at breakpoints (length n).
 * @param n       Number of table entries (? 2).
 * @return        0 on success, -1 on error.
 */
int wh_nl_init_lookup(WH_Nonlinearity* nl,
                       const double* x_vals, const double* y_vals, int n);

/* ??? Nonlinearity evaluation ???????????????????????????????????????????? */

/**
 * wh_nl_evaluate ? Evaluate the static nonlinearity at point x.
 *
 * @param nl  Nonlinearity.
 * @param x   Input value.
 * @return    f(x).
 */
double wh_nl_evaluate(const WH_Nonlinearity* nl, double x);

/**
 * wh_nl_evaluate_batch ? Evaluate over an array of inputs.
 *
 * @param nl  Nonlinearity.
 * @param x   Input array (length n).
 * @param y   Output array (pre-allocated, length n).
 * @param n   Number of points.
 */
void wh_nl_evaluate_batch(const WH_Nonlinearity* nl,
                           const double* x, double* y, int n);

/**
 * wh_nl_derivative ? Compute derivative df/dx at point x.
 *
 * For splines, this uses the analytical derivative of the cubic polynomial.
 * For RBF, uses chain rule on Gaussian basis.
 *
 * @param nl  Nonlinearity.
 * @param x   Input value.
 * @return    f'(x).
 */
double wh_nl_derivative(const WH_Nonlinearity* nl, double x);

/* ??? Nonlinearity analysis ?????????????????????????????????????????????? */

/**
 * wh_nl_get_range ? Estimate the output range over a given input domain.
 *
 * @param nl    Nonlinearity.
 * @param x_min Lower bound of input domain.
 * @param x_max Upper bound of input domain.
 * @param f_min Output: minimum f(x) on [x_min, x_max].
 * @param f_max Output: maximum f(x) on [x_min, x_max].
 */
void wh_nl_get_range(const WH_Nonlinearity* nl,
                      double x_min, double x_max,
                      double* f_min, double* f_max);

/**
 * wh_nl_is_monotonic ? Check if nonlinearity is monotonically increasing.
 *
 * Uses sampling across domain. For polynomial, checks derivative sign.
 *
 * @param nl  Nonlinearity.
 * @return    1 if f(x1) ? f(x2) for all x1 ? x2, 0 otherwise.
 */
int wh_nl_is_monotonic(const WH_Nonlinearity* nl);

/**
 * wh_nl_is_odd ? Check if nonlinearity satisfies f(-x) = -f(x).
 *
 * @param nl  Nonlinearity.
 * @return    1 if odd, 0 otherwise (approximate check via sampling).
 */
int wh_nl_is_odd(const WH_Nonlinearity* nl);

/**
 * wh_nl_find_root ? Find x such that f(x) = target using bisection.
 *
 * @param nl     Nonlinearity.
 * @param target Target value.
 * @param a      Left bound of search interval.
 * @param b      Right bound of search interval.
 * @param tol    Tolerance for convergence.
 * @return       Root x*, or NAN if no root found in interval.
 */
double wh_nl_find_root(const WH_Nonlinearity* nl, double target,
                        double a, double b, double tol);

/**
 * wh_nl_print ? Print nonlinearity parameters to stdout.
 */
void wh_nl_print(const WH_Nonlinearity* nl);

/**
 * wh_nl_copy ? Deep copy of a nonlinearity.
 *
 * @param dest  Destination.
 * @param src   Source.
 */
void wh_nl_copy(WH_Nonlinearity* dest, const WH_Nonlinearity* src);

#endif /* WH_NONLINEAR_H */
