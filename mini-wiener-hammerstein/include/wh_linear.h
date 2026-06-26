/**
 * wh_linear.h ? Linear Dynamic Block Implementations
 *
 * Provides the mathematical foundation for linear time-invariant (LTI) blocks
 * used in Wiener-Hammerstein models. Supports three representations:
 *
 *   FIR:  y[k] = ?_{i=0}^{nb-1} b[i] * u[k-i]
 *   IIR:  y[k] = ? b[i]*u[k-i] - ? a[i]*y[k-i]
 *   SS:   x[k+1] = A*x[k] + B*u[k]; y[k] = C*x[k] + D*u[k]
 *
 * Each representation has equivalent input-output behavior under proper
 * conversion, though numerical properties differ. FIR is always stable;
 * IIR and SS require pole checks.
 *
 * Theory:
 *   The transfer function of an LTI system is H(z) = B(z)/A(z) where
 *   B(z) = b0 + b1*z^{-1} + ... and A(z) = 1 + a1*z^{-1} + ...
 *   The system is BIBO stable iff all poles satisfy |p_i| < 1.
 *
 * References:
 *   - Ljung, L. (1999). System Identification: Theory for the User. 2nd ed.
 *   - Oppenheim, A.V. & Schafer, R.W. (2010). Discrete-Time Signal Processing.
 *
 * Knowledge Level: L3 (Mathematical Structures)
 */

#ifndef WH_LINEAR_H
#define WH_LINEAR_H

#include "wh_model.h"
#include <complex.h>

/* ??? Linear block life cycle ???????????????????????????????????????????? */

/**
 * wh_linear_init_fir ? Initialize a linear block as an FIR filter.
 *
 * H(z) = b[0] + b[1]*z^{-1} + ... + b[nb-1]*z^{-(nb-1)}
 *
 * @param block  Pointer to linear block to initialize.
 * @param b      Array of nb FIR coefficients.
 * @param nb     Number of coefficients (? 1).
 * @param Ts     Sampling period.
 * @return       0 on success, -1 if nb exceeds WH_MAX_ORDER.
 */
int wh_linear_init_fir(WH_LinearBlock* block, const double* b,
                        int nb, double Ts);

/**
 * wh_linear_init_iir ? Initialize as an IIR transfer function.
 *
 * H(z) = B(z)/A(z) with A(0) = 1 (monic denominator).
 *
 * @param block  Pointer to linear block.
 * @param b      Numerator coefficients B(z) (length nb).
 * @param nb     Number of B coefficients.
 * @param a      Denominator coefficients A(z) (length na, a[0] should be 1).
 * @param na     Number of A coefficients.
 * @param Ts     Sampling period.
 * @return       0 on success, -1 on parameter error.
 */
int wh_linear_init_iir(WH_LinearBlock* block, const double* b, int nb,
                        const double* a, int na, double Ts);

/**
 * wh_linear_init_ss ? Initialize as state-space model.
 *
 * Continuous-time: dx/dt = A*x + B*u, y = C*x + D*u
 * Discrete-time:   x[k+1] = A*x[k] + B*u[k], y[k] = C*x[k] + D*u[k]
 *
 * @param block  Pointer to linear block.
 * @param A      State matrix (row-major, order?order).
 * @param B      Input matrix/vector (length order).
 * @param C      Output matrix/vector (length order).
 * @param D      Feedthrough term.
 * @param order  State dimension.
 * @param Ts     Sampling period (0 for continuous-time).
 * @return       0 on success, -1 on error.
 */
int wh_linear_init_ss(WH_LinearBlock* block, const double* A,
                       const double* B, const double* C, double D,
                       int order, double Ts);

/* ??? Linear block evaluation ???????????????????????????????????????????? */

/**
 * wh_linear_evaluate ? Apply linear block to one input sample.
 *
 * Updates internal state. Computes y[k] = H(q)*u[k].
 *
 * @param block  Linear block.
 * @param u      Input sample.
 * @return       Output sample.
 */
double wh_linear_evaluate(WH_LinearBlock* block, double u);

/**
 * wh_linear_evaluate_batch ? Apply linear block to a sequence of inputs.
 *
 * @param block    Linear block (state updated after each sample).
 * @param u        Input sequence (length n).
 * @param y        Output sequence (pre-allocated, length n).
 * @param n        Number of samples.
 */
void wh_linear_evaluate_batch(WH_LinearBlock* block,
                               const double* u, double* y, int n);

/* ??? Linear block analysis ?????????????????????????????????????????????? */

/**
 * wh_linear_reset ? Reset internal state to zero.
 */
void wh_linear_reset(WH_LinearBlock* block);

/**
 * wh_linear_get_dc_gain ? Compute DC gain H(z=1).
 *
 * For FIR: sum of all b coefficients.
 * For IIR: sum(b)/sum(a).
 * For SS: D + C*(I-A)^{-1}*B.
 *
 * Returns: DC gain value, or NAN if system has pole at z=1.
 */
double wh_linear_get_dc_gain(const WH_LinearBlock* block);

/**
 * wh_linear_freq_response ? Compute H(e^{j?}) at frequency ?.
 *
 * @param block   Linear block.
 * @param omega   Normalized frequency in radians/sample [0, ?].
 * @param mag     Output: magnitude |H(e^{j?})|.
 * @param phase   Output: phase angle(H(e^{j?})) in radians.
 */
void wh_linear_freq_response(const WH_LinearBlock* block, double omega,
                              double* mag, double* phase);

/**
 * wh_linear_is_stable ? Check BIBO stability.
 *
 * For discrete-time systems: all poles |p_i| < 1.
 * For FIR: always stable.
 *
 * @param block  Linear block.
 * @return       1 if BIBO stable, 0 otherwise.
 */
int wh_linear_is_stable(const WH_LinearBlock* block);

/**
 * wh_linear_get_pole_radius ? Get maximum pole magnitude.
 *
 * @param block  Linear block.
 * @return       max_i |p_i|, or 0 for FIR systems.
 */
double wh_linear_get_pole_radius(const WH_LinearBlock* block);

/**
 * wh_linear_get_delay ? Return pure input-output delay in samples.
 *
 * For FIR: index of first non-zero b coefficient.
 * For IIR: index of first non-zero b coefficient (assuming causal, proper).
 *
 * @param block  Linear block.
 * @return       Number of pure delay samples.
 */
int wh_linear_get_delay(const WH_LinearBlock* block);

/**
 * wh_linear_impulse_response ? Compute impulse response up to n_samples.
 *
 * Applies unit impulse ?[0]=1, ?[k]=0 for k>0 and records output.
 *
 * @param block     Linear block (state will be modified).
 * @param impulse   Pre-allocated output array (length n_samples).
 * @param n_samples Number of impulse response samples.
 */
void wh_linear_impulse_response(WH_LinearBlock* block,
                                 double* impulse, int n_samples);

/**
 * wh_linear_step_response ? Compute step response up to n_samples.
 *
 * @param block     Linear block (state will be modified).
 * @param step_resp Pre-allocated output array (length n_samples).
 * @param n_samples Number of step response samples.
 */
void wh_linear_step_response(WH_LinearBlock* block,
                              double* step_resp, int n_samples);

/**
 * wh_linear_compute_poles ? Compute poles of the linear block.
 *
 * Finds roots of denominator polynomial A(z) using the companion matrix
 * eigenvalue method. Poles are returned as complex numbers.
 *
 * @param block   Linear block.
 * @param poles   Pre-allocated array for pole positions (length max_order).
 * @param max_poles Maximum number of poles to compute.
 * @return        Number of poles found (0 for FIR, 1..order for IIR/SS).
 */
int wh_linear_compute_poles(const WH_LinearBlock* block,
                             double complex* poles, int max_poles);

/**
 * wh_linear_print ? Print linear block parameters to stdout.
 */
void wh_linear_print(const WH_LinearBlock* block);

/**
 * wh_linear_copy ? Deep copy of a linear block.
 *
 * @param dest  Destination block.
 * @param src   Source block.
 */
void wh_linear_copy(WH_LinearBlock* dest, const WH_LinearBlock* src);

/**
 * wh_linear_convert_to_iir ? Convert any linear block to IIR representation.
 *
 * FIR is trivially converted (A(z)=1). State-space is converted via
 * the controllable canonical form.
 *
 * @param dest  Destination IIR block.
 * @param src   Source block (any type).
 * @return      0 on success, -1 on error.
 */
int wh_linear_convert_to_iir(WH_LinearBlock* dest, const WH_LinearBlock* src);

#endif /* WH_LINEAR_H */
