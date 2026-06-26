/**
 * wh_signal.h ? Signal Design and Generation for System Identification
 *
 * Provides excitation signal generation for WH model identification.
 * Proper input design is crucial for identifiability of block-structured
 * nonlinear models.
 *
 * Key principles (Schoukens et al., 2016):
 *   - Multisines allow frequency-domain separation of linear and nonlinear
 *     contributions via the Best Linear Approximation framework.
 *   - Random-phase multisines have user-controlled amplitude spectrum and
 *     random phases, yielding asymptotically Gaussian time-domain distribution.
 *   - The "robust method" uses multiple realizations of random-phase multisines
 *     to separate nonlinear distortions from disturbing noise.
 *   - Swept-sine (chirp) signals provide continuous frequency coverage.
 *
 * References:
 *   - Pintelon, R. & Schoukens, J. (2012). System Identification:
 *     A Frequency Domain Approach. 2nd ed. Wiley-IEEE Press.
 *   - Godfrey, K. (1993). Perturbation Signals for System Identification.
 *
 * Knowledge Level: L3 (Mathematical Structures), L5 (Algorithms)
 */

#ifndef WH_SIGNAL_H
#define WH_SIGNAL_H

#include <stddef.h>

/* ??? Signal statistics ?????????????????????????????????????????????????? */

/**
 * wh_signal_mean ? Compute sample mean of a signal.
 *
 * ? = (1/N) * ?_{i=0}^{N-1} x[i]
 */
double wh_signal_mean(const double* x, int N);

/**
 * wh_signal_variance ? Compute sample variance.
 *
 * ?? = (1/(N-1)) * ? (x[i] - ?)?
 */
double wh_signal_variance(const double* x, int N);

/**
 * wh_signal_rms ? Compute root-mean-square value.
 *
 * RMS = sqrt((1/N) * ? x[i]?)
 */
double wh_signal_rms(const double* x, int N);

/**
 * wh_signal_peak ? Compute peak absolute value (L? norm).
 */
double wh_signal_peak(const double* x, int N);

/**
 * wh_signal_crest_factor ? Compute crest factor: peak / RMS.
 *
 * Indicates how "peaky" the signal is. Low crest factor ? more uniform
 * amplitude distribution, better for exciting nonlinear systems.
 */
double wh_signal_crest_factor(const double* x, int N);

/**
 * wh_signal_autocorrelation ? Compute autocorrelation at lag k.
 *
 * R(k) = (1/(N-k)) * ?_{i=0}^{N-k-1} x[i]*x[i+k]   for k ? 0
 */
double wh_signal_autocorrelation(const double* x, int N, int k);

/* ??? Multisine generation ??????????????????????????????????????????????? */

/**
 * wh_signal_multisine ? Generate a random-phase multisine signal.
 *
 * s(t) = (1/?F) * ?_{k?K} A_k * sin(2??f_k?t/T_s + ?_k)
 *
 * where K is the set of excited frequency lines, f_k are uniformly spaced
 * frequencies, A_k are user-specified amplitudes (typically equal), and
 * ?_k are independent uniformly-distributed random phases on [0, 2?).
 *
 * Properties:
 *   - User-controlled amplitude spectrum (no dips in frequency coverage).
 *   - Asymptotically Gaussian amplitude distribution (by Central Limit Theorem).
 *   - Periodic with period T = N*T_s when frequencies are harmonics.
 *
 * @param out         Output signal array (length N).
 * @param N           Number of samples.
 * @param fs          Sampling frequency (Hz).
 * @param f_min       Minimum excited frequency (Hz), ? fs/N.
 * @param f_max       Maximum excited frequency (Hz), ? fs/2.
 * @param n_harmonics Number of excited harmonics in [f_min, f_max].
 * @param amplitude   Peak amplitude of each harmonic.
 * @param seed        Random seed (0 = use time-based seed).
 * @return            0 on success.
 */
int wh_signal_multisine(double* out, int N, double fs,
                         double f_min, double f_max, int n_harmonics,
                         double amplitude, unsigned int seed);

/**
 * wh_signal_multisine_odd ? Generate odd-odd multisine (only odd harmonics).
 *
 * Uses only odd frequency lines that are not multiples of 3.
 * This avoids exciting even-order nonlinear distortions at the excited lines,
 * enabling separation of even and odd nonlinear contributions.
 *
 * @param out         Output signal array (length N).
 * @param N           Number of samples. Must be power of 2 for FFT efficiency.
 * @param fs          Sampling frequency.
 * @param f_max       Maximum frequency (will use odd harmonics up to f_max).
 * @param n_harmonics Target number of excited harmonics.
 * @param amplitude   Peak amplitude.
 * @param seed        Random seed.
 * @return            0 on success.
 */
int wh_signal_multisine_odd(double* out, int N, double fs,
                             double f_max, int n_harmonics,
                             double amplitude, unsigned int seed);

/**
 * wh_signal_multisine_full ? Full multisine: all harmonics from f_min to f_max.
 *
 * Excites every harmonic in the specified range. Useful for frequency
 * response function (FRF) estimation.
 *
 * @param out         Output signal.
 * @param N           Number of samples.
 * @param fs          Sampling frequency.
 * @param f_min       Minimum frequency.
 * @param f_max       Maximum frequency.
 * @param amplitude   Amplitude per harmonic.
 * @param seed        Random seed.
 * @return            Number of excited harmonics.
 */
int wh_signal_multisine_full(double* out, int N, double fs,
                              double f_min, double f_max,
                              double amplitude, unsigned int seed);

/* ??? Other excitation signals ??????????????????????????????????????????? */

/**
 * wh_signal_chirp ? Generate a linear chirp (swept sine).
 *
 * u(t) = A * sin(2? * (f0 + (f1-f0)*t/(2*T)) * t)
 *
 * Sweeps frequency linearly from f0 to f1 over duration T.
 *
 * @param out      Output signal (length N).
 * @param N        Number of samples.
 * @param fs       Sampling frequency.
 * @param f0       Start frequency (Hz).
 * @param f1       End frequency (Hz).
 * @param A        Amplitude.
 * @return         0 on success.
 */
int wh_signal_chirp(double* out, int N, double fs,
                     double f0, double f1, double A);

/**
 * wh_signal_prbs ? Generate Pseudo-Random Binary Sequence.
 *
 * Uses a maximal-length linear feedback shift register (LFSR).
 * PRBS has a flat spectrum up to the clock frequency and is persistently
 * exciting of order 2^m - 1 for an m-stage shift register.
 *
 * @param out      Output signal (values -A or +A).
 * @param N        Number of samples.
 * @param A        Amplitude.
 * @param n_stages Number of LFSR stages (m, 2 ? m ? 31).
 * @param seed     Initial LFSR state (non-zero).
 * @return         0 on success.
 */
int wh_signal_prbs(double* out, int N, double A, int n_stages,
                    unsigned int seed);

/**
 * wh_signal_arx ? Generate colored Gaussian noise via AR filtering.
 *
 * e(t) ~ N(0, ??), then filtered: u(t) = -? a_i?u(t-i) + e(t)
 *
 * This approximates the spectrum of real-world excitation signals.
 *
 * @param out      Output signal (length N).
 * @param N        Number of samples.
 * @param a        AR coefficients (a[0]=1 for monic, length order+1).
 * @param order    AR order.
 * @param sigma    Standard deviation of driving white noise.
 * @param seed     Random seed.
 * @return         0 on success.
 */
int wh_signal_arx(double* out, int N, const double* a, int order,
                   double sigma, unsigned int seed);

/**
 * wh_signal_gaussian ? Generate white Gaussian noise.
 *
 * Uses Box-Muller transform to convert uniform to Gaussian.
 *
 * @param out      Output signal (length N).
 * @param N        Number of samples.
 * @param mean     Mean value.
 * @param sigma    Standard deviation.
 * @param seed     Random seed.
 * @return         0 on success.
 */
int wh_signal_gaussian(double* out, int N, double mean, double sigma,
                        unsigned int seed);

/**
 * wh_signal_step ? Generate step signal.
 *
 * u[k] = u0 for k < k_step, u[k] = u0 + amplitude for k ? k_step
 *
 * @param out      Output signal (length N).
 * @param N        Number of samples.
 * @param k_step   Sample index where step occurs.
 * @param u0       Initial value.
 * @param amplitude Step amplitude.
 */
void wh_signal_step(double* out, int N, int k_step,
                     double u0, double amplitude);

/**
 * wh_signal_sine ? Generate pure sinusoidal signal.
 *
 * u[k] = A * sin(2? * f * k / fs + phase) + offset
 *
 * @param out      Output signal (length N).
 * @param N        Number of samples.
 * @param fs       Sampling frequency (Hz).
 * @param f        Frequency (Hz).
 * @param A        Amplitude.
 * @param phase    Initial phase (radians).
 * @param offset   DC offset.
 */
void wh_signal_sine(double* out, int N, double fs, double f,
                     double A, double phase, double offset);

/**
 * wh_signal_ramp ? Generate ramp signal.
 *
 * u[k] = u0 + slope * k / fs
 */
void wh_signal_ramp(double* out, int N, double u0, double slope, double fs);

/* ??? Signal conditioning ???????????????????????????????????????????????? */

/**
 * wh_signal_normalize ? Normalize signal to zero mean and unit variance.
 *
 * out[k] = (x[k] - mean(x)) / std(x)
 *
 * @param in_out  Input and output signal (modified in-place, length N).
 * @param N       Number of samples.
 */
void wh_signal_normalize(double* in_out, int N);

/**
 * wh_signal_detrend ? Remove linear trend from signal.
 *
 * Fits y = a + b*k to the data and subtracts it.
 *
 * @param in_out  Signal to detrend (modified in-place, length N).
 * @param N       Number of samples.
 */
void wh_signal_detrend(double* in_out, int N);

/**
 * wh_signal_downsample ? Downsample signal by integer factor.
 *
 * Uses anti-aliasing via simple averaging (moving average filter).
 *
 * @param in        Input signal.
 * @param n_in      Number of input samples.
 * @param out       Output (downsampled) signal.
 * @param factor    Downsampling factor (? 1).
 * @return          Number of output samples = n_in / factor.
 */
int wh_signal_downsample(const double* in, int n_in, double* out, int factor);

/**
 * wh_signal_filter_lp ? Apply simple first-order low-pass filter.
 *
 * y[k] = ? * x[k] + (1-?) * y[k-1]   where ? = 2?*f_c/fs
 *
 * @param in_out  Signal (filtered in-place, length N).
 * @param N       Number of samples.
 * @param fc      Cutoff frequency (Hz).
 * @param fs      Sampling frequency (Hz).
 */
void wh_signal_filter_lp(double* in_out, int N, double fc, double fs);

#endif /* WH_SIGNAL_H */
