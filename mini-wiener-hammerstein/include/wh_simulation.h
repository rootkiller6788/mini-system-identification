/**
 * wh_simulation.h ? Time-Domain Simulation of Wiener-Hammerstein Models
 *
 * Provides a complete simulation engine for WH models, including:
 *   - Noise-free simulation (deterministic output)
 *   - Simulation with output noise (colored or white)
 *   - Multi-realization simulation (Monte Carlo)
 *   - Transient removal (discard initial samples until steady state)
 *   - Intermediate signal access (x(t), w(t))
 *
 * Theory:
 *   The WH model simulates the cascade:
 *     x[k] = ? b?_i?u[k-i] - ? a?_i?x[k-i]        (L1: IIR filter)
 *     w[k] = f(x[k])                                (N: static nonlinearity)
 *     y?[k] = ? b?_i?w[k-i] - ? a?_i?y?[k-i]        (L2: IIR filter)
 *     y[k] = y?[k] + v[k]                             (+ noise)
 *
 * For state-space representations, the discrete-time update is:
 *     ?[k+1] = A??[k] + B?u[k],   x[k] = C??[k]
 *
 * References:
 *   - Ljung, L. (1999). System Identification: Theory for the User. 2nd ed.
 *   - Pintelon, R. & Schoukens, J. (2012). System Identification:
 *     A Frequency Domain Approach. 2nd ed. Wiley-IEEE Press.
 *
 * Knowledge Level: L5 (Algorithms/Methods), L6 (Canonical Problems)
 */

#ifndef WH_SIMULATION_H
#define WH_SIMULATION_H

#include "wh_model.h"

/* ??? Simulation configuration ??????????????????????????????????????????? */

/**
 * WH_SimConfig ? Configuration for WH model simulation.
 */
typedef struct {
    int     n_transient;     /* Number of initial samples to discard      */
    int     add_noise;       /* Flag: add noise to output (1=yes)         */
    double  noise_std;       /* Standard deviation of additive noise      */
    int     noise_seed;      /* Seed for noise generator (0=use time)     */
    int     record_intermediate; /* Flag: record x(t) and w(t) signals   */
    int     verbosity;       /* 0=silent, 1=summary, 2=per-sample        */
} WH_SimConfig;

/**
 * WH_SimOutput ? Output of a simulation run.
 */
typedef struct {
    double*  y;              /* Output signal (length n_effective)        */
    double*  x;              /* Intermediate x(t) = L1 output (optional)  */
    double*  w;              /* Intermediate w(t) = N output (optional)   */
    double*  u;              /* Input signal (length n_effective)         */
    int      n_samples;      /* Number of effective samples (after transient) */
    double   mse;            /* Mean squared error if reference provided  */
    double   fit_percent;    /* FIT metric if reference provided          */
} WH_SimOutput;

/* ??? Core simulation API ???????????????????????????????????????????????? */

/**
 * wh_sim_config_default ? Get default simulation configuration.
 *
 * Defaults: n_transient=100, add_noise=0, noise_std=0.01,
 *           noise_seed=42, record_intermediate=0, verbosity=0.
 */
WH_SimConfig wh_sim_config_default(void);

/**
 * wh_sim_run ? Run a simulation of the WH model.
 *
 * Simulates the full WH model on the given input signal: u ? L1 ? N ? L2 ? y.
 * Optionally adds output noise and records intermediate signals.
 *
 * @param model   WH model to simulate.
 * @param u       Input signal (length n_samples).
 * @param n_samples Number of input samples.
 * @param config  Simulation configuration.
 * @param output  Pre-allocated output structure. Caller must free y, x, w.
 * @return        0 on success, -1 on error.
 */
int wh_sim_run(const WH_Model* model, const double* u, int n_samples,
               const WH_SimConfig* config, WH_SimOutput* output);

/**
 * wh_sim_run_with_reference ? Simulate and compare with reference output.
 *
 * Computes the FIT metric: FIT = 100 * (1 - ||y_ref - y_sim|| / ||y_ref - mean(y_ref)||)
 *
 * @param model    WH model.
 * @param u        Input signal.
 * @param y_ref    Reference output signal.
 * @param n_samples Number of samples.
 * @param config   Simulation configuration.
 * @param output   Output structure.
 * @return         0 on success.
 */
int wh_sim_run_with_reference(const WH_Model* model,
                               const double* u, const double* y_ref,
                               int n_samples, const WH_SimConfig* config,
                               WH_SimOutput* output);

/**
 * wh_sim_output_free ? Free memory allocated for simulation output.
 */
void wh_sim_output_free(WH_SimOutput* output);

/**
 * wh_sim_compute_fit ? Compute the FIT metric between simulated and reference.
 *
 * FIT = 100 * max(0, 1 - ||y_ref - y_sim||? / ||y_ref - mean(y_ref)||?)
 *
 * This is the standard metric in system identification (Ljung, 1999).
 * FIT = 100% means perfect fit; FIT < 0 means model is worse than constant.
 *
 * @param y_ref    Reference output.
 * @param y_sim    Simulated output.
 * @param n        Number of samples.
 * @return         FIT percentage in [??, 100].
 */
double wh_sim_compute_fit(const double* y_ref, const double* y_sim, int n);

/**
 * wh_sim_compute_mse ? Compute Mean Squared Error.
 *
 * MSE = (1/n) * ?_{i=0}^{n-1} (y_ref[i] - y_sim[i])?
 */
double wh_sim_compute_mse(const double* y_ref, const double* y_sim, int n);

/**
 * wh_sim_compute_rmse ? Compute Root Mean Squared Error.
 */
double wh_sim_compute_rmse(const double* y_ref, const double* y_sim, int n);

/**
 * wh_sim_compute_mae ? Compute Mean Absolute Error.
 */
double wh_sim_compute_mae(const double* y_ref, const double* y_sim, int n);

/**
 * wh_sim_compute_nrmse ? Compute Normalized RMSE (by signal range).
 */
double wh_sim_compute_nrmse(const double* y_ref, const double* y_sim, int n);

/* ??? Multi-realization simulation ??????????????????????????????????????? */

/**
 * wh_sim_monte_carlo ? Run multiple simulation realizations and compute statistics.
 *
 * @param model        WH model.
 * @param u            Input signal (same for all realizations).
 * @param n_samples    Number of samples.
 * @param n_runs       Number of Monte Carlo runs.
 * @param config_base  Base simulation config (noise seed randomized each run).
 * @param mean_y       Output: mean output across runs (length n_samples).
 * @param std_y        Output: standard deviation across runs (length n_samples).
 * @return             0 on success.
 */
int wh_sim_monte_carlo(const WH_Model* model, const double* u, int n_samples,
                        int n_runs, const WH_SimConfig* config_base,
                        double* mean_y, double* std_y);

/* ??? Transient analysis ????????????????????????????????????????????????? */

/**
 * wh_sim_find_transient ? Determine number of samples needed to reach steady state.
 *
 * Applies a step input and counts samples until output settles within
 * tolerance band. Based on the slowest time constant of L1 and L2.
 *
 * @param model      WH model.
 * @param tol        Relative tolerance for steady state detection (e.g., 0.01).
 * @return           Estimated number of transient samples.
 */
int wh_sim_find_transient(const WH_Model* model, double tol);

/**
 * wh_sim_impulse_response ? Compute impulse response of the full WH model.
 *
 * Applies a unit impulse u[0]=1, u[k>0]=0 and records output.
 * Note: Because the system is nonlinear, the impulse response depends on
 * the impulse amplitude. A small-amplitude impulse approximates the
 * linearized behavior around the origin.
 *
 * @param model      WH model.
 * @param amplitude  Impulse amplitude.
 * @param response   Pre-allocated output array (length n).
 * @param n          Number of samples.
 */
void wh_sim_impulse_response(const WH_Model* model, double amplitude,
                              double* response, int n);

/**
 * wh_sim_step_response ? Compute step response of the full WH model.
 *
 * @param model      WH model.
 * @param amplitude  Step amplitude.
 * @param response   Pre-allocated output array (length n).
 * @param n          Number of samples.
 */
void wh_sim_step_response(const WH_Model* model, double amplitude,
                           double* response, int n);

/**
 * wh_sim_frequency_sweep ? Simulate with chirp input and compute output spectrum.
 *
 * Input: u(t) = A * sin(2? * (f0 + (f1-f0)*t/(2*T)) * t)
 * This sweeps frequency from f0 to f1 linearly.
 *
 * @param model      WH model.
 * @param f0         Start frequency (Hz).
 * @param f1         End frequency (Hz).
 * @param fs         Sampling frequency (Hz).
 * @param duration   Sweep duration (seconds).
 * @param A          Amplitude.
 * @param y_out      Output: output signal (length fs*duration).
 * @return           0 on success.
 */
int wh_sim_frequency_sweep(const WH_Model* model,
                            double f0, double f1, double fs,
                            double duration, double A, double* y_out);

#endif /* WH_SIMULATION_H */
