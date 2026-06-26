/**
 * clid_types.h — Core Type Definitions for Closed-Loop System Identification
 *
 * Defines the fundamental data structures for representing plants, controllers,
 * noise models, measurement data, and estimation results in closed-loop
 * identification problems.
 *
 * References:
 *   Ljung (1999) "System Identification: Theory for the User", 2nd ed., Ch.13
 *   Van den Hof (1998) "Closed-loop issues in system identification"
 *   Forssell & Ljung (1999) "Closed-loop identification revisited"
 *
 * Course Coverage: ETH 227-0216, Stanford AA203, MIT 6.241J
 */
#ifndef CLID_TYPES_H
#define CLID_TYPES_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ─── L1: Core Definition — System Model Type Enumeration ──────────── */
/** Models of the plant dynamics supported in closed-loop identification.
 *  Each corresponds to a distinct noise-model/plant-model pair as defined
 *  in Ljung (1999) §4.2.
 */
typedef enum {
    CLID_MODEL_ARX = 0,
    CLID_MODEL_ARMAX,
    CLID_MODEL_OE,
    CLID_MODEL_BJ,
    CLID_MODEL_STATE_SPACE,
    CLID_MODEL_FIR,
    CLID_MODEL_ARARX,
    CLID_MODEL_COUNT
} CLID_ModelType;

/* ─── L1: Closed-Loop Identification Method Enumeration ─────────────── */
typedef enum {
    CLID_METHOD_DIRECT = 0,
    CLID_METHOD_INDIRECT,
    CLID_METHOD_JOINT_IO,
    CLID_METHOD_TWO_STAGE,
    CLID_METHOD_COPRIME,
    CLID_METHOD_ASYM,
    CLID_METHOD_COUNT
} CLID_MethodType;

/* ─── L1: Noise Model Structure Enumeration ─────────────────────────── */
typedef enum {
    CLID_NOISE_WHITE = 0,
    CLID_NOISE_MA,
    CLID_NOISE_GENERAL,
    CLID_NOISE_ARMA,
    CLID_NOISE_COUNT
} CLID_NoiseModelType;

/* ─── L3: Mathematical Structure — Scalar Transfer Function ──────────── */
/** Rational transfer function: G(q) = B(q)/A(q) in the backward shift
 *  operator q^{-1}.  The fundamental building block for polynomial model
 *  descriptions (ARX, ARMAX, OE, BJ).
 *
 *  Mathematical structure:  B(q) = b0 + b1 q^{-1} + ... + b_{nb} q^{-nb}
 *                           A(q) = 1   + a1 q^{-1} + ... + a_{na} q^{-na}
 */
typedef struct {
    int      na;
    int      nb;
    int      nk;
    double  *a;
    double  *b;
    double   Ts;
} CLID_TransferFcn;

/* ─── L3: Mathematical Structure — State-Space Model ─────────────────── */
/** Discrete-time state-space model in innovation form:
 *      x(k+1) = A x(k) + B u(k) + K e(k)
 *      y(k)   = C x(k) + D u(k) + e(k)
 *  where e(k) is white noise with covariance Lambda.
 */
typedef struct {
    int      nx, nu, ny;
    double  *A, *B, *C, *D, *K, *Lambda;
    double   Ts;
} CLID_StateSpace;

/* ─── L1: Controller Model ───────────────────────────────────────────── */
typedef struct {
    int      is_state_space;
    union {
        CLID_TransferFcn tf;
        CLID_StateSpace  ss;
    } form;
    int      has_integrator;
    double   bandwidth;
} CLID_Controller;

/* ─── L3: Measurement Dataset ────────────────────────────────────────── */
/** Input-output data from a closed-loop experiment.
 *  Key property: u(t) and e(t) are correlated through feedback —
 *  the defining characteristic of closed-loop data.
 */
typedef struct {
    int      N, nu, ny, nr;
    double  *u, *y, *r;
    double   Ts;
    int      under_feedback;
    int      controller_knowledge;
} CLID_Dataset;

/* ─── L1: Identification Result ──────────────────────────────────────── */
typedef struct {
    CLID_ModelType  model_type;
    int             is_state_space;
    union {
        CLID_TransferFcn tf;
        CLID_StateSpace  ss;
    } identified_model;
    double         *noise_model;
    int             noise_na, noise_nc;
    double          loss_function;
    double          fit_percent;
    double          fpe;
    double          aic;
    double         *param_cov;
    int             n_params;
} CLID_Estimate;

/* ─── L2: Feedback Structure ─────────────────────────────────────────── */
typedef struct {
    CLID_TransferFcn  plant;
    CLID_Controller   controller;
    int               feedback_sign;
    int               excitation_point;
} CLID_FeedbackLoop;

/* ─── L1: Identification Options ─────────────────────────────────────── */
typedef struct {
    CLID_MethodType      method;
    CLID_ModelType       plant_model;
    CLID_NoiseModelType  noise_model;
    int                  na_min, na_max, nb_min, nb_max, nk;
    int                  max_iter;
    double               tolerance;
    int                  use_instrument, instrument_lag;
    int                  prefilter, prefilter_order;
    double              *prefilter_num, *prefilter_den;
} CLID_Options;

/* ─── L1: Excitation Signal Design ───────────────────────────────────── */
typedef struct {
    int      signal_type;
    double   amplitude, frequency_min, frequency_max;
    int      N, clock_period, num_periods;
    double   Ts;
    double  *multisine_amps, *multisine_phases;
    int      num_sines;
    uint64_t seed;
} CLID_ExcitationDesign;

/* ─── L2: Identifiability Result ─────────────────────────────────────── */
typedef struct {
    int     is_identifiable;
    int     pe_order_sufficient;
    int     controller_complexity_ok;
    int     noise_model_adequate;
    int     info_matrix_rank_defect;
    double  condition_number;
} CLID_Identifiability;

/* ─── L3: Asymptotic Covariance ──────────────────────────────────────── */
typedef struct {
    double *cov_matrix;
    int     p;
    double  trace_asym;
    double  det_asym;
    double *eigenvalues;
    double  condition_nr;
} CLID_AsymptoticCov;

/* ─── L3: Frequency Response ─────────────────────────────────────────── */
typedef struct {
    int      n_freqs;
    double  *omega, *mag, *phase, *mag_db;
} CLID_FrequencyResponse;

/* ─── L1: Uncertainty Region ─────────────────────────────────────────── */
typedef struct {
    double *center;
    double *P_inv;
    int     n_params;
    double  chi2_quantile;
    double  confidence;
    double  volume;
} CLID_UncertaintyRegion;

/* ─── L2: Bias Report ───────────────────────────────────────────────── */
typedef struct {
    double   bias_magnitude;
    double   bias_percent;
    int      bias_source;
    double  *bias_vector;
    int      n_params;
    double   worst_case_freq;
    double   worst_case_bias;
} CLID_BiasReport;

/* ─── L1: Prediction Error Sequence ──────────────────────────────────── */
typedef struct {
    int      N;
    double  *epsilon, *y_hat;
    double   V_N;
    double   epsilon_mean, epsilon_var, epsilon_rms;
    double   whiteness_q, whiteness_pval;
} CLID_PredictionError;

/* ─── Allocation / Deallocation API ──────────────────────────────────── */
CLID_TransferFcn       clid_tf_alloc(int na, int nb, int nk, double Ts);
void                   clid_tf_free(CLID_TransferFcn *tf);
CLID_StateSpace        clid_ss_alloc(int nx, int nu, int ny, double Ts);
void                   clid_ss_free(CLID_StateSpace *ss);
CLID_Dataset           clid_data_alloc(int N, int nu, int ny, int nr, double Ts);
void                   clid_data_free(CLID_Dataset *data);
CLID_Estimate          clid_estimate_alloc(int n_params);
void                   clid_estimate_free(CLID_Estimate *est);
CLID_Options           clid_options_default(void);
CLID_PredictionError   clid_pe_alloc(int N);
void                   clid_pe_free(CLID_PredictionError *pe);
CLID_FrequencyResponse clid_fr_alloc(int n_freqs);
void                   clid_fr_free(CLID_FrequencyResponse *fr);
CLID_UncertaintyRegion clid_ur_alloc(int n_params, double confidence);
void                   clid_ur_free(CLID_UncertaintyRegion *ur);

#ifdef __cplusplus
}
#endif

#endif /* CLID_TYPES_H */
