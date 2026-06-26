#ifndef NLSID_CORE_H
#define NLSID_CORE_H

#include <stddef.h>
#include <stdbool.h>

/* ============================================================================
 * mini-nonlinear-system-id: Core Definitions
 *
 * Nonlinear System Identification -- the art and science of building
 * mathematical models of nonlinear dynamical systems from observed
 * input-output data.
 *
 * Reference: Ljung (1999) "System Identification: Theory for the User"
 *            Nelles (2001) "Nonlinear System Identification"
 * ============================================================================ */

typedef struct {
    double* data;
    int length;
    double sample_time;
} Signal;

typedef struct {
    Signal** channels;
    int n_channels;
    int length;
} InputSignal;

typedef struct {
    Signal** channels;
    int n_channels;
    int length;
} OutputSignal;

typedef struct {
    InputSignal* input;
    OutputSignal* output;
    int n_samples;
    double sample_time;
    bool is_validation;
} NLSIDDataset;

typedef enum {
    NLSID_MODEL_NARX = 0,
    NLSID_MODEL_NARMAX = 1,
    NLSID_MODEL_HAMMERSTEIN = 2,
    NLSID_MODEL_WIENER = 3,
    NLSID_MODEL_VOLTERRA = 4,
    NLSID_MODEL_BILINEAR = 5,
    NLSID_MODEL_NEURAL_NET = 6,
    NLSID_MODEL_BASIS_FUNC = 7,
    NLSID_MODEL_WIENER_HAMMERSTEIN = 8,
    NLSID_MODEL_GP = 9,
    NLSID_MODEL_LPV = 10
} NLSIDModelType;

typedef enum {
    BASIS_POLYNOMIAL = 0,
    BASIS_SIGMOID = 1,
    BASIS_RBF = 2,
    BASIS_WAVELET = 3,
    BASIS_FOURIER = 4,
    BASIS_PIECEWISE_LINEAR = 5,
    BASIS_CUSTOM = 6
} BasisType;

typedef struct {
    BasisType type;
    int dim;
    double* params;
    int n_params;
    double (*evaluate)(const double* x, const double* params, int n_params);
} BasisFunction;

typedef struct {
    BasisFunction** bases;
    double* weights;
    int n_bases;
    int input_dim;
    double offset;
} BasisExpansion;

typedef struct {
    int ny;
    int nu;
    int nk;
    int n_inputs;
    int n_outputs;
    int regressor_dim;
    BasisExpansion* expansion;
    double* theta;
    int n_params;
} NARXModel;

typedef struct {
    BasisExpansion* static_nonlinearity;
    double* g_params;
    int n_g_params;
    double* a_coeffs;
    double* b_coeffs;
    int na;
    int nb;
    int nk;
    double* v;
    int n_v;
    int n_params_total;
} HammersteinModel;

typedef struct {
    double* a_coeffs;
    double* b_coeffs;
    int na;
    int nb;
    int nk;
    double* x_intermediate;
    int n_x;
    BasisExpansion* static_nonlinearity;
    double* h_params;
    int n_h_params;
    int n_params_total;
} WienerModel;

typedef struct {
    int order;
    int memory;
    double* kernel_h0;
    double* kernel_h1;
    double* kernel_h2;
    double* kernel_h3;
    int n_kernels;
    int input_dim;
} VolterraModel;

typedef struct {
    int n_states;
    int n_inputs;
    int n_outputs;
    double* A;
    double* B;
    double* C;
    double** N;
    double* D;
    int n_params_total;
} BilinearModel;

typedef enum {
    ACTIVATION_TANH = 0,
    ACTIVATION_SIGMOID = 1,
    ACTIVATION_RELU = 2,
    ACTIVATION_LINEAR = 3,
    ACTIVATION_LEAKY_RELU = 4,
    ACTIVATION_SWISH = 5
} ActivationType;

typedef struct {
    int n_layers;
    int* layer_sizes;
    double** weights;
    double** biases;
    ActivationType* activations;
    int n_params_total;
    int ny;
    int nu;
    int nk;
    int regressor_dim;
} NeuralNetModel;

struct NLSIDModel;

typedef struct NLSIDModel {
    NLSIDModelType type;
    char* name;
    int n_params;
    double* params;
    void* specific;
    int (*simulate)(struct NLSIDModel* model, const double* input, int n_steps,
                    double* output_pred);
    int (*predict_one_step)(struct NLSIDModel* model,
                            const double* input, const double* output_meas,
                            int t, double* y_hat);
    int (*get_params)(struct NLSIDModel* model, double* params);
    int (*set_params)(struct NLSIDModel* model, const double* params);
    void (*free)(struct NLSIDModel* model);
} NLSIDModel;

typedef enum {
    COST_QUADRATIC = 0,
    COST_ABSOLUTE = 1,
    COST_HUBER = 2,
    COST_EPSILON_INSENSITIVE = 3
} CostFunctionType;

typedef struct {
    int algorithm;
    int max_iterations;
    double tolerance;
    double lambda;
    double lambda_decay;
    double step_size_init;
    double step_size_min;
    double armijo_c1;
    double armijo_backtrack;
    CostFunctionType cost_type;
    double huber_delta;
    double epsilon_ins;
    int init_method;
    double init_stddev;
    bool verbose;
    int print_interval;
} NLSIDConfig;

#define NLSID_CONFIG_DEFAULT {1, 200, 1e-6, 0.001, 10.0, 1.0, 1e-10, 1e-4, 0.5, COST_QUADRATIC, 1.345, 0.1, 1, 0.01, false, 50}

typedef struct {
    NLSIDModel* model;
    double* param_estimates;
    double* param_std_errors;
    int n_params;
    double mse;
    double mse_validation;
    double fit_percent;
    double fit_percent_validation;
    double aic;
    double bic;
    double mdl;
    double fpe;
    double residual_mean;
    double residual_variance;
    double residual_skewness;
    double residual_kurtosis;
    double residual_whiteness;
    double residual_independence;
    int iterations_used;
    double final_gradient_norm;
    double final_cost;
    bool converged;
    double condition_number;
    double time_elapsed_sec;
} NLSIDResult;

typedef struct {
    double condition_number;
    double minimum_eigenvalue;
    int estimated_pe_order;
    bool is_pe;
    double* eigenvalues;
    int n_eigenvalues;
} PersistenceExcitation;

typedef struct {
    bool is_nonlinear;
    double nonlinearity_index;
    double coherence_thd;
    double bispectrum_magnitude;
    double correlation_dimension;
    double lyapunov_exponent;
    double surrogate_pvalue;
    bool coherence_test_pass;
    bool bispectrum_test_pass;
    bool sorrogate_test_pass;
    bool correlation_dim_test_pass;
    bool lyapunov_test_pass;
} NonlinearityTest;

Signal* nlsid_signal_create(int length, double sample_time);
void nlsid_signal_free(Signal* sig);
void nlsid_signal_set(Signal* sig, int index, double value);
double nlsid_signal_get(const Signal* sig, int index);
void nlsid_signal_fill(Signal* sig, double value);
void nlsid_signal_copy(const Signal* src, Signal* dst);
void nlsid_signal_add_noise(Signal* sig, double stddev, unsigned int* seed);
double nlsid_signal_mean(const Signal* sig);
double nlsid_signal_variance(const Signal* sig);
double nlsid_signal_rms(const Signal* sig);

InputSignal* nlsid_input_create(int n_channels, int length, double Ts);
void nlsid_input_free(InputSignal* in);
OutputSignal* nlsid_output_create(int n_channels, int length, double Ts);
void nlsid_output_free(OutputSignal* out);

NLSIDDataset* nlsid_dataset_create(int n_inputs, int n_outputs,
                                    int n_samples, double Ts);
void nlsid_dataset_free(NLSIDDataset* ds);
int nlsid_dataset_split(NLSIDDataset* ds, double ratio,
                         NLSIDDataset** estimation, NLSIDDataset** validation);
void nlsid_dataset_normalize(NLSIDDataset* ds);
void nlsid_dataset_remove_mean(NLSIDDataset* ds);

NLSIDModel* nlsid_model_create(NLSIDModelType type, const char* name);
void nlsid_model_free(NLSIDModel* model);
int nlsid_model_simulate(NLSIDModel* model, const double* input, int n_steps,
                          double* output_pred);
int nlsid_model_get_params(NLSIDModel* model, double* params);
int nlsid_model_set_params(NLSIDModel* model, const double* params);
int nlsid_model_nparams(const NLSIDModel* model);

PersistenceExcitation* nlsid_test_pe(const Signal* u, int max_order, int window);
void nlsid_pe_free(PersistenceExcitation* pe);
void nlsid_pe_print(const PersistenceExcitation* pe);

NonlinearityTest* nlsid_detect_nonlinearity(const NLSIDDataset* ds);
void nlsid_nonlinearity_free(NonlinearityTest* nt);
void nlsid_nonlinearity_print(const NonlinearityTest* nt);

double nlsid_compute_higher_order_coherence(const Signal* u, const Signal* y,
                                              int max_order);
double nlsid_compute_surrogate_pvalue(const Signal* u, const Signal* y,
                                       int n_surrogates, unsigned int* seed);
double nlsid_estimate_correlation_dimension(const Signal* y, int embed_dim,
                                              double radius);

double* nlsid_build_regressor(const double* y_hist, int ny,
                               const double* u_hist, int nu, int nk,
                               int t, int* reg_dim);
int nlsid_regressor_dimension(int ny, int nu, int n_inputs, int n_outputs);

double nlsid_compute_mse(const double* y, const double* y_hat, int n);
double nlsid_compute_fit(const double* y, const double* y_hat, int n);
double nlsid_compute_aic(double mse, int n_data, int n_params);
double nlsid_compute_bic(double mse, int n_data, int n_params);
double nlsid_compute_fpe(double mse, int n_data, int n_params);

void nlsid_result_print(const NLSIDResult* result);
void nlsid_model_print(const NLSIDModel* model);
void nlsid_signal_print(const Signal* sig, int max_samples);

#endif /* NLSID_CORE_H */