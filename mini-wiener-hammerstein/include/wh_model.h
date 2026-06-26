/**
 * wh_model.h ? Wiener-Hammerstein Model Core Definitions
 *
 * The Wiener-Hammerstein (WH) model is a block-structured nonlinear model
 * consisting of a linear dynamic block (L1), followed by a static nonlinearity (N),
 * followed by another linear dynamic block (L2):
 *
 *   u(t) ? [ L1 ] ? x(t) ? [ N ] ? w(t) ? [ L2 ] ? y(t)
 *
 * This structure captures a rich class of nonlinear dynamic systems while
 * maintaining structural interpretability. The WH model generalizes both
 * the Wiener model (L ? N) and the Hammerstein model (N ? L).
 *
 * References:
 *   - Billings, S.A. (2013). Nonlinear System Identification: NARMAX Methods.
 *   - Giri, F. & Bai, E.W. (2010). Block-oriented Nonlinear System Identification.
 *   - Schoukens, J. et al. (2015). "Identification of Wiener-Hammerstein Systems."
 *     Automatica, 52, 1-11.
 *
 * Knowledge Level: L1 (Definitions), L3 (Mathematical Structures)
 */

#ifndef WH_MODEL_H
#define WH_MODEL_H

#include <stddef.h>
#include <stdint.h>

/* ??? Model dimensions ??????????????????????????????????????????????????? */

#define WH_MAX_ORDER         64
#define WH_MAX_NL_PARAMS     32
#define WH_MAX_DATA_LEN      65536

/* ??? Enumerations ??????????????????????????????????????????????????????? */

typedef enum {
    WH_NL_POLYNOMIAL,
    WH_NL_SPLINE,
    WH_NL_SATURATION,
    WH_NL_DEADZONE,
    WH_NL_PIECEWISE_LINEAR,
    WH_NL_SIGMOID,
    WH_NL_TANH,
    WH_NL_GAUSSIAN_RBF,
    WH_NL_LOOKUP_TABLE,
    WH_NL_COUNT
} wh_nl_type_t;

typedef enum {
    WH_LIN_FIR,
    WH_LIN_IIR_TF,
    WH_LIN_STATE_SPACE,
    WH_LIN_COUNT
} wh_lin_type_t;

typedef enum {
    WH_ID_BLA,
    WH_ID_ITERATIVE,
    WH_ID_OVERPARAM,
    WH_ID_PEM_GRADIENT,
    WH_ID_COUNT
} wh_id_method_t;

typedef enum {
    WH_STATUS_OK = 0,
    WH_STATUS_UNSTABLE = 1,
    WH_STATUS_NOT_IDENTIFIED = 2,
    WH_STATUS_LOW_FIT = 3,
    WH_STATUS_PARAM_VIOLATION = 4,
    WH_STATUS_NUMERICAL_ERROR = 5
} wh_status_t;

/* ??? Core Data Structures ??????????????????????????????????????????????? */

typedef struct {
    wh_lin_type_t type;
    int nb;
    int na;
    int order;
    double b[64];
    double a[64];
    double A[4096];
    double B[64];
    double C[64];
    double D;
    double state[64];
    double state_buffer[4096];
    int state_dim;
    double Ts;
} WH_LinearBlock;

typedef struct {
    wh_nl_type_t type;
    int n_params;
    double params[32];
    int n_knots;
    double knots[32];
    double spline_coeffs[128];
    double lut_x[32];
    double lut_y[32];
    int lut_size;
    int n_centers;
    double centers[32];
    double rbf_widths[32];
    double rbf_weights[32];
} WH_Nonlinearity;

typedef struct {
    int order_C;
    int order_D;
    double C[64];
    double D[64];
    double noise_variance;
    double noise_buffer[64];
} WH_NoiseModel;

typedef struct {
    WH_LinearBlock   L1;
    WH_Nonlinearity  N;
    WH_LinearBlock   L2;
    WH_NoiseModel    noise;
    wh_id_method_t   method;
    wh_status_t      status;
    int              is_identified;
    double           fit_percent;
    double           mse;
    double           aic;
    double           bic;
    int              n_params_total;
    int              n_data_used;
    double           L1_state[64];
    double           L2_state[64];
    double           x_current;
    double           w_current;
} WH_Model;

/* ??? Core API ??????????????????????????????????????????????????????????? */

WH_Model* wh_model_create(void);
void wh_model_free(WH_Model* model);
WH_Model* wh_model_copy(const WH_Model* src);
double wh_model_evaluate(WH_Model* model, double u);
int wh_model_simulate(WH_Model* model,
                       const double* u, double* y, int n_samples);
void wh_model_reset(WH_Model* model);
int wh_model_get_delay(const WH_Model* model);
int wh_model_is_stable(const WH_Model* model);
void wh_model_print(const WH_Model* model);
const char* wh_model_get_nl_type_name(wh_nl_type_t type);
const char* wh_model_get_lin_type_name(wh_lin_type_t type);
int wh_model_count_parameters(const WH_Model* model);

#endif /* WH_MODEL_H */
