#ifndef PEM_PREDICTOR_H
#define PEM_PREDICTOR_H

#include "pem_core.h"

int pem_arx_nparam(int na, int nb);
int pem_armax_nparam(int na, int nb, int nc);
int pem_oe_nparam(int nb, int nf);
int pem_bj_nparam(int nb, int nc, int nd, int nf);

typedef struct {
    double *eps_history;
    double *w_history;
    int     max_lag;
    int     head;
} PEMPredictorState;

PEMPredictorState* pem_predictor_state_alloc(int max_lag);
void pem_predictor_state_free(PEMPredictorState *state);
void pem_predictor_state_reset(PEMPredictorState *state);

double pem_predict_arx(const double *theta, int na, int nb, int nk,
                       const double *u, const double *y, int t);
double pem_residual_arx(const double *theta, int na, int nb, int nk,
                        const double *u, const double *y, int t);
double pem_predict_armax(const double *theta, int na, int nb, int nc, int nk,
                         const double *u, const double *y, int t,
                         PEMPredictorState *state);
double pem_predict_oe(const double *theta, int nb, int nf, int nk,
                      const double *u, int t, PEMPredictorState *state);
double pem_predict_bj(const double *theta, int nb, int nc, int nd, int nf, int nk,
                      const double *u, const double *y, int t,
                      PEMPredictorState *state);
double pem_predict_fir(const double *theta, int nb, int nk,
                       const double *u, int t);

void pem_predict_arx_batch(const double *theta, int na, int nb, int nk,
                           const double *u, const double *y, int N, double *y_hat);
void pem_predict_armax_batch(const double *theta, int na, int nb, int nc, int nk,
                             const double *u, const double *y, int N, double *y_hat);
void pem_predict_oe_batch(const double *theta, int nb, int nf, int nk,
                          const double *u, int N, double *y_hat);
void pem_predict_bj_batch(const double *theta, int nb, int nc, int nd, int nf, int nk,
                          const double *u, const double *y, int N, double *y_hat);

void pem_residuals_arx(const double *theta, int na, int nb, int nk,
                       const double *u, const double *y, int N, double *epsilon);
void pem_residuals_armax(const double *theta, int na, int nb, int nc, int nk,
                         const double *u, const double *y, int N, double *epsilon);
void pem_residuals_oe(const double *theta, int nb, int nf, int nk,
                      const double *u, const double *y, int N, double *epsilon);
void pem_residuals_bj(const double *theta, int nb, int nc, int nd, int nf, int nk,
                      const double *u, const double *y, int N, double *epsilon);
void pem_kstep_predict_oe(const double *theta, int nb, int nf, int nk,
                          const double *u, int N, int k, double *y_hat);

#endif
