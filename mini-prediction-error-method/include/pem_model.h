#ifndef PEM_MODEL_H
#define PEM_MODEL_H

#include "pem_core.h"
#include "pem_predictor.h"
#include "pem_criterion.h"

int pem_estimate_arx_ls(const PEMData *data, int na, int nb, int nk,
                        PEMResult *result, const PEMOptions *opts);
int pem_estimate_arx(const PEMData *data, int na, int nb, int nk,
                     const double *theta0, PEMResult *result,
                     const PEMOptions *opts);
int pem_estimate_armax(const PEMData *data, int na, int nb, int nc, int nk,
                       const double *theta0, PEMResult *result,
                       const PEMOptions *opts);
int pem_estimate_oe(const PEMData *data, int nb, int nf, int nk,
                    const double *theta0, PEMResult *result,
                    const PEMOptions *opts);
int pem_estimate_bj(const PEMData *data, int nb, int nc, int nd, int nf, int nk,
                    const double *theta0, PEMResult *result,
                    const PEMOptions *opts);
int pem_estimate_fir(const PEMData *data, int nb, int nk,
                     PEMResult *result, const PEMOptions *opts);

int pem_simulate_model(PEMModelStructure structure, const double *theta,
                       const int *orders, int nk,
                       const double *u, int N, double *y_sim, const double *y0);
int pem_predict_kstep(PEMModelStructure structure, const double *theta,
                      const int *orders, int nk,
                      const double *u, const double *y, int N, int k,
                      double *y_hat);

#endif
