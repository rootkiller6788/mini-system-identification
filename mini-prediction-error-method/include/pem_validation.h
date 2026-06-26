#ifndef PEM_VALIDATION_H
#define PEM_VALIDATION_H

#include "pem_core.h"

double pem_validation_fit(const double *y, const double *y_hat, int N);
double pem_validation_rsquared(const double *y, const double *y_hat, int N);
double pem_validation_adj_rsquared(const double *y, const double *y_hat, int N, int d);
double pem_validation_aic(double loss, int N, int d);
double pem_validation_aicc(double loss, int N, int d);
double pem_validation_bic(double loss, int N, int d);
double pem_validation_fpe(double loss, int N, int d);
double pem_validation_ljung_box(const double *eps, int N, int m);
double pem_validation_crosscorr_max(const double *eps, const double *u, int N, int max_lag);
int pem_validate_model(const double *y, const double *y_hat,
                       const double *eps, const double *u,
                       int N, int npar, PEMValidation *v);
int pem_cross_validate_blocks(const double *y, const double *y_hat, int N, int K,
                              double *avg_fit, double *std_fit);

#endif