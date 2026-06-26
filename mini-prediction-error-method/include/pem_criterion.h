#ifndef PEM_CRITERION_H
#define PEM_CRITERION_H

#include "pem_core.h"

typedef enum {
    PEM_LOSS_QUADRATIC = 0,
    PEM_LOSS_ABSOLUTE  = 1,
    PEM_LOSS_HUBER     = 2,
    PEM_LOSS_VAPNIK    = 3
} PEMLossFunction;

double pem_loss_eval(double eps, PEMLossFunction loss_type, double param);
double pem_loss_derivative(double eps, PEMLossFunction loss_type, double param);

double pem_criterion_arx(const double *theta, int na, int nb, int nk,
                         const PEMData *data);
double pem_criterion_armax(const double *theta, int na, int nb, int nc, int nk,
                           const PEMData *data);
double pem_criterion_oe(const double *theta, int nb, int nf, int nk,
                        const PEMData *data);
double pem_criterion_bj(const double *theta, int nb, int nc, int nd, int nf, int nk,
                        const PEMData *data);
double pem_criterion_general(const double *eps, int N,
                             PEMLossFunction loss_type, double param);

void pem_gradient_arx(const double *theta, int na, int nb, int nk,
                      const PEMData *data, double *g);
void pem_gradient_armax(const double *theta, int na, int nb, int nc, int nk,
                        const PEMData *data, double *g);
void pem_gradient_oe(const double *theta, int nb, int nf, int nk,
                     const PEMData *data, double *g);
void pem_gradient_bj(const double *theta, int nb, int nc, int nd, int nf, int nk,
                     const PEMData *data, double *g);

void pem_hessian_arx(const double *theta, int na, int nb, int nk,
                     const PEMData *data, double *H);
void pem_hessian_oe(const double *theta, int nb, int nf, int nk,
                    const PEMData *data, double *H);
void pem_hessian_armax(const double *theta, int na, int nb, int nc, int nk,
                       const PEMData *data, double *H);
void pem_hessian_bj(const double *theta, int nb, int nc, int nd, int nf, int nk,
                    const PEMData *data, double *H);

void pem_regularize_hessian(double *H, int npar, double lambda);

#endif
