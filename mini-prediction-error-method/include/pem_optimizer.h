#ifndef PEM_OPTIMIZER_H
#define PEM_OPTIMIZER_H

#include "pem_core.h"

int pem_cholesky(double *A, int n);
void pem_cholesky_solve(const double *L, const double *b, double *x, int n);
int pem_solve_spd(const double *A, const double *b, double *x, int n);

void pem_matvec(const double *A, const double *x, double *y, int m, int n);
void pem_matmul(const double *A, const double *B, double *C, int m, int p, int n);
void pem_transpose(const double *A, double *AT, int m, int n);
double pem_condition_number(const double *A, int n);
double pem_det_spd(const double *A, int n);
int pem_inverse_spd(const double *A, double *A_inv, int n);

typedef double (*pem_eval_function)(const double *x, void *data);

int pem_linesearch_backtrack(const double *x, const double *p, const double *g,
                             int n, double fx, double alpha_init,
                             double c1, double rho, int max_iter,
                             double *x_new,
                             pem_eval_function eval_fn, void *eval_data,
                             double *alpha_out, double *fn_out);

typedef double (*pem_obj_function)(const double *theta, void *data);
typedef void   (*pem_grad_function)(const double *theta, void *data, double *g);
typedef void   (*pem_hess_function)(const double *theta, void *data, double *H);

int pem_optimize_gauss_newton(double *theta, int npar,
                              pem_obj_function eval_f,
                              pem_grad_function eval_g,
                              pem_hess_function eval_H,
                              void *data,
                              const PEMOptions *opts,
                              PEMResult *result);
int pem_optimize_levenberg_marquardt(double *theta, int npar,
                                     pem_obj_function eval_f,
                                     pem_grad_function eval_g,
                                     pem_hess_function eval_H,
                                     void *data,
                                     const PEMOptions *opts,
                                     PEMResult *result);
int pem_optimize_sgd(double *theta, int npar,
                     pem_obj_function eval_f,
                     pem_grad_function eval_g,
                     void *data,
                     const PEMOptions *opts,
                     PEMResult *result);

#endif
