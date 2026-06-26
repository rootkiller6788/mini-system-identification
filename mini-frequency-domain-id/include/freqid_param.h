#ifndef FREQID_PARAM_H
#define FREQID_PARAM_H
#include "freqid_defs.h"

typedef struct { size_t na, nb, nk; double *A, *B; int fitted; } freqid_arx_model;
int  freqid_arx_create(freqid_arx_model *arx, size_t na, size_t nb, size_t nk);
void freqid_arx_free(freqid_arx_model *arx);
freqid_complex freqid_arx_eval(const freqid_arx_model *arx, double omega, double Ts);
freqid_transfer_function *freqid_arx_to_tf(const freqid_arx_model *arx, double Ts);

typedef struct { size_t nb, nf, nk; double *B, *F; } freqid_oe_model;
int  freqid_oe_create(freqid_oe_model *oe, size_t nb, size_t nf, size_t nk);
void freqid_oe_free(freqid_oe_model *oe);
freqid_complex freqid_oe_eval(const freqid_oe_model *oe, double omega, double Ts);

typedef struct { size_t na, nb, nc, nk; double *A, *B, *C; } freqid_armax_model;
int  freqid_armax_create(freqid_armax_model *armax, size_t na, size_t nb, size_t nc, size_t nk);
void freqid_armax_free(freqid_armax_model *armax);

typedef struct { size_t nb, nc, nd, nf, nk; double *B, *C, *D, *F; } freqid_bj_model;
int  freqid_bj_create(freqid_bj_model *bj, size_t nb, size_t nc, size_t nd, size_t nf, size_t nk);
void freqid_bj_free(freqid_bj_model *bj);

int freqid_order_select_aic(const freqid_frf *frf, size_t n_max, int is_discrete, size_t *best_m, size_t *best_n);
int freqid_order_select_bic(const freqid_frf *frf, size_t n_max, int is_discrete, size_t *best_m, size_t *best_n);
int freqid_order_select_cv(const freqid_frf *frf_train, const freqid_frf *frf_valid, size_t n_max, int is_discrete, size_t *best_m, size_t *best_n);

#endif
