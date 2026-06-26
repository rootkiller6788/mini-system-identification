#ifndef FREQID_IDENTIFY_H
#define FREQID_IDENTIFY_H
#include "freqid_defs.h"
#include "freqid_spectrum.h"

freqid_frf *freqid_etfe(const double *u, const double *y, size_t N, double fs);
freqid_transfer_function *freqid_ls_fit(const freqid_frf *frf_measured, const double *weight, size_t m, size_t n, int is_discrete, size_t max_iter, double tol);
freqid_transfer_function *freqid_sk_fit(const freqid_frf *frf_measured, size_t m, size_t n, int is_discrete, size_t max_iter, double tol);
double freqid_fit_percent(const freqid_frf *measured, const freqid_frf *modeled);
double freqid_wsse(const freqid_frf *measured, const freqid_frf *modeled, const double *weight);
double freqid_aic(const freqid_frf *measured, const freqid_frf *modeled, size_t n_params, size_t n_data);
freqid_transfer_function *freqid_ml_fit(const freqid_frf *frf_measured, const double *noise_var, size_t m, size_t n, int is_discrete);

typedef struct { size_t n_states, n_inputs, n_outputs; double *A, *B, *C, *D; } freqid_state_space;
freqid_state_space *freqid_tf_to_ss(const freqid_transfer_function *tf);
freqid_transfer_function *freqid_ss_to_tf(const freqid_state_space *ss);
void freqid_ss_free(freqid_state_space *ss);

#endif
