#ifndef FREQID_CONVERT_H
#define FREQID_CONVERT_H
#include "freqid_defs.h"

freqid_transfer_function *freqid_c2d_tustin(const freqid_transfer_function *tf_ct, double Ts);
freqid_transfer_function *freqid_c2d_zoh(const freqid_transfer_function *tf_ct, double Ts);
freqid_transfer_function *freqid_d2c_tustin(const freqid_transfer_function *tf_dt, double Ts);
double freqid_prewarp_frequency(double fc, double Ts);
void freqid_pole_mapping_ct2dt(const freqid_complex *poles_ct, size_t n, double Ts, freqid_complex *poles_dt);
void freqid_pole_to_wn_zeta(freqid_complex pole, double *wn, double *zeta);

typedef int (*freqid_nonlin_func)(const double *x, const double *u, double *dx, double *y, size_t nx, size_t nu, size_t ny);
int freqid_linearize(freqid_nonlin_func f, const double *x0, const double *u0, size_t nx, size_t nu, size_t ny, double **A_out, double **B_out, double **C_out, double **D_out);

double freqid_df_ideal_relay(double A, double D);
double freqid_df_saturation(double A, double K, double delta);
double freqid_df_deadzone(double A, double K, double delta);
void freqid_df_backlash(double A, double K, double b, double *N_real, double *N_imag);
int freqid_harmonic_balance_residual(double A, double w, const freqid_transfer_function *G, double N_A, double *real_res, double *imag_res);

int freqid_estimate_pole_excess(const freqid_frf *frf, int *n_excess);
double freqid_estimate_dc_gain(const freqid_frf *frf);
int freqid_detect_integrator(const freqid_frf *frf, double *has_integrator);

#endif
