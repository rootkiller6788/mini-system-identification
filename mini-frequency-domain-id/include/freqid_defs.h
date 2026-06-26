#ifndef FREQID_DEFS_H
#define FREQID_DEFS_H
#include <stddef.h>
#include <complex.h>
#include <math.h>
typedef double complex freqid_complex;

freqid_complex freqid_complex_make(double real, double imag);
double         freqid_complex_real(freqid_complex z);
double         freqid_complex_imag(freqid_complex z);
freqid_complex freqid_complex_conj(freqid_complex z);
double         freqid_complex_mag(freqid_complex z);
double         freqid_complex_phase_rad(freqid_complex z);
double         freqid_complex_phase_deg(freqid_complex z);
double         freqid_complex_db(freqid_complex z);
freqid_complex freqid_complex_add(freqid_complex a, freqid_complex b);
freqid_complex freqid_complex_sub(freqid_complex a, freqid_complex b);
freqid_complex freqid_complex_mul(freqid_complex a, freqid_complex b);
freqid_complex freqid_complex_div(freqid_complex a, freqid_complex b);

typedef struct { double w_min, w_max; size_t n; double *w; } freqid_freq_vector;
int  freqid_freq_vector_linear(freqid_freq_vector *fv, double w_min, double w_max, size_t n);
int  freqid_freq_vector_log(freqid_freq_vector *fv, double w_min, double w_max, size_t n);
void freqid_freq_vector_free(freqid_freq_vector *fv);

typedef struct { freqid_complex value; double magnitude, phase_deg, db; } freqid_frf_point;
typedef struct { freqid_freq_vector freq; freqid_frf_point *points; } freqid_frf;

freqid_frf *freqid_frf_from_dft(const freqid_complex *U_fft, const freqid_complex *Y_fft, const double *freq_hz, size_t n_freq);
freqid_frf *freqid_frf_h1_estimator(const double *u, const double *y, size_t n_data, double fs, size_t n_fft, double overlap);
freqid_frf *freqid_frf_h2_estimator(const double *u, const double *y, size_t n_data, double fs, size_t n_fft, double overlap);
void freqid_frf_free(freqid_frf *frf);

typedef struct { size_t num_order, den_order; double *num, *den; int is_discrete; } freqid_transfer_function;
int  freqid_tf_create(freqid_transfer_function *tf, const double *num, size_t num_order, const double *den, size_t den_order, int is_discrete);
freqid_complex freqid_tf_eval(const freqid_transfer_function *tf, freqid_complex s);
freqid_frf *freqid_tf_eval_frf(const freqid_transfer_function *tf, const freqid_freq_vector *fv);
void freqid_tf_free(freqid_transfer_function *tf);

typedef struct { freqid_freq_vector freq; double *mag_db, *phase_deg; } freqid_bode_data;
freqid_bode_data *freqid_bode_from_frf(const freqid_frf *frf);
freqid_bode_data *freqid_bode_from_tf(const freqid_transfer_function *tf, const freqid_freq_vector *fv);
void freqid_bode_free(freqid_bode_data *bd);

typedef struct { double *real_part, *imag_part; size_t n; } freqid_nyquist_data;
freqid_nyquist_data *freqid_nyquist_from_frf(const freqid_frf *frf);
void freqid_nyquist_free(freqid_nyquist_data *nd);

typedef struct { double *mag_db, *phase_deg; size_t n; } freqid_nichols_data;
freqid_nichols_data *freqid_nichols_from_frf(const freqid_frf *frf);
void freqid_nichols_free(freqid_nichols_data *nd);

typedef struct { double *t, *y; size_t n; } freqid_time_response;
freqid_time_response *freqid_impulse_response(const freqid_transfer_function *tf, double t_end, size_t n_pts);
freqid_time_response *freqid_step_response(const freqid_transfer_function *tf, double t_end, size_t n_pts);
void freqid_time_response_free(freqid_time_response *tr);

#endif
