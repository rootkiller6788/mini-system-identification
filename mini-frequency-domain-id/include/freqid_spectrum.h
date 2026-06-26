#ifndef FREQID_SPECTRUM_H
#define FREQID_SPECTRUM_H
#include "freqid_defs.h"

typedef enum { FREQID_WIN_RECTANGLE=0, FREQID_WIN_HANN, FREQID_WIN_HAMMING, FREQID_WIN_BLACKMAN, FREQID_WIN_BARTLETT, FREQID_WIN_KAISER, FREQID_WIN_NUM_TYPES } freqid_window_type;

double *freqid_window_generate(freqid_window_type win_type, size_t N, double beta);
void freqid_window_apply(double *x, const double *w, size_t N);
double freqid_window_coherent_gain(const double *w, size_t N);

int freqid_dft_real(const double *x, size_t N, freqid_complex **X_out);
int freqid_idft(const freqid_complex *X, size_t N, double **x_out);
void freqid_fft_radix2(freqid_complex *x, size_t N, int inv);
void freqid_magnitude_spectrum(const freqid_complex *X, size_t N, double *mag_out);

int freqid_psd_welch(const double *x, size_t N, double fs, size_t n_fft, double overlap, freqid_window_type win_type, double **freq_out, double **psd_out, size_t *n_freq_out);
int freqid_cpsd_welch(const double *x, const double *y, size_t N, double fs, size_t n_fft, double overlap, freqid_window_type win_type, double **freq_out, freqid_complex **cpsd_out, size_t *n_freq_out);
int freqid_coherence(const double *x, const double *y, size_t N, double fs, size_t n_fft, double overlap, freqid_window_type win_type, double **freq_out, double **coh_out, size_t *n_freq_out);
int freqid_psd_blackman_tukey(const double *x, size_t N, size_t max_lag, double fs, double **freq_out, double **psd_out, size_t *n_freq_out);
int freqid_autocorr(const double *x, size_t N, size_t max_lag, double **r_out);

#endif
