#ifndef FREQID_APP_H
#define FREQID_APP_H
#include "freqid_defs.h"

freqid_transfer_function *freqid_app_identify_dc_motor(const double *t, const double *u, const double *y, size_t N, double fs);
freqid_transfer_function *freqid_app_identify_spring_mass_damper(const double *u, const double *y, size_t N, double fs, double *m_out, double *c_out, double *k_out);

typedef struct { double freq_hz, magnitude_db, q_factor, damping_ratio; } freqid_resonance_info;
int freqid_app_find_resonances(const freqid_frf *frf, freqid_resonance_info *resonances, size_t max_res, size_t *n_found);

double freqid_app_fault_index(const freqid_frf *baseline, const freqid_frf *current, double freq_low, double freq_high);
freqid_transfer_function *freqid_app_closed_loop_identify(const double *r, const double *y, size_t N, double fs, const freqid_transfer_function *controller);

int freqid_app_generate_swept_sine(double f_start, double f_end, double duration, double fs, double amplitude, double **t_out, double **u_out, size_t *n_out);
int freqid_app_generate_multisine(const double *freqs, const double *amps, const double *phases, size_t n_freqs, double fs, double duration, double **u_out, size_t *n_out);
void freqid_app_smooth_frf(freqid_frf *frf, size_t window_radius);

typedef struct { double freq_hz; double damping; int stable; } freqid_flutter_mode;
int freqid_app_flutter_analysis(const freqid_frf *frf, double airspeed, freqid_flutter_mode *modes, size_t max_modes, size_t *n_found);

int freqid_app_order_tracking(const double *vibration, const double *rpm, size_t N, double fs, size_t max_order, double ***order_mag_out, size_t *n_orders);
void freqid_app_order_tracking_free(double **om, size_t n_orders);

int freqid_app_acoustic_impedance(const double *mic1, const double *mic2, double mic_spacing, double fs, size_t N, freqid_frf **H12_out);
int freqid_app_grid_impedance(const double *v_grid, const double *i_grid, double fs, size_t N, double *R_out, double *L_out);

#endif
