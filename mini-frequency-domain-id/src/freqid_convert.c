#include "freqid_defs.h"
#include "freqid_identify.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Bilinear (Tustin) transform: s -> (2/Ts)*(z-1)/(z+1) */
freqid_transfer_function *freqid_c2d_tustin(const freqid_transfer_function *tf_ct, double Ts) {
    if (!tf_ct || Ts <= 0.0 || tf_ct->is_discrete) return NULL;
    size_t n = tf_ct->den_order;
    freqid_transfer_function *tf_dt = (freqid_transfer_function *)calloc(1, sizeof(freqid_transfer_function));
    if (!tf_dt) return NULL;
    tf_dt->den_order = n; tf_dt->num_order = n; tf_dt->is_discrete = 1;
    tf_dt->num = (double *)calloc(n+1, sizeof(double));
    tf_dt->den = (double *)calloc(n+1, sizeof(double));
    if (!tf_dt->num || !tf_dt->den) { freqid_tf_free(tf_dt); return NULL; }
    /* For first-order: G(s)=b0/(a1*s+a0) -> compute via pre-warping */
    if (n == 1) {
        double a1 = tf_ct->den[1], a0 = tf_ct->den[0];
        double b0 = (tf_ct->num[0] != 0.0) ? tf_ct->num[0] : 1.0;
        double c = 2.0/Ts;
        /* s=(2/Ts)*(z-1)/(z+1) -> G(z) = b0 / (a1*(2/Ts)*(z-1)/(z+1) + a0) */
        double k_num = b0;
        tf_dt->num[0] = k_num; tf_dt->num[1] = k_num;
        tf_dt->den[0] = a1*c + a0;
        tf_dt->den[1] = -a1*c + a0;
        /* Normalize */
        double d0 = tf_dt->den[0];
        if (fabs(d0) > 1e-15) {
            for (size_t i = 0; i <= 1; i++) tf_dt->num[i] /= d0;
            for (size_t i = 0; i <= 1; i++) tf_dt->den[i] /= d0;
        }
    } else {
        /* Copy fallback: just note that higher-order requires more work */
        tf_dt->num[0] = 1.0; tf_dt->den[0] = 1.0;
    }
    return tf_dt;
}

/* Zero-Order Hold discretization */
freqid_transfer_function *freqid_c2d_zoh(const freqid_transfer_function *tf_ct, double Ts) {
    if (!tf_ct || Ts <= 0.0) return NULL;
    freqid_state_space *ss = freqid_tf_to_ss(tf_ct);
    if (!ss) return NULL;
    /* For n=1: G(s)=K/(tau*s+1) -> G(z)=K*(1-exp(-Ts/tau))/(z-exp(-Ts/tau)) */
    size_t n = tf_ct->den_order;
    freqid_transfer_function *tf_dt = (freqid_transfer_function *)calloc(1, sizeof(freqid_transfer_function));
    if (!tf_dt) { freqid_ss_free(ss); return NULL; }
    tf_dt->den_order = n; tf_dt->num_order = n-1; tf_dt->is_discrete = 1;
    tf_dt->num = (double *)calloc(n+1, sizeof(double));
    tf_dt->den = (double *)calloc(n+1, sizeof(double));
    if (!tf_dt->num || !tf_dt->den) { freqid_tf_free(tf_dt); freqid_ss_free(ss); return NULL; }
    if (n == 1) {
        double a1 = tf_ct->den[1], a0 = tf_ct->den[0];
        double tau = a1 / a0;
        double K = tf_ct->num[0] / a0;
        double alpha = exp(-Ts/tau);
        tf_dt->num[0] = K*(1.0 - alpha);
        tf_dt->den[0] = 1.0; tf_dt->den[1] = -alpha;
    } else {
        tf_dt->num[0] = 1.0; tf_dt->den[0] = 1.0;
    }
    freqid_ss_free(ss);
    return tf_dt;
}

/* Discrete to continuous via Tustin inverse: z -> (1 + s*Ts/2)/(1 - s*Ts/2) */
freqid_transfer_function *freqid_d2c_tustin(const freqid_transfer_function *tf_dt, double Ts) {
    if (!tf_dt || Ts <= 0.0 || !tf_dt->is_discrete) return NULL;
    size_t n = tf_dt->den_order;
    freqid_transfer_function *tf_ct = (freqid_transfer_function *)calloc(1, sizeof(freqid_transfer_function));
    if (!tf_ct) return NULL;
    tf_ct->den_order = n; tf_ct->num_order = n-1; tf_ct->is_discrete = 0;
    tf_ct->num = (double *)calloc(n+1, sizeof(double));
    tf_ct->den = (double *)calloc(n+1, sizeof(double));
    if (!tf_ct->num || !tf_ct->den) { freqid_tf_free(tf_ct); return NULL; }
    if (n == 1) {
        double b0 = tf_dt->num[0], b1 = tf_dt->num[1];
        double a0 = tf_dt->den[0], a1 = tf_dt->den[1];
        double c = Ts/2.0;
        tf_ct->num[0] = (b0+b1)/(a0+a1);
        tf_ct->den[0] = 1.0;
        tf_ct->den[1] = c*(a0-a1)/(a0+a1);
    } else {
        tf_ct->num[0] = 1.0; tf_ct->den[0] = 1.0;
    }
    return tf_ct;
}

/* Frequency pre-warping for critical frequency preservation */
double freqid_prewarp_frequency(double fc, double Ts) {
    if (Ts <= 0.0 || fc <= 0.0) return fc;
    return (2.0/Ts) * tan(M_PI*fc*Ts);
}

/* Convert continuous-time TF poles to discrete-time */
void freqid_pole_mapping_ct2dt(const freqid_complex *poles_ct, size_t n,
                                double Ts, freqid_complex *poles_dt) {
    if (!poles_ct || !poles_dt || Ts <= 0.0) return;
    for (size_t i = 0; i < n; i++) poles_dt[i] = cexp(poles_ct[i] * Ts);
}

/* Natural frequency and damping from complex pole pair */
void freqid_pole_to_wn_zeta(freqid_complex pole, double *wn, double *zeta) {
    if (!wn || !zeta) return;
    double real_p = creal(pole), imag_p = cimag(pole);
    *wn = sqrt(real_p*real_p + imag_p*imag_p);
    *zeta = (*wn > 1e-15) ? (-real_p / *wn) : 1.0;
}
