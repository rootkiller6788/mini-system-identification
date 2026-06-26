#include "freqid_defs.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ================================================================
 * Describing Function Analysis (L8: Nonlinear Frequency-Domain)
 * ================================================================
 * The describing function N(A,w) is the complex ratio of the
 * fundamental harmonic of the output to a sinusoidal input of
 * amplitude A and frequency w, for a static nonlinearity.
 *
 * This extends frequency-domain methods to nonlinear systems.
 * Ref: Khalil (2002), Ch. 7; Slotine & Li (1991), Ch. 4.
 */

/* Ideal relay describing function: N(A) = 4*D/(pi*A) */
double freqid_df_ideal_relay(double A, double D) {
    if (A < 1e-15) return 0.0;
    return 4.0 * D / (M_PI * A);
}

/* Saturation describing function:
 * N(A) = (2*K/pi)*(arcsin(delta/A) + (delta/A)*sqrt(1-(delta/A)^2))
 * for A > delta; N(A) = K for A <= delta */
double freqid_df_saturation(double A, double K, double delta) {
    if (A <= delta) return K;
    double ratio = delta / A;
    double asin_term = asin(ratio);
    double sqrt_term = ratio * sqrt(1.0 - ratio*ratio);
    return (2.0*K/M_PI) * (asin_term + sqrt_term);
}

/* Dead-zone describing function:
 * N(A) = K*(1 - (2/pi)*(arcsin(delta/A) + (delta/A)*sqrt(1-(delta/A)^2)))
 * for A > delta; N(A) = 0 for A <= delta */
double freqid_df_deadzone(double A, double K, double delta) {
    if (A <= delta) return 0.0;
    double ratio = delta / A;
    double asin_term = asin(ratio);
    double sqrt_term = ratio * sqrt(1.0 - ratio*ratio);
    return K * (1.0 - (2.0/M_PI)*(asin_term + sqrt_term));
}

/* Backlash (hysteresis) describing function:
 * Complex-valued because backlash introduces phase lag.
 * N(A) = N_R(A) + j*N_I(A) where:
 * N_R = (K/pi)*(pi/2 + arcsin(1-2b/A) + 2(1-2b/A)*sqrt((b/A)*(1-b/A)))
 * N_I = -(4Kb/(pi*A))*(1 - b/A)  for A > b
 */
void freqid_df_backlash(double A, double K, double b,
                         double *N_real, double *N_imag) {
    if (A <= b) { *N_real = 0.0; *N_imag = 0.0; return; }
    double ratio = b / A;
    double arg = 1.0 - 2.0*ratio;
    if (arg < -1.0) arg = -1.0; if (arg > 1.0) arg = 1.0;
    double asin_term = asin(arg);
    double sqrt_term = 2.0 * arg * sqrt(ratio*(1.0-ratio));
    *N_real = (K/M_PI) * (M_PI/2.0 + asin_term + sqrt_term);
    *N_imag = -(4.0*K*b/(M_PI*A)) * (1.0 - ratio);
}

/* ================================================================
 * Small-Signal Linearization (L4: Linearization Theory)
 * ================================================================
 * Linearizes a nonlinear system dx/dt = f(x,u), y = h(x,u)
 * around an equilibrium point (x0, u0).
 * Computes A = df/dx|_{x0,u0}, B = df/du|_{x0,u0},
 *          C = dh/dx|_{x0,u0}, D = dh/du|_{x0,u0}
 * via finite differences (numerical Jacobian).
 */

typedef int (*freqid_nonlin_func)(const double *x, const double *u,
                                   double *dx, double *y, size_t nx, size_t nu, size_t ny);

int freqid_linearize(freqid_nonlin_func f, const double *x0, const double *u0,
                      size_t nx, size_t nu, size_t ny,
                      double **A_out, double **B_out,
                      double **C_out, double **D_out) {
    if (!f || !x0 || !u0 || nx == 0 || !A_out || !B_out || !C_out || !D_out) return -1;
    double eps = 1e-6;
    double *dx_p = (double *)malloc(nx * sizeof(double));
    double *dx_m = (double *)malloc(nx * sizeof(double));
    double *y_p = (double *)malloc(ny * sizeof(double));
    double *y_m = (double *)malloc(ny * sizeof(double));
    double *x_pert = (double *)malloc(nx * sizeof(double));
    if (!dx_p || !dx_m || !y_p || !y_m || !x_pert) {
        free(dx_p); free(dx_m); free(y_p); free(y_m); free(x_pert); return -1;
    }
    double *A = (double *)calloc(nx*nx, sizeof(double));
    double *B = (double *)calloc(nx*nu, sizeof(double));
    double *C = (double *)calloc(ny*nx, sizeof(double));
    double *D = (double *)calloc(ny*nu, sizeof(double));
    if (!A || !B || !C || !D) { free(A); free(B); free(C); free(D); free(dx_p); free(dx_m); free(y_p); free(y_m); free(x_pert); return -1; }

    /* Compute A = df/dx via central differences */
    for (size_t j = 0; j < nx; j++) {
        memcpy(x_pert, x0, nx*sizeof(double));
        x_pert[j] = x0[j] + eps;
        f(x_pert, u0, dx_p, y_p, nx, nu, ny);
        x_pert[j] = x0[j] - eps;
        f(x_pert, u0, dx_m, y_m, nx, nu, ny);
        for (size_t i = 0; i < nx; i++) A[i*nx+j] = (dx_p[i] - dx_m[i]) / (2.0*eps);
        for (size_t i = 0; i < ny; i++) C[i*nx+j] = (y_p[i] - y_m[i]) / (2.0*eps);
    }

    /* Compute B = df/du and D = dh/du */
    f(x0, u0, dx_p, y_p, nx, nu, ny); /* baseline */
    double *u_pert = (double *)malloc(nu * sizeof(double));
    if (!u_pert) { free(A); free(B); free(C); free(D); free(dx_p); free(dx_m); free(y_p); free(y_m); free(x_pert); return -1; }
    for (size_t j = 0; j < nu; j++) {
        memcpy(u_pert, u0, nu*sizeof(double));
        u_pert[j] = u0[j] + eps;
        f(x0, u_pert, dx_p, y_p, nx, nu, ny);
        for (size_t i = 0; i < nx; i++) B[i*nu+j] = (dx_p[i] - dx_p[i]) / eps;
        for (size_t i = 0; i < ny; i++) D[i*nu+j] = (y_p[i] - y_p[i]) / eps;
    }
    free(u_pert);
    free(dx_p); free(dx_m); free(y_p); free(y_m); free(x_pert);
    *A_out = A; *B_out = B; *C_out = C; *D_out = D;
    return 0;
}

/* ================================================================
 * Harmonic Balance Residual (L8)
 * ================================================================
 * Computes the residual of the harmonic balance equation for
 * a SISO nonlinear system: G(jw)*N(A) + 1 = 0 at limit cycle.
 *
 * Solving R(A,w) = 0 yields the predicted amplitude and frequency
 * of limit cycles in feedback systems with one nonlinearity.
 */

int freqid_harmonic_balance_residual(double A, double w,
                                      const freqid_transfer_function *G,
                                      double N_A, double *real_res, double *imag_res) {
    if (!G || !real_res || !imag_res) return -1;
    freqid_complex G_jw = freqid_tf_eval(G, I * w);
    freqid_complex loop_gain = G_jw * N_A;
    freqid_complex residual = loop_gain + 1.0;
    *real_res = creal(residual);
    *imag_res = cimag(residual);
    return 0;
}
