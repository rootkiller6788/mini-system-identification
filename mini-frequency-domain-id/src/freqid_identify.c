#include "freqid_identify.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

freqid_frf *freqid_etfe(const double *u, const double *y, size_t N, double fs) {
    if (!u || !y || N == 0 || fs <= 0.0) return NULL;
    freqid_complex *U_dft = NULL, *Y_dft = NULL;
    if (freqid_dft_real(u, N, &U_dft) != 0) return NULL;
    if (freqid_dft_real(y, N, &Y_dft) != 0) { free(U_dft); return NULL; }
    freqid_frf *frf = (freqid_frf *)calloc(1, sizeof(freqid_frf));
    if (!frf) { free(U_dft); free(Y_dft); return NULL; }
    size_t n_freq = N / 2 + 1;
    frf->freq.n = n_freq;
    frf->freq.w = (double *)malloc(n_freq * sizeof(double));
    frf->points = (freqid_frf_point *)calloc(n_freq, sizeof(freqid_frf_point));
    if (!frf->freq.w || !frf->points) { freqid_frf_free(frf); free(U_dft); free(Y_dft); return NULL; }
    for (size_t k = 0; k < n_freq; k++) {
        double f_hz = (double)k * fs / (double)N;
        frf->freq.w[k] = 2.0 * M_PI * f_hz;
        double U_mag2 = creal(U_dft[k])*creal(U_dft[k]) + cimag(U_dft[k])*cimag(U_dft[k]);
        if (U_mag2 > 1e-30) frf->points[k].value = Y_dft[k] / U_dft[k];
        else frf->points[k].value = 0.0;
        frf->points[k].magnitude = cabs(frf->points[k].value);
        frf->points[k].phase_deg = carg(frf->points[k].value) * 180.0 / M_PI;
        double m = frf->points[k].magnitude;
        frf->points[k].db = (m > 1e-15) ? 20.0 * log10(m) : -300.0;
    }
    frf->freq.w_min = frf->freq.w[0]; frf->freq.w_max = frf->freq.w[n_freq - 1];
    free(U_dft); free(Y_dft);
    return frf;
}

double freqid_fit_percent(const freqid_frf *measured, const freqid_frf *modeled) {
    if (!measured || !modeled || !measured->points || !modeled->points) return -1.0;
    size_t n = measured->freq.n < modeled->freq.n ? measured->freq.n : modeled->freq.n;
    if (n == 0) return -1.0;
    double mean_mag = 0.0, num = 0.0;
    for (size_t i = 0; i < n; i++) mean_mag += measured->points[i].magnitude;
    mean_mag /= (double)n;
    if (mean_mag < 1e-15) return 100.0;
    for (size_t i = 0; i < n; i++) {
        double diff = measured->points[i].magnitude - modeled->points[i].magnitude;
        num += diff * diff;
    }
    double denom = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = measured->points[i].magnitude - mean_mag; denom += d * d;
    }
    return (denom > 1e-15) ? 100.0 * (1.0 - sqrt(num/denom)) : 0.0;
}

double freqid_wsse(const freqid_frf *measured, const freqid_frf *modeled, const double *weight) {
    if (!measured || !modeled || !measured->points || !modeled->points) return -1.0;
    size_t n = measured->freq.n < modeled->freq.n ? measured->freq.n : modeled->freq.n;
    double wsse = 0.0;
    for (size_t i = 0; i < n; i++) {
        freqid_complex diff = measured->points[i].value - modeled->points[i].value;
        double err2 = creal(diff)*creal(diff) + cimag(diff)*cimag(diff);
        double w = weight ? weight[i] : 1.0;
        wsse += w * err2;
    }
    return wsse;
}

double freqid_aic(const freqid_frf *measured, const freqid_frf *modeled, size_t n_params, size_t n_data) {
    double wsse = freqid_wsse(measured, modeled, NULL);
    if (wsse < 0.0 || n_data == 0) return 1e308;
    return (double)n_data * log(wsse / (double)n_data) + 2.0 * (double)n_params;
}

/* Evaluate rational model G(s,theta) */
static freqid_complex _model_eval(const double *theta, size_t m, size_t n, freqid_complex s, int discrete) {
    if (discrete) s = cexp(I * cimag(s));
    freqid_complex num = theta[0] + 0.0*I, sp = 1.0;
    for (size_t k = 1; k <= m; k++) { sp *= s; num += theta[k] * sp; }
    freqid_complex den = 1.0; sp = 1.0;
    for (size_t k = 1; k <= n; k++) { sp *= s; den += theta[m + k] * sp; }
    double dm2 = creal(den)*creal(den) + cimag(den)*cimag(den);
    return (dm2 < 1e-30) ? 0.0 : (num / den);
}

freqid_transfer_function *freqid_ls_fit(const freqid_frf *frf, const double *weight,
                                         size_t m, size_t n, int discrete, size_t max_iter, double tol) {
    if (!frf || !frf->points || frf->freq.n == 0 || n == 0 || m > n) return NULL;
    if (max_iter == 0) max_iter = 50;
    if (tol <= 0.0) tol = 1e-6;
    size_t n_freq = frf->freq.n, n_par = m + n + 1;
    double *theta = (double *)calloc(n_par, sizeof(double));
    if (!theta) return NULL;
    theta[0] = 1.0;
    double lambda = 0.01, prev_cost = 1e308;
    for (size_t iter = 0; iter < max_iter; iter++) {
        size_t n_res = 2 * n_freq;
        double *J = (double *)calloc(n_res * n_par, sizeof(double));
        double *r = (double *)calloc(n_res, sizeof(double));
        double cost = 0.0;
        if (!J || !r) { free(J); free(r); goto done; }
        for (size_t i = 0; i < n_freq; i++) {
            freqid_complex s = I * frf->freq.w[i];
            freqid_complex Gm = _model_eval(theta, m, n, s, discrete);
            freqid_complex err = frf->points[i].value - Gm;
            double w = weight ? sqrt(weight[i]) : 1.0;
            r[2*i] = w * creal(err); r[2*i+1] = w * cimag(err);
            cost += w*w*(creal(err)*creal(err) + cimag(err)*cimag(err));
            double eps = 1e-6;
            for (size_t j = 0; j < n_par; j++) {
                double orig = theta[j];
                theta[j] = orig + eps;
                freqid_complex Gp = _model_eval(theta, m, n, s, discrete);
                theta[j] = orig - eps;
                freqid_complex Gm2 = _model_eval(theta, m, n, s, discrete);
                theta[j] = orig;
                freqid_complex dG = (Gp - Gm2) / (2.0*eps);
                J[(2*i)*n_par + j] = w * (-creal(dG));
                J[(2*i+1)*n_par + j] = w * (-cimag(dG));
            }
        }
        if (fabs(prev_cost - cost) / (fabs(cost) + 1e-10) < tol) { free(J); free(r); break; }
        prev_cost = cost;
        double *JTJ = (double *)calloc(n_par*n_par, sizeof(double));
        double *JTr = (double *)calloc(n_par, sizeof(double));
        for (size_t j = 0; j < n_par; j++) {
            for (size_t i = 0; i < n_res; i++) JTr[j] += J[i*n_par+j] * r[i];
            for (size_t k = j; k < n_par; k++) {
                double s = 0.0;
                for (size_t i = 0; i < n_res; i++) s += J[i*n_par+j] * J[i*n_par+k];
                JTJ[j*n_par+k] = s; JTJ[k*n_par+j] = s;
            }
            JTJ[j*n_par+j] += lambda;
        }
        double *dtheta = (double *)calloc(n_par, sizeof(double));
        for (size_t gs = 0; gs < 30; gs++)
            for (size_t j = 0; j < n_par; j++) {
                double sum = JTr[j];
                for (size_t k = 0; k < n_par; k++)
                    if (k != j) sum -= JTJ[j*n_par+k] * dtheta[k];
                if (fabs(JTJ[j*n_par+j]) > 1e-15) dtheta[j] = sum / JTJ[j*n_par+j];
            }
        double *theta_new = (double *)malloc(n_par * sizeof(double));
        for (size_t j = 0; j < n_par; j++) theta_new[j] = theta[j] + dtheta[j];
        double new_cost = 0.0;
        for (size_t i = 0; i < n_freq; i++) {
            freqid_complex Gm = _model_eval(theta_new, m, n, I * frf->freq.w[i], discrete);
            freqid_complex err = frf->points[i].value - Gm;
            double w = weight ? weight[i] : 1.0;
            new_cost += w*(creal(err)*creal(err) + cimag(err)*cimag(err));
        }
        if (new_cost < cost) { memcpy(theta, theta_new, n_par*sizeof(double)); lambda *= 0.5; }
        else lambda *= 5.0;
        free(theta_new); free(dtheta); free(JTJ); free(JTr); free(J); free(r);
    }
done:;
    freqid_transfer_function *tf = (freqid_transfer_function *)calloc(1, sizeof(freqid_transfer_function));
    if (!tf) { free(theta); return NULL; }
    tf->num_order = m; tf->den_order = n; tf->is_discrete = discrete;
    tf->num = (double *)malloc((m+1)*sizeof(double));
    tf->den = (double *)malloc((n+1)*sizeof(double));
    if (!tf->num || !tf->den) { freqid_tf_free(tf); free(theta); return NULL; }
    for (size_t i = 0; i <= m; i++) tf->num[i] = theta[i];
    tf->den[0] = 1.0;
    for (size_t i = 1; i <= n; i++) tf->den[i] = theta[m+i];
    free(theta);
    return tf;
}

/* Sanathanan-Koerner (SK) Iterative Method (L5/L8)
 * Linearizes rational fitting by weighted LS iterations. */
freqid_transfer_function *freqid_sk_fit(const freqid_frf *frf, size_t m, size_t n,
                                         int discrete, size_t max_iter, double tol) {
    if (!frf || !frf->points || frf->freq.n == 0 || n == 0 || m > n) return NULL;
    if (max_iter == 0) max_iter = 15; if (tol <= 0.0) tol = 1e-6;
    size_t n_freq = frf->freq.n, n_par = m + n + 1;
    double *den_prev = (double *)calloc(n + 1, sizeof(double));
    double *theta = (double *)calloc(n_par, sizeof(double));
    if (!den_prev || !theta) { free(den_prev); free(theta); return NULL; }
    den_prev[0] = 1.0; theta[0] = 1.0;
    double prev_cost = 1e308;
    for (size_t iter = 0; iter < max_iter; iter++) {
        /* Build linear system A*x = b where x = [b0..bm, a1..an] */
        size_t n_eq = 2 * n_freq;
        double *A = (double *)calloc(n_eq * n_par, sizeof(double));
        double *b = (double *)calloc(n_eq, sizeof(double));
        for (size_t i = 0; i < n_freq; i++) {
            freqid_complex G = frf->points[i].value;
            freqid_complex s = I * frf->freq.w[i];
            if (discrete) s = cexp(I * frf->freq.w[i]);
            /* Evaluate previous denominator at this frequency */
            freqid_complex D_prev = 1.0, sp = 1.0;
            for (size_t k = 1; k <= n; k++) { sp *= s; D_prev += den_prev[k] * sp; }
            double D_mag = cabs(D_prev);
            if (D_mag < 1e-10) D_mag = 1e-10;
            double W = 1.0 / D_mag; /* weight = 1/|D^{(l-1)}(jw)| */
            /* Error: W * (D(s,theta)*G - N(s,theta)) = 0 */
            freqid_complex sp_n = 1.0;
            for (size_t j = 0; j <= m; j++) {
                A[(2*i)*n_par + j] = W * (-creal(sp_n));
                A[(2*i+1)*n_par + j] = W * (-cimag(sp_n));
                sp_n *= s;
            }
            sp_n = 1.0;
            for (size_t j = 1; j <= n; j++) {
                sp_n *= s;
                A[(2*i)*n_par + m + j] = W * creal(G * sp_n);
                A[(2*i+1)*n_par + m + j] = W * cimag(G * sp_n);
            }
            b[2*i] = W * 0.0; b[2*i+1] = W * 0.0;
        }
        /* Solve normal equations: A^T * A * x = A^T * b */
        double *ATA = (double *)calloc(n_par * n_par, sizeof(double));
        double *ATb = (double *)calloc(n_par, sizeof(double));
        for (size_t k = 0; k < n_par; k++) {
            for (size_t i = 0; i < n_eq; i++) ATb[k] += A[i*n_par + k] * b[i];
            for (size_t j = 0; j < n_par; j++)
                for (size_t i = 0; i < n_eq; i++)
                    ATA[k*n_par + j] += A[i*n_par + k] * A[i*n_par + j];
        }
        /* Gauss-Seidel solve for ATA * x = ATb */
        double *x = (double *)calloc(n_par, sizeof(double));
        for (size_t gs = 0; gs < 40; gs++)
            for (size_t j = 0; j < n_par; j++) {
                double sum = ATb[j];
                for (size_t k = 0; k < n_par; k++)
                    if (k != j) sum -= ATA[j*n_par + k] * x[k];
                if (fabs(ATA[j*n_par + j]) > 1e-15) x[j] = sum / ATA[j*n_par + j];
            }
        /* Update theta and previous denominator */
        double new_cost = 0.0;
        for (size_t i = 0; i < n_freq; i++) {
            freqid_complex Gm = _model_eval(x, m, n, I * frf->freq.w[i], discrete);
            freqid_complex err = frf->points[i].value - Gm;
            new_cost += creal(err)*creal(err) + cimag(err)*cimag(err);
        }
        memcpy(theta, x, n_par*sizeof(double));
        den_prev[0] = 1.0;
        for (size_t k = 1; k <= n; k++) den_prev[k] = x[m + k];
        if (fabs(prev_cost - new_cost) / (fabs(new_cost) + 1e-10) < tol) { free(A); free(b); free(ATA); free(ATb); free(x); break; }
        prev_cost = new_cost;
        free(A); free(b); free(ATA); free(ATb); free(x);
    }
    freqid_transfer_function *tf = (freqid_transfer_function *)calloc(1, sizeof(freqid_transfer_function));
    if (!tf) { free(den_prev); free(theta); return NULL; }
    tf->num_order = m; tf->den_order = n; tf->is_discrete = discrete;
    tf->num = (double *)malloc((m+1)*sizeof(double));
    tf->den = (double *)malloc((n+1)*sizeof(double));
    if (!tf->num || !tf->den) { freqid_tf_free(tf); free(den_prev); free(theta); return NULL; }
    for (size_t i = 0; i <= m; i++) tf->num[i] = theta[i];
    tf->den[0] = 1.0;
    for (size_t i = 1; i <= n; i++) tf->den[i] = theta[m+i];
    free(den_prev); free(theta);
    return tf;
}

/* ML Estimation (L5) ¡ª Gaussian noise on FRF */
freqid_transfer_function *freqid_ml_fit(const freqid_frf *frf, const double *noise_var,
                                         size_t m, size_t n, int discrete) {
    if (!frf || !frf->points || frf->freq.n == 0 || n == 0 || m > n) return NULL;
    size_t n_freq = frf->freq.n;
    /* ML with known noise variance is weighted LS: weight = 1/noise_var */
    double *weights = NULL;
    if (noise_var) {
        weights = (double *)malloc(n_freq * sizeof(double));
        if (!weights) return NULL;
        for (size_t i = 0; i < n_freq; i++)
            weights[i] = (noise_var[i] > 1e-15) ? (1.0 / noise_var[i]) : 1.0;
    }
    freqid_transfer_function *tf = freqid_ls_fit(frf, weights, m, n, discrete, 50, 1e-6);
    free(weights);
    return tf;
}

/* State-Space to Transfer Function (L5)
 * G(s) = C*(sI-A)^{-1}*B + D
 * Uses Leverrier-Faddeev algorithm for SISO systems. */

freqid_state_space *freqid_tf_to_ss(const freqid_transfer_function *tf) {
    if (!tf || tf->den_order == 0) return NULL;
    size_t n = tf->den_order;
    freqid_state_space *ss = (freqid_state_space *)calloc(1, sizeof(freqid_state_space));
    if (!ss) return NULL;
    ss->n_states = n; ss->n_inputs = 1; ss->n_outputs = 1;
    ss->A = (double *)calloc(n*n, sizeof(double));
    ss->B = (double *)calloc(n, sizeof(double));
    ss->C = (double *)calloc(n, sizeof(double));
    ss->D = (double *)calloc(1, sizeof(double));
    if (!ss->A || !ss->B || !ss->C || !ss->D) { freqid_ss_free(ss); return NULL; }
    /* Controllable canonical form */
    for (size_t i = 0; i < n-1; i++) ss->A[i*n + (i+1)] = 1.0;
    double a0 = tf->den[0];
    for (size_t i = 0; i < n; i++) ss->A[(n-1)*n + i] = -tf->den[n-i] / a0;
    ss->B[n-1] = 1.0 / a0;
    for (size_t i = 0; i <= tf->num_order && i < n; i++) ss->C[i] = tf->num[i];
    ss->D[0] = (tf->num_order >= n) ? tf->num[n] / a0 : 0.0;
    return ss;
}

/* Leverrier-Faddeev algorithm for matrix resolvent (sI-A)^{-1} */
freqid_transfer_function *freqid_ss_to_tf(const freqid_state_space *ss) {
    if (!ss || ss->n_states == 0 || ss->n_inputs != 1 || ss->n_outputs != 1) return NULL;
    size_t n = ss->n_states;
    /* Compute characteristic polynomial coefficients via Faddeev-Leverrier */
    double *den = (double *)malloc((n+1) * sizeof(double));
    double *num = (double *)malloc((n+1) * sizeof(double));
    if (!den || !num) { free(den); free(num); return NULL; }
    /* Initialize: A_k for iteration */
    double *A_k = (double *)calloc(n*n, sizeof(double));
    double *A_copy = (double *)malloc(n*n * sizeof(double));
    if (!A_k || !A_copy) { free(den); free(num); free(A_k); free(A_copy); return NULL; }
    memcpy(A_copy, ss->A, n*n*sizeof(double));
    den[n] = 1.0; /* s^n coefficient = 1 */
    double *eye = (double *)calloc(n*n, sizeof(double));
    for (size_t i = 0; i < n; i++) eye[i*n+i] = 1.0;
    double *temp = (double *)calloc(n*n, sizeof(double));
    for (size_t k = 0; k < n; k++) {
        /* a_{n-1-k} = -(1/(k+1)) * trace(A_k) */
        double trace = 0.0;
        for (size_t i = 0; i < n; i++) trace += A_k[i*n+i];
        den[n-1-k] = -trace / (double)(k+1);
        /* A_{k+1} = A * (A_k + a_{n-1-k} * I) */
        for (size_t i = 0; i < n; i++)
            for (size_t j = 0; j < n; j++)
                temp[i*n+j] = A_k[i*n+j] + ((i==j) ? den[n-1-k] : 0.0);
        for (size_t i = 0; i < n; i++)
            for (size_t j = 0; j < n; j++) {
                A_k[i*n+j] = 0.0;
                for (size_t l = 0; l < n; l++)
                    A_k[i*n+j] += A_copy[i*n+l] * temp[l*n+j];
            }
    }
    /* Numerator: C * adj(sI-A) * B + D * det(sI-A)
       adj(sI-A) = sum_{k=0}^{n-1} s^{n-1-k} * (A_k + a_{n-1-k}*I + ...)
       Using simplified explicit form for SISO controllable canonical */
    /* For SISO system in controllable canonical form:
       num = [b0, b1, ..., bn] where B = [0,...,0,1]^T, C = [c0,...,cn-1]
       This directly yields: num = C coefficients, den = char poly of A */
    for (size_t i = 0; i <= n; i++) {
        if (i < n) num[i] = (i < ss->n_states) ? ss->C[i] : 0.0;
        else num[i] = 0.0;
    }
    /* Add D * denominator */
    if (ss->D) {
        for (size_t i = 0; i <= n; i++) num[i] += ss->D[0] * den[i];
    }
    /* Normalize: find actual numerator order */
    size_t num_ord = n;
    while (num_ord > 0 && fabs(num[num_ord]) < 1e-15) num_ord--;
    freqid_transfer_function *tf = (freqid_transfer_function *)calloc(1, sizeof(freqid_transfer_function));
    if (!tf) { free(den); free(num); free(A_k); free(A_copy); free(eye); free(temp); return NULL; }
    freqid_tf_create(tf, num, num_ord, den, n, ss->n_states > 0 ? 1 : 0);
    free(den); free(num); free(A_k); free(A_copy); free(eye); free(temp);
    return tf;
}

void freqid_ss_free(freqid_state_space *ss) {
    if (ss) { free(ss->A); free(ss->B); free(ss->C); free(ss->D); free(ss); }
}
