#include "pem_core.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

/* ============================================================================
 * PEM Core -- Memory Management, Polynomial Operations, Utilities
 * ============================================================================ */

/* --- Memory Management --- */

PEMData* pem_data_alloc(int N) {
    PEMData *data = (PEMData*)calloc(1, sizeof(PEMData));
    if (!data) return NULL;
    data->N = N;
    data->Ts = 1.0;
    data->u = (double*)calloc((size_t)N, sizeof(double));
    data->y = (double*)calloc((size_t)N, sizeof(double));
    data->t = NULL;
    data->name = NULL;
    if (!data->u || !data->y) { pem_data_free(data); return NULL; }
    return data;
}

void pem_data_free(PEMData *data) {
    if (!data) return;
    free(data->u); free(data->y);
    free(data->t); free(data->name);
    free(data);
}

PEMPolynomial pem_poly_alloc(int order) {
    PEMPolynomial p;
    p.order = order > 0 ? order : 1;
    p.coeff = (double*)calloc((size_t)p.order, sizeof(double));
    return p;
}

void pem_poly_free(PEMPolynomial *p) {
    if (!p) return;
    free(p->coeff);
    p->coeff = NULL; p->order = 0;
}

PEMResult* pem_result_alloc(int npar) {
    PEMResult *r = (PEMResult*)calloc(1, sizeof(PEMResult));
    if (!r) return NULL;
    r->npar = npar;
    r->theta_hat = (double*)calloc((size_t)npar, sizeof(double));
    r->covariance = (double*)calloc((size_t)(npar * npar), sizeof(double));
    r->gradient = (double*)calloc((size_t)npar, sizeof(double));
    r->information_matrix = (double*)calloc((size_t)(npar * npar), sizeof(double));
    r->status = PEM_NOT_STARTED;
    if (!r->theta_hat || !r->covariance || !r->gradient || !r->information_matrix) {
        pem_result_free(r); return NULL;
    }
    return r;
}

void pem_result_free(PEMResult *result) {
    if (!result) return;
    free(result->theta_hat); free(result->covariance);
    free(result->gradient); free(result->information_matrix);
    free(result);
}

PEMOptions pem_options_default(void) {
    PEMOptions opts;
    opts.max_iterations = 100;
    opts.tol_param = 1e-6;
    opts.tol_gradient = 1e-6;
    opts.tol_cost = 1e-8;
    opts.lambda_init = 1e-3;
    opts.lambda_factor = 10.0;
    opts.lambda_max = 1e10;
    opts.lambda_min = 1e-12;
    opts.algorithm = PEM_OPT_LM;
    opts.verbose = false;
    opts.compute_covariance = true;
    opts.line_search_c1 = 1e-4;
    opts.line_search_rho = 0.5;
    opts.max_line_search = 30;
    return opts;
}

PEMValidation* pem_validation_alloc(void) {
    return (PEMValidation*)calloc(1, sizeof(PEMValidation));
}

void pem_validation_free(PEMValidation *v) { free(v); }

/* --- Polynomial Operations --- */

double pem_poly_apply(const PEMPolynomial *p, const double *u, int t) {
    double result = 0.0;
    for (int k = 0; k < p->order; k++) {
        int idx = t - k;
        if (idx >= 0) result += p->coeff[k] * u[idx];
    }
    return result;
}

PEMPolynomial pem_poly_add(const PEMPolynomial *a, const PEMPolynomial *b) {
    int max_order = (a->order > b->order) ? a->order : b->order;
    PEMPolynomial r = pem_poly_alloc(max_order);
    for (int i = 0; i < max_order; i++) {
        double ai = (i < a->order) ? a->coeff[i] : 0.0;
        double bi = (i < b->order) ? b->coeff[i] : 0.0;
        r.coeff[i] = ai + bi;
    }
    return r;
}

PEMPolynomial pem_poly_mul(const PEMPolynomial *a, const PEMPolynomial *b) {
    int result_order = a->order + b->order - 1;
    if (result_order < 1) result_order = 1;
    PEMPolynomial r = pem_poly_alloc(result_order);
    for (int k = 0; k < result_order; k++) {
        double sum = 0.0;
        for (int i = 0; i < a->order; i++) {
            int j = k - i;
            if (j >= 0 && j < b->order) sum += a->coeff[i] * b->coeff[j];
        }
        r.coeff[k] = sum;
    }
    return r;
}

int pem_poly_long_division(const PEMPolynomial *num, const PEMPolynomial *den,
                           double *quotient, int max_terms,
                           double *remainder, int max_rem) {
    /* Polynomial long division: num/den = quotient + remainder/den
     * Uses synthetic division. Assumes den[0] = 1 (monic). */
    if (max_terms <= 0 || max_rem <= 0) return 0;
    for (int i = 0; i < max_rem && i < num->order; i++) remainder[i] = num->coeff[i];
    for (int i = num->order; i < max_rem; i++) remainder[i] = 0.0;
    double d0 = den->coeff[0];
    if (fabs(d0) < 1e-15) return -1;
    int n_quot = 0;
    for (int k = 0; k < max_terms && k < max_rem; k++) {
        if (fabs(remainder[k]) < 1e-15 && k >= num->order) break;
        double qk = remainder[k] / d0;
        quotient[n_quot++] = qk;
        for (int j = 0; j < den->order && (k + j) < max_rem; j++)
            remainder[k + j] -= qk * den->coeff[j];
    }
    return n_quot;
}

void pem_tf_simulate(const PEMTransferFunction *G, const double *u,
                     double *y, int N, const double *y0) {
    /* y(t) = G(q)u(t)  where G(q)=N(q)/D(q)
     * y(t) = -sum d_k y(t-k) + sum n_k u(t-k) for k>=0 */
    int nd = G->denominator.order;
    int nn = G->numerator.order;
    int max_init = (nd > 1) ? (nd - 1) : 0;
    if (max_init > 0 && y0) {
        for (int i = 0; i < max_init && i < N; i++) y[i] = y0[i];
    }
    for (int t = 0; t < N; t++) {
        double yt = 0.0;
        for (int k = 0; k < nn; k++) {
            int idx = t - k;
            if (idx >= 0) yt += G->numerator.coeff[k] * u[idx];
        }
        for (int k = 1; k < nd; k++) {
            int idx = t - k;
            if (idx >= 0) yt -= G->denominator.coeff[k] * y[idx];
        }
        if (t >= max_init || !y0) y[t] = yt;
    }
}

/* --- Utility Functions --- */

double pem_mean(const double *x, int n) {
    if (n <= 0) return 0.0;
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += x[i];
    return sum / (double)n;
}

double pem_variance(const double *x, int n, double mean) {
    if (n <= 1) return 0.0;
    double sum_sq = 0.0;
    for (int i = 0; i < n; i++) {
        double d = x[i] - mean;
        sum_sq += d * d;
    }
    return sum_sq / (double)(n - 1);
}

double pem_norm2(const double *v, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += v[i] * v[i];
    return sqrt(sum);
}

double pem_dot(const double *a, const double *b, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += a[i] * b[i];
    return sum;
}

double pem_rms_error(const double *y, const double *y_hat, int N) {
    if (N <= 0) return 0.0;
    double sum_sq = 0.0;
    for (int i = 0; i < N; i++) {
        double e = y[i] - y_hat[i];
        sum_sq += e * e;
    }
    return sqrt(sum_sq / (double)N);
}

double pem_nrmse_fit(const double *y, const double *y_hat, int N) {
    if (N <= 1) return 0.0;
    double y_mean = pem_mean(y, N);
    double num = 0.0, den = 0.0;
    for (int i = 0; i < N; i++) {
        double e = y[i] - y_hat[i];
        num += e * e;
        double d = y[i] - y_mean;
        den += d * d;
    }
    if (den < 1e-15) return 100.0;
    double nrmse = sqrt(num / den);
    return 100.0 * (1.0 - nrmse);
}

void pem_result_print(const PEMResult *r) {
    if (!r) return;
    printf("=== PEM Estimation Result ===\n");
    printf("Status: ");
    switch (r->status) {
        case PEM_CONVERGED:         printf("Converged\n"); break;
        case PEM_MAX_ITER:          printf("Max iterations\n"); break;
        case PEM_DIVERGED:          printf("Diverged\n"); break;
        case PEM_SINGULAR_HESSIAN:  printf("Singular Hessian\n"); break;
        case PEM_LINE_SEARCH_FAIL:  printf("Line search failed\n"); break;
        case PEM_NOT_STARTED:       printf("Not started\n"); break;
        case PEM_GRADIENT_TOL:      printf("Gradient tolerance\n"); break;
    }
    printf("Iterations: %d\n", r->iterations);
    printf("Loss: initial=%.6e final=%.6e\n", r->loss_init, r->loss_final);
    printf("Parameters (%d):\n", r->npar);
    for (int i = 0; i < r->npar; i++)
        printf("  theta[%d] = % .6e\n", i, r->theta_hat[i]);
    if (r->condition_number > 0)
        printf("Hessian condition: %.2e\n", r->condition_number);
    printf("Time: %.4f sec\n", r->elapsed_sec);
}

void pem_validation_print(const PEMValidation *v) {
    if (!v) return;
    printf("=== Model Validation ===\n");
    printf("N=%d Loss=%.6e Fit=%.2f%%\n", v->N, v->loss, v->fit_percent);
    printf("AIC=%.4f AICc=%.4f BIC=%.4f FPE=%.4f\n",
           v->aic, v->aicc, v->bic, v->fpe);
    printf("R2=%.4f AdjR2=%.4f\n", v->r_squared, v->adjusted_r_squared);
    printf("ResidWhiteness=%.4f CrossCorrMax=%.4f\n",
           v->residual_whiteness, v->crosscorr_max);
}