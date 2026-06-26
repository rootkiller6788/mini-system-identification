#include "nlsid_models.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ============================================================================
 * Part 1: Basis Function Implementations
 * Each basis function type captures a different functional form that
 * can be used in basis expansions for nonlinear system models.
 * ============================================================================ */

double basis_eval_polynomial(const double* x, const double* p, int np) {
    if (!x || !p || np < 2) return 0.0;
    int degree = (int)p[0];
    int dim = np - 1;
    double dot = 0.0;
    for (int i = 0; i < dim; i++) dot += p[i + 1] * x[i];
    double result = 1.0;
    double base = dot;
    int d = (degree > 0) ? degree : 1;
    for (int i = 0; i < d; i++) result *= base;
    return result;
}

double basis_eval_sigmoid(const double* x, const double* p, int np) {
    if (!x || !p || np < 2) return 0.0;
    int dim = np - 1;
    double dot = 0.0;
    for (int i = 0; i < dim; i++) dot += p[i + 1] * x[i];
    double val = -p[0] * dot;
    if (val > 50.0) return 1.0;
    if (val < -50.0) return 0.0;
    return 1.0 / (1.0 + exp(val));
}

double basis_eval_rbf(const double* x, const double* p, int np) {
    if (!x || !p || np < 2) return 0.0;
    int dim = np - 1;
    double dist_sq = 0.0;
    for (int i = 0; i < dim; i++) {
        double d = x[i] - p[i + 1];
        dist_sq += d * d;
    }
    double sigma = p[0];
    if (sigma < 1e-10) sigma = 1e-10;
    double exponent = -dist_sq / (2.0 * sigma * sigma);
    if (exponent < -50.0) return 0.0;
    return exp(exponent);
}

double basis_eval_wavelet_morlet(const double* x, const double* p, int np) {
    if (!x || !p || np < 2) return 0.0;
    int dim = np - 1;
    double dist_sq = 0.0;
    for (int i = 0; i < dim; i++) {
        double d = x[i] - p[i + 1];
        dist_sq += d * d;
    }
    double dist = sqrt(dist_sq);
    return cos(p[0] * dist) * exp(-dist_sq / 2.0);
}

double basis_eval_fourier(const double* x, const double* p, int np) {
    if (!x || !p || np < 3) return 0.0;
    int dim = np - 2;
    double dot = 0.0;
    for (int i = 0; i < dim; i++) dot += p[i + 2] * x[i];
    return cos(2.0 * M_PI * p[0] * dot + p[1]);
}

double basis_eval_piecewise_linear(const double* x, const double* p, int np) {
    if (!x || !p || np < 2) return 0.0;
    int dim = np - 1;
    double dot = 0.0;
    for (int i = 0; i < dim; i++) dot += p[i + 1] * x[i];
    double result = dot - p[0];
    return (result > 0.0) ? result : 0.0;
}

BasisFunction* basis_create(BasisType type, int dim, const double* params,
                             int n_params) {
    BasisFunction* bf = (BasisFunction*)calloc(1, sizeof(BasisFunction));
    if (!bf) return NULL;
    bf->type = type;
    bf->dim = dim;
    bf->n_params = n_params;
    bf->params = (double*)calloc((size_t)n_params, sizeof(double));
    if (!bf->params) { free(bf); return NULL; }
    if (params) memcpy(bf->params, params, (size_t)n_params * sizeof(double));
    switch (type) {
        case BASIS_POLYNOMIAL: bf->evaluate = basis_eval_polynomial; break;
        case BASIS_SIGMOID: bf->evaluate = basis_eval_sigmoid; break;
        case BASIS_RBF: bf->evaluate = basis_eval_rbf; break;
        case BASIS_WAVELET: bf->evaluate = basis_eval_wavelet_morlet; break;
        case BASIS_FOURIER: bf->evaluate = basis_eval_fourier; break;
        case BASIS_PIECEWISE_LINEAR: bf->evaluate = basis_eval_piecewise_linear; break;
        default: bf->evaluate = NULL; break;
    }
    return bf;
}

void basis_free(BasisFunction* bf) {
    if (!bf) return;
    free(bf->params);
    free(bf);
}

double basis_evaluate(const BasisFunction* bf, const double* x) {
    if (!bf || !bf->evaluate || !x) return 0.0;
    return bf->evaluate(x, bf->params, bf->n_params);
}

/* ============================================================================
 * Basis Expansion
 * ============================================================================ */

BasisExpansion* basis_expansion_create(int input_dim, int n_bases) {
    BasisExpansion* be = (BasisExpansion*)calloc(1, sizeof(BasisExpansion));
    if (!be) return NULL;
    be->input_dim = input_dim;
    be->n_bases = 0;
    be->bases = NULL;
    be->weights = NULL;
    be->offset = 0.0;
    return be;
}

void basis_expansion_free(BasisExpansion* be) {
    if (!be) return;
    if (be->bases) {
        for (int i = 0; i < be->n_bases; i++) basis_free(be->bases[i]);
        free(be->bases);
    }
    free(be->weights);
    free(be);
}

void basis_expansion_add_basis(BasisExpansion* be, BasisType type,
                                const double* params, int n_params) {
    if (!be) return;
    int n = be->n_bases;
    be->bases = (BasisFunction**)realloc(be->bases,
        (size_t)(n + 1) * sizeof(BasisFunction*));
    be->weights = (double*)realloc(be->weights,
        (size_t)(n + 1) * sizeof(double));
    be->bases[n] = basis_create(type, be->input_dim, params, n_params);
    be->weights[n] = 0.0;
    be->n_bases = n + 1;
}

double basis_expansion_eval(const BasisExpansion* be, const double* x) {
    if (!be || !x) return 0.0;
    double y = be->offset;
    for (int i = 0; i < be->n_bases; i++) {
        if (be->bases[i] && be->bases[i]->evaluate)
            y += be->weights[i] * basis_evaluate(be->bases[i], x);
    }
    return y;
}

void basis_expansion_eval_vector(const BasisExpansion* be,
                                  const double** X, int n_samples, double* y) {
    if (!be || !X || !y) return;
    for (int t = 0; t < n_samples; t++)
        y[t] = basis_expansion_eval(be, X[t]);
}

void basis_expansion_get_jacobian(const BasisExpansion* be,
                                   const double* x, double* J) {
    if (!be || !x || !J) return;
    J[0] = 1.0;
    for (int i = 0; i < be->n_bases; i++)
        J[i + 1] = basis_evaluate(be->bases[i], x);
}

int basis_expansion_nparams(const BasisExpansion* be) {
    if (!be) return 0;
    return 1 + be->n_bases;
}

void basis_expansion_pack_params(const BasisExpansion* be, double* theta) {
    if (!be || !theta) return;
    theta[0] = be->offset;
    for (int i = 0; i < be->n_bases; i++) theta[i + 1] = be->weights[i];
}

void basis_expansion_unpack_params(BasisExpansion* be, const double* theta) {
    if (!be || !theta) return;
    be->offset = theta[0];
    for (int i = 0; i < be->n_bases; i++) be->weights[i] = theta[i + 1];
}

void basis_expansion_print(const BasisExpansion* be) {
    if (!be) { printf("BasisExpansion: NULL\n"); return; }
    printf("Basis Expansion: %d bases, input_dim=%d\n", be->n_bases, be->input_dim);
    printf("  Offset (bias): %.4f\n", be->offset);
    for (int i = 0; i < be->n_bases; i++) {
        printf("  Basis %d: type=%d, weight=%.4f\n", i,
               be->bases[i] ? be->bases[i]->type : -1, be->weights[i]);
    }
}

BasisExpansion* basis_expansion_polynomial(int input_dim, int max_degree) {
    BasisExpansion* be = basis_expansion_create(input_dim, 0);
    if (!be) return NULL;
    for (int deg = 1; deg <= max_degree; deg++) {
        for (int d = 0; d < input_dim; d++) {
            double* params = (double*)calloc((size_t)(input_dim + 1), sizeof(double));
            params[0] = (double)deg;
            params[d + 1] = 1.0;
            basis_expansion_add_basis(be, BASIS_POLYNOMIAL, params, input_dim + 1);
            free(params);
        }
    }
    return be;
}

BasisExpansion* basis_expansion_rbf_uniform(int input_dim, int n_centers,
                                              double range_lo, double range_hi,
                                              double sigma) {
    BasisExpansion* be = basis_expansion_create(input_dim, 0);
    if (!be) return NULL;
    double range = range_hi - range_lo;
    if (range <= 0.0) range = 1.0;
    int cpd = (int)ceil(pow((double)n_centers, 1.0 / input_dim));
    if (cpd < 1) cpd = 1;
    for (int idx = 0; idx < n_centers; idx++) {
        double* params = (double*)calloc((size_t)(input_dim + 1), sizeof(double));
        params[0] = sigma;
        int rem = idx;
        for (int d = 0; d < input_dim; d++) {
            int pos = rem % cpd;
            rem /= cpd;
            params[d + 1] = range_lo + ((pos + 0.5) / cpd) * range;
        }
        basis_expansion_add_basis(be, BASIS_RBF, params, input_dim + 1);
        free(params);
    }
    return be;
}

BasisExpansion* basis_expansion_fourier_trunc(int input_dim, int n_harmonics) {
    BasisExpansion* be = basis_expansion_create(input_dim, 0);
    if (!be) return NULL;
    for (int h = 1; h <= n_harmonics; h++) {
        for (int d = 0; d < input_dim; d++) {
            double* pc = (double*)calloc((size_t)(input_dim + 2), sizeof(double));
            pc[0] = (double)h; pc[1] = 0.0; pc[d + 2] = 1.0;
            basis_expansion_add_basis(be, BASIS_FOURIER, pc, input_dim + 2);
            free(pc);
            double* ps = (double*)calloc((size_t)(input_dim + 2), sizeof(double));
            ps[0] = (double)h; ps[1] = -M_PI / 2.0; ps[d + 2] = 1.0;
            basis_expansion_add_basis(be, BASIS_FOURIER, ps, input_dim + 2);
            free(ps);
        }
    }
    return be;
}

/* ============================================================================
 * Part 2: NARX Model
 * ============================================================================ */

NARXModel* narx_create(int ny, int nu, int nk, int n_inputs, int n_outputs) {
    NARXModel* narx = (NARXModel*)calloc(1, sizeof(NARXModel));
    if (!narx) return NULL;
    narx->ny = ny; narx->nu = nu; narx->nk = nk;
    narx->n_inputs = n_inputs; narx->n_outputs = n_outputs;
    narx->regressor_dim = ny * n_outputs + nu * n_inputs;
    narx->expansion = NULL; narx->theta = NULL; narx->n_params = 0;
    return narx;
}

void narx_free(NARXModel* narx) {
    if (!narx) return;
    basis_expansion_free(narx->expansion);
    free(narx->theta);
    free(narx);
}

void narx_set_expansion(NARXModel* narx, BasisExpansion* expansion) {
    if (!narx) return;
    if (narx->expansion) basis_expansion_free(narx->expansion);
    narx->expansion = expansion;
    narx->n_params = basis_expansion_nparams(expansion);
    free(narx->theta);
    narx->theta = (double*)calloc((size_t)narx->n_params, sizeof(double));
}

double narx_predict_one_step(const NARXModel* narx,
                              const double* y_hist, const double* u_hist,
                              int t) {
    if (!narx || !narx->expansion) return 0.0;
    int reg_dim;
    double* phi = nlsid_build_regressor(y_hist, narx->ny, u_hist,
                                         narx->nu, narx->nk, t, &reg_dim);
    if (!phi) return 0.0;
    double pred = basis_expansion_eval(narx->expansion, phi);
    free(phi);
    return pred;
}

int narx_simulate(const NARXModel* narx, const double* input, int n_steps,
                  const double* y0, double* y_pred) {
    if (!narx || !input || !y_pred || n_steps <= 0) return -1;
    if (!narx->expansion) return -1;
    int max_lag = (narx->ny > narx->nu + narx->nk) ? narx->ny : narx->nu + narx->nk;
    double* y_hist = (double*)calloc((size_t)(max_lag + n_steps), sizeof(double));
    for (int i = 0; i < max_lag && y0; i++) y_hist[i] = y0[i];
    for (int t = 0; t < n_steps; t++) {
        int reg_dim;
        double* phi = nlsid_build_regressor(&y_hist[max_lag + t - 1],
            narx->ny, &input[t - narx->nk + 1], narx->nu, narx->nk, 1, &reg_dim);
        if (!phi) { free(y_hist); return -1; }
        y_hist[max_lag + t] = basis_expansion_eval(narx->expansion, phi);
        y_pred[t] = y_hist[max_lag + t];
        free(phi);
    }
    free(y_hist);
    return 0;
}

int narx_nparams(const NARXModel* narx) { return narx ? narx->n_params : 0; }

void narx_get_params(const NARXModel* narx, double* theta) {
    if (narx && theta) memcpy(theta, narx->theta, (size_t)narx->n_params * sizeof(double));
}

void narx_set_params(NARXModel* narx, const double* theta) {
    if (!narx || !theta) return;
    memcpy(narx->theta, theta, (size_t)narx->n_params * sizeof(double));
    if (narx->expansion) basis_expansion_unpack_params(narx->expansion, theta);
}

void narx_compute_regressor_matrix(const NARXModel* narx,
                                    const double* y, const double* u,
                                    int n_samples, double** Phi, int* n_reg) {
    if (!narx || !y || !u || !Phi || !n_reg) return;
    int reg_dim = narx->regressor_dim;
    int offset = (narx->ny > narx->nu + narx->nk) ? narx->ny : narx->nu + narx->nk;
    int n_eff = n_samples - offset;
    if (n_eff <= 0) { *n_reg = 0; return; }
    *n_reg = n_eff;
    *Phi = (double*)calloc((size_t)(n_eff * reg_dim), sizeof(double));
    if (!*Phi) { *n_reg = 0; return; }
    for (int t = offset; t < n_samples; t++) {
        int row = t - offset;
        for (int i = 0; i < narx->ny && i < reg_dim; i++)
            (*Phi)[row * reg_dim + i] = y[t - 1 - i];
        for (int i = 0; i < narx->nu && (narx->ny + i) < reg_dim; i++) {
            int idx = t - narx->nk - i;
            (*Phi)[row * reg_dim + narx->ny + i] = (idx >= 0) ? u[idx] : 0.0;
        }
    }
}

void narx_print(const NARXModel* narx) {
    if (!narx) { printf("NARXModel: NULL\n"); return; }
    printf("NARX Model: ny=%d, nu=%d, nk=%d\n", narx->ny, narx->nu, narx->nk);
    printf("  Regressor dimension: %d\n", narx->regressor_dim);
    printf("  Parameters: %d\n", narx->n_params);
    if (narx->expansion) basis_expansion_print(narx->expansion);
}
/* ============================================================================
 * Part 3: Hammerstein Model
 * Implementation of: u(t) -> g(u(t)) = v(t) -> G(q)v(t) = y(t)
 * ============================================================================ */

HammersteinModel* hammerstein_create(int na, int nb, int nk) {
    HammersteinModel* hm = (HammersteinModel*)calloc(1, sizeof(HammersteinModel));
    if (!hm) return NULL;
    hm->na = na; hm->nb = nb; hm->nk = nk;
    hm->a_coeffs = (double*)calloc((size_t)(na + 1), sizeof(double));
    hm->b_coeffs = (double*)calloc((size_t)(nb + 1), sizeof(double));
    if (hm->a_coeffs) hm->a_coeffs[0] = 1.0;
    if (hm->b_coeffs) hm->b_coeffs[0] = 1.0;
    hm->static_nonlinearity = NULL;
    hm->g_params = NULL; hm->n_g_params = 0;
    hm->n_params_total = na + nb + 1;
    return hm;
}

void hammerstein_free(HammersteinModel* hm) {
    if (!hm) return;
    basis_expansion_free(hm->static_nonlinearity);
    free(hm->g_params); free(hm->a_coeffs);
    free(hm->b_coeffs); free(hm->v);
    free(hm);
}

void hammerstein_set_static_nl(HammersteinModel* hm, BasisExpansion* nl) {
    if (!hm) return;
    if (hm->static_nonlinearity) basis_expansion_free(hm->static_nonlinearity);
    hm->static_nonlinearity = nl;
    hm->n_g_params = basis_expansion_nparams(nl);
    hm->n_params_total = hm->n_g_params + hm->na + hm->nb + 1;
}

void hammerstein_set_linear_part(HammersteinModel* hm,
                                  const double* a, const double* b) {
    if (!hm) return;
    if (a) memcpy(hm->a_coeffs, a, (size_t)(hm->na + 1) * sizeof(double));
    if (b) memcpy(hm->b_coeffs, b, (size_t)(hm->nb + 1) * sizeof(double));
}

void hammerstein_compute_v(const HammersteinModel* hm,
                            const double* u, int n, double* v_out) {
    if (!hm || !u || !v_out) return;
    if (hm->static_nonlinearity) {
        for (int t = 0; t < n; t++)
            v_out[t] = basis_expansion_eval(hm->static_nonlinearity, &u[t]);
    } else {
        for (int t = 0; t < n; t++) v_out[t] = u[t];
    }
}

int hammerstein_simulate(const HammersteinModel* hm,
                          const double* input, int n_steps,
                          const double* v0, double* y_pred) {
    if (!hm || !input || !y_pred || n_steps <= 0) return -1;
    double* v = (double*)calloc((size_t)n_steps, sizeof(double));
    if (!v) return -1;
    hammerstein_compute_v(hm, input, n_steps, v);

    int max_order = (hm->na > hm->nb) ? hm->na : hm->nb;
    double* y_buf = (double*)calloc((size_t)(n_steps + max_order), sizeof(double));
    if (!y_buf) { free(v); return -1; }

    for (int t = 0; t < n_steps; t++) {
        int idx = max_order + t;
        double sum = 0.0;
        for (int i = 0; i <= hm->nb; i++) {
            int vi = idx - hm->nk - i;
            sum += hm->b_coeffs[i] * ((vi >= 0 && vi < n_steps) ? v[vi] : 0.0);
        }
        for (int i = 1; i <= hm->na; i++)
            sum -= hm->a_coeffs[i] * y_buf[idx - i];
        y_buf[idx] = sum;
        y_pred[t] = sum;
    }
    free(v); free(y_buf);
    return 0;
}

double hammerstein_predict_one_step(const HammersteinModel* hm,
                                     const double* u_hist,
                                     const double* y_hist, int t) {
    if (!hm) return 0.0;
    double vt = 0.0;
    if (hm->static_nonlinearity && u_hist)
        vt = basis_expansion_eval(hm->static_nonlinearity, &u_hist[t]);
    double y_hat = 0.0;
    for (int i = 0; i <= hm->nb; i++) {
        int idx = t - hm->nk - i;
        double val = (idx >= 0 && u_hist) ? u_hist[idx] : 0.0;
        if (hm->static_nonlinearity)
            val = basis_expansion_eval(hm->static_nonlinearity, &val);
        y_hat += hm->b_coeffs[i] * val;
    }
    for (int i = 1; i <= hm->na; i++) {
        double yp = (t - i >= 0 && y_hist) ? y_hist[t - i] : 0.0;
        y_hat -= hm->a_coeffs[i] * yp;
    }
    return y_hat;
}

int hammerstein_nparams(const HammersteinModel* hm) {
    return hm ? hm->n_params_total : 0;
}

void hammerstein_get_params(const HammersteinModel* hm, double* theta) {
    if (!hm || !theta) return;
    int idx = 0;
    if (hm->static_nonlinearity) {
        int ng = basis_expansion_nparams(hm->static_nonlinearity);
        basis_expansion_pack_params(hm->static_nonlinearity, &theta[idx]);
        idx += ng;
    }
    theta[idx++] = hm->a_coeffs[0];
    for (int i = 1; i <= hm->na; i++) theta[idx++] = hm->a_coeffs[i];
    for (int i = 0; i <= hm->nb; i++) theta[idx++] = hm->b_coeffs[i];
}

void hammerstein_set_params(HammersteinModel* hm, const double* theta) {
    if (!hm || !theta) return;
    int idx = 0;
    if (hm->static_nonlinearity) {
        int ng = basis_expansion_nparams(hm->static_nonlinearity);
        basis_expansion_unpack_params(hm->static_nonlinearity, &theta[idx]);
        idx += ng;
    }
    hm->a_coeffs[0] = theta[idx++];
    for (int i = 1; i <= hm->na; i++) hm->a_coeffs[i] = theta[idx++];
    for (int i = 0; i <= hm->nb; i++) hm->b_coeffs[i] = theta[idx++];
}

void hammerstein_print(const HammersteinModel* hm) {
    if (!hm) { printf("HammersteinModel: NULL\n"); return; }
    printf("Hammerstein Model: na=%d, nb=%d, nk=%d\n", hm->na, hm->nb, hm->nk);
    printf("  Total params: %d\n", hm->n_params_total);
    if (hm->static_nonlinearity) basis_expansion_print(hm->static_nonlinearity);
}

/* ============================================================================
 * Part 4: Wiener Model
 * ============================================================================ */

WienerModel* wiener_create(int na, int nb, int nk) {
    WienerModel* wm = (WienerModel*)calloc(1, sizeof(WienerModel));
    if (!wm) return NULL;
    wm->na = na; wm->nb = nb; wm->nk = nk;
    wm->a_coeffs = (double*)calloc((size_t)(na + 1), sizeof(double));
    wm->b_coeffs = (double*)calloc((size_t)(nb + 1), sizeof(double));
    if (wm->a_coeffs) wm->a_coeffs[0] = 1.0;
    if (wm->b_coeffs) wm->b_coeffs[0] = 1.0;
    wm->static_nonlinearity = NULL;
    wm->h_params = NULL; wm->n_h_params = 0;
    wm->n_params_total = na + nb + 1;
    return wm;
}

void wiener_free(WienerModel* wm) {
    if (!wm) return;
    basis_expansion_free(wm->static_nonlinearity);
    free(wm->h_params); free(wm->a_coeffs);
    free(wm->b_coeffs); free(wm->x_intermediate);
    free(wm);
}

void wiener_set_linear_part(WienerModel* wm, const double* a, const double* b) {
    if (!wm) return;
    if (a) memcpy(wm->a_coeffs, a, (size_t)(wm->na + 1) * sizeof(double));
    if (b) memcpy(wm->b_coeffs, b, (size_t)(wm->nb + 1) * sizeof(double));
}

void wiener_set_static_nl(WienerModel* wm, BasisExpansion* nl) {
    if (!wm) return;
    if (wm->static_nonlinearity) basis_expansion_free(wm->static_nonlinearity);
    wm->static_nonlinearity = nl;
    wm->n_h_params = basis_expansion_nparams(nl);
    wm->n_params_total = wm->n_h_params + wm->na + wm->nb + 1;
}

void wiener_compute_x(const WienerModel* wm, const double* u, int n,
                       double* x_out) {
    if (!wm || !u || !x_out) return;
    int max_order = (wm->na > wm->nb) ? wm->na : wm->nb;
    double* x_buf = (double*)calloc((size_t)(n + max_order), sizeof(double));
    if (!x_buf) return;
    for (int t = 0; t < n; t++) {
        int idx = max_order + t;
        double sum = 0.0;
        for (int i = 0; i <= wm->nb; i++)
            sum += wm->b_coeffs[i] * ((t - wm->nk - i >= 0) ? u[t - wm->nk - i] : 0.0);
        for (int i = 1; i <= wm->na; i++)
            sum -= wm->a_coeffs[i] * x_buf[idx - i];
        x_buf[idx] = sum;
        x_out[t] = sum;
    }
    free(x_buf);
}

int wiener_simulate(const WienerModel* wm, const double* input, int n_steps,
                     const double* x0, double* y_pred) {
    if (!wm || !input || !y_pred || n_steps <= 0) return -1;
    double* x = (double*)calloc((size_t)n_steps, sizeof(double));
    if (!x) return -1;
    wiener_compute_x(wm, input, n_steps, x);
    if (wm->static_nonlinearity) {
        for (int t = 0; t < n_steps; t++)
            y_pred[t] = basis_expansion_eval(wm->static_nonlinearity, &x[t]);
    } else {
        memcpy(y_pred, x, (size_t)n_steps * sizeof(double));
    }
    free(x);
    return 0;
}

int wiener_nparams(const WienerModel* wm) {
    return wm ? wm->n_params_total : 0;
}

void wiener_get_params(const WienerModel* wm, double* theta) {
    if (!wm || !theta) return;
    int idx = 0;
    theta[idx++] = wm->a_coeffs[0];
    for (int i = 1; i <= wm->na; i++) theta[idx++] = wm->a_coeffs[i];
    for (int i = 0; i <= wm->nb; i++) theta[idx++] = wm->b_coeffs[i];
    if (wm->static_nonlinearity)
        basis_expansion_pack_params(wm->static_nonlinearity, &theta[idx]);
}

void wiener_set_params(WienerModel* wm, const double* theta) {
    if (!wm || !theta) return;
    int idx = 0;
    wm->a_coeffs[0] = theta[idx++];
    for (int i = 1; i <= wm->na; i++) wm->a_coeffs[i] = theta[idx++];
    for (int i = 0; i <= wm->nb; i++) wm->b_coeffs[i] = theta[idx++];
    if (wm->static_nonlinearity)
        basis_expansion_unpack_params(wm->static_nonlinearity, &theta[idx]);
}

void wiener_print(const WienerModel* wm) {
    if (!wm) { printf("WienerModel: NULL\n"); return; }
    printf("Wiener Model: na=%d, nb=%d, nk=%d\n", wm->na, wm->nb, wm->nk);
    if (wm->static_nonlinearity) basis_expansion_print(wm->static_nonlinearity);
}

/* ============================================================================
 * Part 5: Volterra Model
 * ============================================================================ */

VolterraModel* volterra_create(int order, int memory, int input_dim) {
    if (order < 1 || order > 3 || memory < 1) return NULL;
    VolterraModel* vm = (VolterraModel*)calloc(1, sizeof(VolterraModel));
    if (!vm) return NULL;
    vm->order = order; vm->memory = memory; vm->input_dim = input_dim;
    vm->kernel_h1 = (double*)calloc((size_t)memory, sizeof(double));
    if (order >= 2)
        vm->kernel_h2 = (double*)calloc((size_t)(memory * memory), sizeof(double));
    if (order >= 3)
        vm->kernel_h3 = (double*)calloc((size_t)(memory * memory * memory), sizeof(double));
    int nk = memory;
    if (order >= 2) nk += memory * memory;
    if (order >= 3) nk += memory * memory * memory;
    vm->n_kernels = nk;
    return vm;
}

void volterra_free(VolterraModel* vm) {
    if (!vm) return;
    free(vm->kernel_h0); free(vm->kernel_h1);
    free(vm->kernel_h2); free(vm->kernel_h3);
    free(vm);
}

double volterra_eval(const VolterraModel* vm, const double* u_hist) {
    if (!vm || !u_hist) return 0.0;
    double y = vm->kernel_h0 ? *vm->kernel_h0 : 0.0;
    int M = vm->memory;
    for (int t1 = 0; t1 < M; t1++)
        y += vm->kernel_h1[t1] * u_hist[t1];
    if (vm->order >= 2)
        for (int t1 = 0; t1 < M; t1++)
            for (int t2 = 0; t2 < M; t2++)
                y += vm->kernel_h2[t1 * M + t2] * u_hist[t1] * u_hist[t2];
    if (vm->order >= 3)
        for (int t1 = 0; t1 < M; t1++)
            for (int t2 = 0; t2 < M; t2++)
                for (int t3 = 0; t3 < M; t3++)
                    y += vm->kernel_h3[(t1 * M + t2) * M + t3]
                         * u_hist[t1] * u_hist[t2] * u_hist[t3];
    return y;
}

int volterra_simulate(const VolterraModel* vm, const double* input,
                       int n_steps, double* output) {
    if (!vm || !input || !output || n_steps <= 0) return -1;
    for (int t = 0; t < n_steps; t++) {
        double uh[64];
        int M = (vm->memory < 64) ? vm->memory : 64;
        for (int k = 0; k < M; k++) {
            int idx = t - k;
            uh[k] = (idx >= 0) ? input[idx] : 0.0;
        }
        output[t] = volterra_eval(vm, uh);
    }
    return 0;
}

int volterra_nparams(const VolterraModel* vm) {
    return vm ? vm->n_kernels : 0;
}

void volterra_get_params(const VolterraModel* vm, double* theta) {
    if (!vm || !theta) return;
    int idx = 0, M = vm->memory;
    if (vm->kernel_h0) theta[idx++] = *vm->kernel_h0;
    for (int i = 0; i < M; i++) theta[idx++] = vm->kernel_h1[i];
    if (vm->order >= 2)
        for (int i = 0; i < M*M; i++) theta[idx++] = vm->kernel_h2[i];
    if (vm->order >= 3)
        for (int i = 0; i < M*M*M; i++) theta[idx++] = vm->kernel_h3[i];
}

void volterra_set_params(VolterraModel* vm, const double* theta) {
    if (!vm || !theta) return;
    int idx = 0, M = vm->memory;
    if (vm->kernel_h0) *vm->kernel_h0 = theta[idx++];
    for (int i = 0; i < M; i++) vm->kernel_h1[i] = theta[idx++];
    if (vm->order >= 2)
        for (int i = 0; i < M*M; i++) vm->kernel_h2[i] = theta[idx++];
    if (vm->order >= 3)
        for (int i = 0; i < M*M*M; i++) vm->kernel_h3[i] = theta[idx++];
}

void volterra_print(const VolterraModel* vm) {
    if (!vm) { printf("VolterraModel: NULL\n"); return; }
    printf("Volterra Model: order=%d, memory=%d, kernels=%d\n",
           vm->order, vm->memory, vm->n_kernels);
}

/* ============================================================================
 * Part 6: Bilinear State-Space Model
 * ============================================================================ */

BilinearModel* bilinear_create(int n_states, int n_inputs, int n_outputs) {
    BilinearModel* bm = (BilinearModel*)calloc(1, sizeof(BilinearModel));
    if (!bm) return NULL;
    int n = n_states, m = n_inputs, p = n_outputs;
    bm->n_states = n; bm->n_inputs = m; bm->n_outputs = p;
    bm->A = (double*)calloc((size_t)(n*n), sizeof(double));
    bm->B = (double*)calloc((size_t)(n*m), sizeof(double));
    bm->C = (double*)calloc((size_t)(p*n), sizeof(double));
    bm->D = (double*)calloc((size_t)(p*m), sizeof(double));
    bm->N = (double**)calloc((size_t)m, sizeof(double*));
    for (int i = 0; i < m; i++)
        bm->N[i] = (double*)calloc((size_t)(n*n), sizeof(double));
    bm->n_params_total = n*n + n*m + p*n + m*n*n + p*m;
    return bm;
}

void bilinear_free(BilinearModel* bm) {
    if (!bm) return;
    free(bm->A); free(bm->B); free(bm->C); free(bm->D);
    if (bm->N) {
        for (int i = 0; i < bm->n_inputs; i++) free(bm->N[i]);
        free(bm->N);
    }
    free(bm);
}

void bilinear_set_A(BilinearModel* bm, const double* A) {
    if (bm && A) memcpy(bm->A, A, (size_t)(bm->n_states*bm->n_states)*sizeof(double));
}
void bilinear_set_B(BilinearModel* bm, const double* B) {
    if (bm && B) memcpy(bm->B, B, (size_t)(bm->n_states*bm->n_inputs)*sizeof(double));
}
void bilinear_set_C(BilinearModel* bm, const double* C) {
    if (bm && C) memcpy(bm->C, C, (size_t)(bm->n_outputs*bm->n_states)*sizeof(double));
}
void bilinear_set_N(BilinearModel* bm, int k, const double* Nk) {
    if (bm && Nk && k>=0 && k<bm->n_inputs)
        memcpy(bm->N[k], Nk, (size_t)(bm->n_states*bm->n_states)*sizeof(double));
}
void bilinear_set_D(BilinearModel* bm, const double* D) {
    if (bm && D) memcpy(bm->D, D, (size_t)(bm->n_outputs*bm->n_inputs)*sizeof(double));
}

int bilinear_simulate(const BilinearModel* bm, const double* input,
                       int n_steps, const double* x0, double* y) {
    if (!bm || !input || !y || n_steps <= 0) return -1;
    int n = bm->n_states, m = bm->n_inputs, p = bm->n_outputs;
    double* x = (double*)calloc((size_t)n, sizeof(double));
    double* xn = (double*)calloc((size_t)n, sizeof(double));
    if (!x || !xn) { free(x); free(xn); return -1; }
    if (x0) memcpy(x, x0, (size_t)n * sizeof(double));

    for (int t = 0; t < n_steps; t++) {
        /* Output */
        for (int i = 0; i < p; i++) {
            y[t*p + i] = 0.0;
            for (int j = 0; j < n; j++) y[t*p + i] += bm->C[i*n + j] * x[j];
            for (int j = 0; j < m; j++) y[t*p + i] += bm->D[i*m + j] * input[t*m + j];
        }
        /* Next state: x(t+1) = A x + sum(N_k x u_k) + B u */
        for (int i = 0; i < n; i++) xn[i] = 0.0;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                xn[i] += bm->A[i*n + j] * x[j];
        for (int k = 0; k < m; k++) {
            double uk = input[t*m + k];
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    xn[i] += bm->N[k][i*n + j] * x[j] * uk;
        }
        for (int i = 0; i < n; i++)
            for (int j = 0; j < m; j++)
                xn[i] += bm->B[i*m + j] * input[t*m + j];
        memcpy(x, xn, (size_t)n * sizeof(double));
    }
    free(x); free(xn);
    return 0;
}

int bilinear_nparams(const BilinearModel* bm) { return bm ? bm->n_params_total : 0; }

void bilinear_get_params(const BilinearModel* bm, double* theta) {
    if (!bm || !theta) return;
    int n=bm->n_states, m=bm->n_inputs, p=bm->n_outputs, idx=0;
    memcpy(theta+idx, bm->A, (size_t)(n*n)*sizeof(double)); idx+=n*n;
    memcpy(theta+idx, bm->B, (size_t)(n*m)*sizeof(double)); idx+=n*m;
    memcpy(theta+idx, bm->C, (size_t)(p*n)*sizeof(double)); idx+=p*n;
    for(int k=0;k<m;k++){memcpy(theta+idx,bm->N[k],(size_t)(n*n)*sizeof(double));idx+=n*n;}
    memcpy(theta+idx, bm->D, (size_t)(p*m)*sizeof(double));
}

void bilinear_set_params(BilinearModel* bm, const double* theta) {
    if (!bm || !theta) return;
    int n=bm->n_states, m=bm->n_inputs, p=bm->n_outputs, idx=0;
    memcpy(bm->A, theta+idx, (size_t)(n*n)*sizeof(double)); idx+=n*n;
    memcpy(bm->B, theta+idx, (size_t)(n*m)*sizeof(double)); idx+=n*m;
    memcpy(bm->C, theta+idx, (size_t)(p*n)*sizeof(double)); idx+=p*n;
    for(int k=0;k<m;k++){memcpy(bm->N[k],theta+idx,(size_t)(n*n)*sizeof(double));idx+=n*n;}
    memcpy(bm->D, theta+idx, (size_t)(p*m)*sizeof(double));
}

void bilinear_print(const BilinearModel* bm) {
    if (!bm) { printf("BilinearModel: NULL\n"); return; }
    printf("Bilinear SS: nx=%d, nu=%d, ny=%d, params=%d\n",
           bm->n_states, bm->n_inputs, bm->n_outputs, bm->n_params_total);
}
/* ============================================================================
 * Part 7: Neural Network Model
 * ============================================================================ */

double activation_eval(double x, ActivationType act) {
    switch (act) {
        case ACTIVATION_TANH: return tanh(x);
        case ACTIVATION_SIGMOID:
            if (x > 50.0) return 1.0;
            if (x < -50.0) return 0.0;
            return 1.0 / (1.0 + exp(-x));
        case ACTIVATION_RELU: return (x > 0.0) ? x : 0.0;
        case ACTIVATION_LEAKY_RELU: return (x > 0.0) ? x : 0.01 * x;
        case ACTIVATION_SWISH: return x / (1.0 + exp(-x));
        default: return x;
    }
}

double activation_derivative(double x, ActivationType act) {
    switch (act) {
        case ACTIVATION_TANH: { double t = tanh(x); return 1.0 - t*t; }
        case ACTIVATION_SIGMOID: {
            double s = activation_eval(x, ACTIVATION_SIGMOID);
            return s * (1.0 - s);
        }
        case ACTIVATION_RELU: return (x > 0.0) ? 1.0 : 0.0;
        case ACTIVATION_LEAKY_RELU: return (x > 0.0) ? 1.0 : 0.01;
        case ACTIVATION_SWISH: {
            double s = activation_eval(x, ACTIVATION_SIGMOID);
            return s + x * s * (1.0 - s);
        }
        default: return 1.0;
    }
}

NeuralNetModel* neuralnet_create(int n_layers, const int* layer_sizes,
                                  const ActivationType* activations) {
    if (n_layers < 2 || !layer_sizes) return NULL;
    NeuralNetModel* nn = (NeuralNetModel*)calloc(1, sizeof(NeuralNetModel));
    if (!nn) return NULL;
    nn->n_layers = n_layers;
    nn->layer_sizes = (int*)calloc((size_t)n_layers, sizeof(int));
    nn->activations = (ActivationType*)calloc((size_t)n_layers, sizeof(ActivationType));
    memcpy(nn->layer_sizes, layer_sizes, (size_t)n_layers * sizeof(int));
    if (activations)
        memcpy(nn->activations, activations, (size_t)n_layers * sizeof(ActivationType));
    nn->weights = (double**)calloc((size_t)(n_layers - 1), sizeof(double*));
    nn->biases = (double**)calloc((size_t)(n_layers - 1), sizeof(double*));
    nn->n_params_total = 0;
    for (int l = 0; l < n_layers - 1; l++) {
        int r = layer_sizes[l + 1], c = layer_sizes[l];
        nn->weights[l] = (double*)calloc((size_t)(r * c), sizeof(double));
        nn->biases[l] = (double*)calloc((size_t)r, sizeof(double));
        nn->n_params_total += r * c + r;
    }
    nn->ny = 0; nn->nu = 0; nn->nk = 0;
    nn->regressor_dim = layer_sizes[0];
    return nn;
}

void neuralnet_free(NeuralNetModel* nn) {
    if (!nn) return;
    free(nn->layer_sizes); free(nn->activations);
    for (int l = 0; l < nn->n_layers - 1; l++) {
        free(nn->weights[l]); free(nn->biases[l]);
    }
    free(nn->weights); free(nn->biases);
    free(nn);
}

int neuralnet_forward(const NeuralNetModel* nn, const double* input,
                       double* output) {
    if (!nn || !input || !output) return -1;
    int max_size = 0;
    for (int l = 0; l < nn->n_layers; l++)
        if (nn->layer_sizes[l] > max_size) max_size = nn->layer_sizes[l];
    double* curr = (double*)calloc((size_t)max_size, sizeof(double));
    double* next = (double*)calloc((size_t)max_size, sizeof(double));
    if (!curr || !next) { free(curr); free(next); return -1; }
    memcpy(curr, input, (size_t)nn->layer_sizes[0] * sizeof(double));
    for (int l = 0; l < nn->n_layers - 1; l++) {
        int cols = nn->layer_sizes[l], rows = nn->layer_sizes[l + 1];
        for (int i = 0; i < rows; i++) {
            double sum = nn->biases[l][i];
            for (int j = 0; j < cols; j++)
                sum += nn->weights[l][i * cols + j] * curr[j];
            next[i] = activation_eval(sum, nn->activations[l + 1]);
        }
        double* tmp = curr; curr = next; next = tmp;
    }
    memcpy(output, curr, (size_t)nn->layer_sizes[nn->n_layers - 1] * sizeof(double));
    free(curr); free(next);
    return 0;
}

void neuralnet_forward_batch(const NeuralNetModel* nn,
                              const double** inputs, int n_samples,
                              double** outputs) {
    if (!nn || !inputs || !outputs) return;
    for (int s = 0; s < n_samples; s++)
        neuralnet_forward(nn, inputs[s], outputs[s]);
}

void neuralnet_set_sysid_regressors(NeuralNetModel* nn, int ny, int nu, int nk) {
    if (!nn) return;
    nn->ny = ny; nn->nu = nu; nn->nk = nk;
    nn->regressor_dim = ny + nu;
}

int neuralnet_simulate(const NeuralNetModel* nn, const double* input,
                        int n_steps, const double* y0, double* y_pred) {
    if (!nn || !input || !y_pred || n_steps <= 0) return -1;
    int ny = nn->ny, nu = nn->nu, nk = nn->nk;
    int dim = ny + nu;
    for (int t = 0; t < n_steps; t++) {
        double reg[50];
        for (int i = 0; i < dim && i < 50; i++) {
            if (i < ny) {
                if (t - 1 - i >= 0) reg[i] = y_pred[t - 1 - i];
                else if (y0) reg[i] = y0[i];
                else reg[i] = 0.0;
            } else {
                int idx = t - nk - (i - ny);
                reg[i] = (idx >= 0) ? input[idx] : 0.0;
            }
        }
        neuralnet_forward(nn, reg, &y_pred[t]);
    }
    return 0;
}

double neuralnet_predict_one_step(const NeuralNetModel* nn,
                                   const double* y_hist,
                                   const double* u_hist, int t) {
    if (!nn) return 0.0;
    int ny = nn->ny, nu = nn->nu, nk = nn->nk;
    double reg[50]; int dim = ny + nu;
    for (int i = 0; i < dim && i < 50; i++) {
        if (i < ny) {
            int idx = t - 1 - i;
            reg[i] = (idx >= 0 && y_hist) ? y_hist[idx] : 0.0;
        } else {
            int idx = t - nk - (i - ny);
            reg[i] = (idx >= 0 && u_hist) ? u_hist[idx] : 0.0;
        }
    }
    double out;
    neuralnet_forward(nn, reg, &out);
    return out;
}

int neuralnet_nparams(const NeuralNetModel* nn) {
    return nn ? nn->n_params_total : 0;
}

void neuralnet_get_params(const NeuralNetModel* nn, double* theta) {
    if (!nn || !theta) return;
    int idx = 0;
    for (int l = 0; l < nn->n_layers - 1; l++) {
        int nw = nn->layer_sizes[l + 1] * nn->layer_sizes[l];
        int nb = nn->layer_sizes[l + 1];
        memcpy(theta + idx, nn->weights[l], (size_t)nw * sizeof(double));
        idx += nw;
        memcpy(theta + idx, nn->biases[l], (size_t)nb * sizeof(double));
        idx += nb;
    }
}

void neuralnet_set_params(NeuralNetModel* nn, const double* theta) {
    if (!nn || !theta) return;
    int idx = 0;
    for (int l = 0; l < nn->n_layers - 1; l++) {
        int nw = nn->layer_sizes[l + 1] * nn->layer_sizes[l];
        int nb = nn->layer_sizes[l + 1];
        memcpy(nn->weights[l], theta + idx, (size_t)nw * sizeof(double));
        idx += nw;
        memcpy(nn->biases[l], theta + idx, (size_t)nb * sizeof(double));
        idx += nb;
    }
}

int neuralnet_nparams_total(const NeuralNetModel* nn) {
    return nn ? nn->n_params_total : 0;
}

void neuralnet_print(const NeuralNetModel* nn) {
    if (!nn) { printf("NeuralNet: NULL\n"); return; }
    printf("Neural Network: %d layers, architecture: [", nn->n_layers);
    for (int l = 0; l < nn->n_layers; l++)
        printf("%d%s", nn->layer_sizes[l], l < nn->n_layers - 1 ? ", " : "");
    printf("], params=%d\n", nn->n_params_total);
    printf("  Regressors: ny=%d, nu=%d, nk=%d\n", nn->ny, nn->nu, nn->nk);
}

/* ============================================================================
 * Part 8: Model Factory Functions
 * These provide convenient creation of NLSIDModel wrappers around
 * the specific model implementations above.
 * ============================================================================ */

/* Forward declare the simulate wrappers used by model factory */
static int narx_model_simulate(NLSIDModel* model, const double* input,
                                int n_steps, double* output_pred) {
    NARXModel* narx = (NARXModel*)model->specific;
    return narx_simulate(narx, input, n_steps, NULL, output_pred);
}

static int narx_model_get_params(NLSIDModel* model, double* params) {
    NARXModel* narx = (NARXModel*)model->specific;
    narx_get_params(narx, params);
    return 0;
}

static int narx_model_set_params(NLSIDModel* model, const double* params) {
    NARXModel* narx = (NARXModel*)model->specific;
    narx_set_params(narx, params);
    return 0;
}

static void narx_model_free(NLSIDModel* model) {
    NARXModel* narx = (NARXModel*)model->specific;
    narx_free(narx);
    free(model->name);
    free(model);
}

NLSIDModel* nlsid_model_create_narx(int ny, int nu, int nk,
                                     int n_inputs, int n_outputs,
                                     const double* theta, int n_params) {
    NARXModel* narx = narx_create(ny, nu, nk, n_inputs, n_outputs);
    if (!narx) return NULL;
    NLSIDModel* model = (NLSIDModel*)calloc(1, sizeof(NLSIDModel));
    if (!model) { narx_free(narx); return NULL; }
    model->type = NLSID_MODEL_NARX;
    model->name = strdup("NARX");
    model->n_params = narx_nparams(narx);
    model->specific = narx;
    model->simulate = narx_model_simulate;
    model->get_params = narx_model_get_params;
    model->set_params = narx_model_set_params;
    model->free = narx_model_free;
    if (theta && n_params > 0) narx_set_params(narx, theta);
    return model;
}