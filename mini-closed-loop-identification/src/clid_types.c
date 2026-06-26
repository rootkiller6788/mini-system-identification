/**
 * clid_types.c — Memory management operations for closed-loop identification types
 *
 * Provides allocation, initialization, and deallocation functions for
 * all core data structures defined in clid_types.h.  Every allocation
 * zeros the memory (calloc) to ensure deterministic initial state.
 *
 * References:
 *   Ljung (1999) "System Identification: Theory for the User"
 */
#include "clid_types.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ─── Transfer Function ─────────────────────────────────────────────── */

CLID_TransferFcn clid_tf_alloc(int na, int nb, int nk, double Ts)
{
    CLID_TransferFcn tf;
    memset(&tf, 0, sizeof(tf));
    tf.na = na;
    tf.nb = nb;
    tf.nk = nk;
    tf.Ts = Ts;
    if (na > 0) {
        tf.a = (double *)calloc((size_t)(na + 1), sizeof(double));
        if (tf.a) tf.a[0] = 1.0;  /* a[0] = 1 by convention (monic) */
    }
    if (nb > 0) {
        tf.b = (double *)calloc((size_t)nb, sizeof(double));
    }
    return tf;
}

void clid_tf_free(CLID_TransferFcn *tf)
{
    if (!tf) return;
    if (tf->a) { free(tf->a); tf->a = NULL; }
    if (tf->b) { free(tf->b); tf->b = NULL; }
    tf->na = tf->nb = tf->nk = 0;
}

/* ─── State-Space Model ──────────────────────────────────────────────── */

CLID_StateSpace clid_ss_alloc(int nx, int nu, int ny, double Ts)
{
    CLID_StateSpace ss;
    memset(&ss, 0, sizeof(ss));
    ss.nx = nx; ss.nu = nu; ss.ny = ny; ss.Ts = Ts;
    if (nx > 0) {
        ss.A = (double *)calloc((size_t)(nx * nx), sizeof(double));
        /* Initialize A to identity for stable initialization */
        if (ss.A) {
            for (int i = 0; i < nx; i++) ss.A[i * nx + i] = 1.0;
        }
        if (nu > 0) ss.B = (double *)calloc((size_t)(nx * nu), sizeof(double));
        if (ny > 0) {
            ss.C = (double *)calloc((size_t)(ny * nx), sizeof(double));
            ss.K = (double *)calloc((size_t)(nx * ny), sizeof(double));
            ss.Lambda = (double *)calloc((size_t)(ny * ny), sizeof(double));
        }
        if (nu > 0 && ny > 0)
            ss.D = (double *)calloc((size_t)(ny * nu), sizeof(double));
    }
    return ss;
}

void clid_ss_free(CLID_StateSpace *ss)
{
    if (!ss) return;
    if (ss->A)      { free(ss->A);      ss->A      = NULL; }
    if (ss->B)      { free(ss->B);      ss->B      = NULL; }
    if (ss->C)      { free(ss->C);      ss->C      = NULL; }
    if (ss->D)      { free(ss->D);      ss->D      = NULL; }
    if (ss->K)      { free(ss->K);      ss->K      = NULL; }
    if (ss->Lambda) { free(ss->Lambda); ss->Lambda = NULL; }
    ss->nx = ss->nu = ss->ny = 0;
}

/* ─── Dataset ────────────────────────────────────────────────────────── */

CLID_Dataset clid_data_alloc(int N, int nu, int ny, int nr, double Ts)
{
    CLID_Dataset ds;
    memset(&ds, 0, sizeof(ds));
    ds.N = N; ds.nu = nu; ds.ny = ny; ds.nr = nr; ds.Ts = Ts;
    ds.under_feedback = 1;   /* Default: closed-loop data */
    ds.controller_knowledge = 0;  /* Unknown by default */
    if (N > 0) {
        if (nu > 0) ds.u = (double *)calloc((size_t)(N * nu), sizeof(double));
        if (ny > 0) ds.y = (double *)calloc((size_t)(N * ny), sizeof(double));
        if (nr > 0) ds.r = (double *)calloc((size_t)(N * nr), sizeof(double));
    }
    return ds;
}

void clid_data_free(CLID_Dataset *data)
{
    if (!data) return;
    if (data->u) { free(data->u); data->u = NULL; }
    if (data->y) { free(data->y); data->y = NULL; }
    if (data->r) { free(data->r); data->r = NULL; }
    data->N = data->nu = data->ny = data->nr = 0;
}

/* ─── Estimate ───────────────────────────────────────────────────────── */

CLID_Estimate clid_estimate_alloc(int n_params)
{
    CLID_Estimate est;
    memset(&est, 0, sizeof(est));
    est.model_type = CLID_MODEL_ARX;
    est.is_state_space = 0;
    est.n_params = n_params;
    if (n_params > 0) {
        est.param_cov = (double *)calloc((size_t)(n_params * n_params),
                                          sizeof(double));
    }
    return est;
}

void clid_estimate_free(CLID_Estimate *est)
{
    if (!est) return;
    if (!est->is_state_space) {
        clid_tf_free(&est->identified_model.tf);
    } else {
        clid_ss_free(&est->identified_model.ss);
    }
    if (est->noise_model) { free(est->noise_model); est->noise_model = NULL; }
    if (est->param_cov)   { free(est->param_cov);   est->param_cov   = NULL; }
    est->n_params = 0;
}

/* ─── Default Options ────────────────────────────────────────────────── */

CLID_Options clid_options_default(void)
{
    CLID_Options opts;
    memset(&opts, 0, sizeof(opts));
    opts.method       = CLID_METHOD_DIRECT;
    opts.plant_model  = CLID_MODEL_ARMAX;
    opts.noise_model  = CLID_NOISE_MA;
    opts.na_min       = 1;    opts.na_max = 10;
    opts.nb_min       = 1;    opts.nb_max = 10;
    opts.nk           = 1;
    opts.max_iter     = 50;
    opts.tolerance    = 1e-6;
    opts.use_instrument = 0;
    opts.instrument_lag = 5;
    opts.prefilter    = 0;
    opts.prefilter_order = 0;
    opts.prefilter_num = NULL;
    opts.prefilter_den = NULL;
    return opts;
}

/* ─── Prediction Error ───────────────────────────────────────────────── */

CLID_PredictionError clid_pe_alloc(int N)
{
    CLID_PredictionError pe;
    memset(&pe, 0, sizeof(pe));
    pe.N = N;
    if (N > 0) {
        pe.epsilon = (double *)calloc((size_t)N, sizeof(double));
        pe.y_hat   = (double *)calloc((size_t)N, sizeof(double));
    }
    return pe;
}

void clid_pe_free(CLID_PredictionError *pe)
{
    if (!pe) return;
    if (pe->epsilon) { free(pe->epsilon); pe->epsilon = NULL; }
    if (pe->y_hat)   { free(pe->y_hat);   pe->y_hat   = NULL; }
    pe->N = 0;
}

/* ─── Frequency Response ─────────────────────────────────────────────── */

CLID_FrequencyResponse clid_fr_alloc(int n_freqs)
{
    CLID_FrequencyResponse fr;
    memset(&fr, 0, sizeof(fr));
    fr.n_freqs = n_freqs;
    if (n_freqs > 0) {
        fr.omega  = (double *)calloc((size_t)n_freqs, sizeof(double));
        fr.mag    = (double *)calloc((size_t)n_freqs, sizeof(double));
        fr.phase  = (double *)calloc((size_t)n_freqs, sizeof(double));
        fr.mag_db = (double *)calloc((size_t)n_freqs, sizeof(double));
    }
    return fr;
}

void clid_fr_free(CLID_FrequencyResponse *fr)
{
    if (!fr) return;
    if (fr->omega)  { free(fr->omega);  fr->omega  = NULL; }
    if (fr->mag)    { free(fr->mag);    fr->mag    = NULL; }
    if (fr->phase)  { free(fr->phase);  fr->phase  = NULL; }
    if (fr->mag_db) { free(fr->mag_db); fr->mag_db = NULL; }
    fr->n_freqs = 0;
}

/* ─── Uncertainty Region ─────────────────────────────────────────────── */

CLID_UncertaintyRegion clid_ur_alloc(int n_params, double confidence)
{
    CLID_UncertaintyRegion ur;
    memset(&ur, 0, sizeof(ur));
    ur.n_params   = n_params;
    ur.confidence = confidence;
    if (n_params > 0) {
        ur.center = (double *)calloc((size_t)n_params, sizeof(double));
        ur.P_inv  = (double *)calloc((size_t)(n_params * n_params),
                                      sizeof(double));
    }
    return ur;
}

void clid_ur_free(CLID_UncertaintyRegion *ur)
{
    if (!ur) return;
    if (ur->center) { free(ur->center); ur->center = NULL; }
    if (ur->P_inv)  { free(ur->P_inv);  ur->P_inv  = NULL; }
    ur->n_params = 0;
}
