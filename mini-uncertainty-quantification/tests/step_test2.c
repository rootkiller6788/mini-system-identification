#include "uq_core.h"
#include "uq_bayesian.h"
#include "uq_sampling.h"
#include "uq_propagation.h"
#include "uq_sensitivity.h"
#include "uq_validation.h"
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <stdlib.h>
#define EPS 1e-9
#define STEP(n) printf("Step %d...\n", n); fflush(stdout)
static double f_test(double* x, void* d) { (void)d; return x[0]*x[0]; }
static double s_model(double* x, void* d) { (void)d; return 2.0*x[0] + 0.5*x[1]*x[1]; }
int main(void) {
    STEP(20);
    UQDistribution* norm = uq_dist_create_normal(0.0, 1.0);
    assert(norm != NULL);
    STEP(21);
    double samples[1000];
    uq_dist_sample_n(norm, 1000, samples);
    STEP(22);
    UQMatrix* X_lm = uq_matrix_create(10, 2);
    UQVector* y_lm = uq_vector_create(10);
    for (int i = 0; i < 10; i++) { uq_matrix_set(X_lm, i, 0, 1.0); uq_matrix_set(X_lm, i, 1, (double)i); y_lm->components[i] = 2.0 + 3.0 * i; }
    UQLinearModel* lm = uq_lm_create(X_lm, y_lm); uq_lm_fit(lm);
    STEP(23);
    int n_lhs = 100; double** lhs_out = (double**)malloc(n_lhs * sizeof(double*));
    for (int i = 0; i < n_lhs; i++) lhs_out[i] = (double*)malloc(2 * sizeof(double));
    uq_lhs_generate(n_lhs, 2, lhs_out, true);
    for (int i = 0; i < n_lhs; i++) free(lhs_out[i]); free(lhs_out);
    STEP(24);
    double data_test[] = {1,2,3,4,5,6,7,8,9,10};
    UQBootstrap* bs = uq_bootstrap_create(UQ_BOOTSTRAP_STANDARD, 10, 100, 1);
    uq_bootstrap_set_data(bs, data_test); uq_bootstrap_generate_replicates(bs); uq_bootstrap_free(bs);
    STEP(25);
    UQGaussianProcess* gp = uq_gp_create(UQ_GP_KERNEL_SQUARED_EXPONENTIAL, 1, 3);
    double Xg[] = {1,2,3}, Yg[] = {1.1,1.9,3.1};
    uq_gp_set_data(gp, Xg, Yg); uq_gp_train(gp); uq_gp_free(gp);
    STEP(26);
    double y_mean, y_std;
    uq_rosenblueth_2p(f_test, NULL, (double[]){0}, (double[]){2}, 1, &y_mean, &y_std);
    STEP(27);
    UQUnscentedTransform* ut = uq_ut_create(1, 1.0, 2.0, 0.0); assert(ut->n_sigma == 3); uq_ut_free(ut);
    STEP(28);
    UQPCE* pce = uq_pce_create(UQ_PCE_HERMITE, 1, 2); uq_pce_build_basis(pce); uq_pce_free(pce);
    STEP(29);
    UQSensitivityAnalysis* sa = uq_sa_create(2, NULL); sa->n_samples = 50; uq_sobol_saltelli(sa, s_model, NULL); uq_sa_free(sa);
    STEP(30);
    double obs[] = {1,2,3,4,5}, preds[] = {1.1,2.2,2.9,4.1,5.3};
    UQValidationResult vr = uq_validate_rmse(obs, preds, 5);
    printf("RMSE: %.4f\n", vr.value);
    STEP(31);
    UQPriorDistribution* prior = uq_prior_create(uq_dist_create_normal(0.0, 1.0), true);
    UQLikelihood* lh = uq_likelihood_create(UQ_LIKELIHOOD_NORMAL, 10);
    lh->params.sigma = 1.0; lh->obs_ptr = y_lm->components;
    UQMCMCState* mcmc = uq_mcmc_create(2, 200, 2, NULL);
    double pred_buf[10] = {0};
    uq_mh_sample(mcmc, prior, lh, pred_buf, 2);
    printf("MH acc rate: %.3f\n", uq_mcmc_acceptance_rate(mcmc, 0));
    STEP(32);
    double* r1 = (double[]){100,0}, *r2 = (double[]){20,50};
    double* um[] = {r1, r2}; double pr[] = {0.7, 0.3};
    double evpi = uq_evpi(um, pr, 2, 2);
    printf("EVPI: %.4f\n", evpi);
    uq_dist_free(norm); uq_lm_free(lm); uq_matrix_free(X_lm); uq_vector_free(y_lm);
    uq_prior_free(prior); uq_likelihood_free(lh); uq_mcmc_free(mcmc);
    printf("All step2 tests passed\n");
    return 0;
}
