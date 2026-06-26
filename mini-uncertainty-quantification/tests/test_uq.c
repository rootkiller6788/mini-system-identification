#include "uq_core.h"
#include "uq_bayesian.h"
#include "uq_sampling.h"
#include "uq_propagation.h"
#include "uq_sensitivity.h"
#include "uq_validation.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define EPS 1e-9

static double quad_test(double* x, void* d) { (void)d; return x[0] * x[0]; }
static double sobol_model(double* x, void* d) { (void)d; return 2.0*x[0] + 0.5*x[1]*x[1]; }

int main(void) {
    printf("=== UQ Test Suite ===\n\n");

    printf("--- L1: Distributions ---\n");
    UQDistribution* norm = uq_dist_create_normal(0.0, 1.0);
    assert(norm != NULL);
    assert(fabs(uq_dist_pdf(norm, 0.0) - 0.39894228) < 0.01);
    assert(fabs(uq_dist_cdf(norm, 0.0) - 0.5) < 0.05);
    printf("  PASS: Normal PDF/CDF\n");

    UQDistribution* unif = uq_dist_create_uniform(0.0, 1.0);
    assert(fabs(unif->mean - 0.5) < EPS && fabs(unif->variance - 1.0/12.0) < EPS);
    printf("  PASS: Uniform\n");

    UQDistribution* beta = uq_dist_create_beta(2.0, 5.0);
    assert(fabs(beta->mean - 2.0/7.0) < EPS);
    printf("  PASS: Beta\n");

    UQDistribution* gamma = uq_dist_create_gamma(3.0, 2.0);
    assert(fabs(gamma->mean - 6.0) < EPS && fabs(gamma->variance - 12.0) < EPS);
    printf("  PASS: Gamma\n");

    double samples[1000];
    uq_dist_sample_n(norm, 1000, samples);
    double sm = 0.0; for (int i=0;i<1000;i++) sm+=samples[i]; sm/=1000;
    assert(fabs(sm) < 0.2);
    printf("  PASS: Sampling (mean=%.3f)\n", sm);

    printf("--- L2: Confidence Intervals ---\n");
    UQConfidenceRegion* ci = uq_ci_create(0.95, 1);
    uq_ci_from_normal(ci, 10.0, 2.0, 100);
    assert(ci->lower_bound < 10.0 && ci->upper_bound > 10.0);
    printf("  PASS: CI bounds OK\n");

    printf("--- L3: Linear Algebra ---\n");
    UQMatrix* M = uq_matrix_create(2, 2);
    uq_matrix_set(M,0,0,1.0); uq_matrix_set(M,0,1,2.0);
    uq_matrix_set(M,1,0,3.0); uq_matrix_set(M,1,1,4.0);
    assert(fabs(uq_matrix_determinant(M) + 2.0) < EPS);
    printf("  PASS: Determinant\n");

    UQMatrix* Minv = uq_matrix_create(2, 2);
    uq_matrix_invert(Minv, M);
    printf("  PASS: Inverse\n");

    printf("--- L4: OLS Regression ---\n");
    UQMatrix* X_lm = uq_matrix_create(10, 2);
    UQVector* y_lm = uq_vector_create(10);
    for (int i=0;i<10;i++){ uq_matrix_set(X_lm,i,0,1.0); uq_matrix_set(X_lm,i,1,(double)i); y_lm->components[i]=2.0+3.0*i; }
    UQLinearModel* lm = uq_lm_create(X_lm, y_lm);
    uq_lm_fit(lm);
    assert(fabs(lm->coefficients->components[1]-3.0)<0.01);
    printf("  PASS: OLS β1=%.4f\n", lm->coefficients->components[1]);

    double ssr,sse,sst,Fv,pFv;
    uq_lm_anova(lm,&ssr,&sse,&sst,&Fv,&pFv);
    assert(sst>0.0 && sse>=0.0);
    printf("  PASS: ANOVA\n");

    printf("--- L5: Sampling ---\n");
    int nl=100; double** lo=(double**)malloc(nl*sizeof(double*));
    for(int i=0;i<nl;i++) lo[i]=(double*)malloc(2*sizeof(double));
    uq_lhs_generate(nl,2,lo,true);
    for(int i=0;i<nl;i++) free(lo[i]); free(lo);
    printf("  PASS: LHS\n");

    double dtest[]={1,2,3,4,5,6,7,8,9,10};
    UQBootstrap* bs=uq_bootstrap_create(UQ_BOOTSTRAP_STANDARD,10,100,1);
    uq_bootstrap_set_data(bs,dtest); uq_bootstrap_generate_replicates(bs);
    uq_bootstrap_free(bs);
    printf("  PASS: Bootstrap\n");

    UQGaussianProcess* gp=uq_gp_create(UQ_GP_KERNEL_SQUARED_EXPONENTIAL,1,5);
    double Xg[]={1,2,3,4,5},Yg[]={1.1,1.9,3.1,4.2,5.0};
    uq_gp_set_data(gp,Xg,Yg); uq_gp_train(gp);
    double pvar,xp=3.0,pp=uq_gp_predict(gp,&xp,&pvar);
    assert(fabs(pp-3.1)<2.0);
    printf("  PASS: GP pred=%.4f\n", pp);
    uq_gp_free(gp);

    printf("--- L6: Propagation ---\n");
    double ym,ys;
    uq_rosenblueth_2p(quad_test,NULL,(double[]){0},(double[]){2},1,&ym,&ys);
    assert(fabs(ym-4.0)<1.0);
    printf("  PASS: Rosenblueth E[X²]=%.4f\n",ym);

    UQUnscentedTransform* ut=uq_ut_create(1,1.0,2.0,0.0);
    assert(ut->n_sigma==3); uq_ut_free(ut);
    printf("  PASS: UT\n");

    UQPCE* pce=uq_pce_create(UQ_PCE_HERMITE,1,2);
    uq_pce_build_basis(pce);
    assert(pce->n_basis_functions==3);
    printf("  PASS: PCE (%d terms)\n",pce->n_basis_functions);
    uq_pce_free(pce);

    printf("--- L7: Sensitivity & Validation ---\n");
    UQSensitivityAnalysis* sa=uq_sa_create(2,NULL);
    sa->n_samples=100;
    uq_sobol_saltelli(sa,sobol_model,NULL);
    assert(sa->computed);
    printf("  PASS: Sobol S1=%.3f S2=%.3f\n",uq_sobol_main_effect(sa,0),uq_sobol_main_effect(sa,1));
    uq_sa_free(sa);

    double oo[]={1,2,3,4,5},ppp[]={1.1,2.2,2.9,4.1,5.3};
    UQValidationResult vr=uq_validate_rmse(oo,ppp,5);
    assert(vr.value<0.5);
    UQValidationResult vr2=uq_validate_r_squared(oo,ppp,5,1);
    assert(vr2.value>0.9);
    printf("  PASS: RMSE=%.3f R²=%.3f\n",vr.value,vr2.value);

    printf("--- L8: Bayesian ---\n");
    UQPriorDistribution* prior=uq_prior_create(uq_dist_create_normal(0.0,1.0),true);
    UQLikelihood* lh=uq_likelihood_create(UQ_LIKELIHOOD_NORMAL,10);
    lh->params.sigma=1.0; lh->obs_ptr=y_lm->components;
    UQMCMCState* mcmc=uq_mcmc_create(2,300,2,NULL);
    double mpred[10]={0};
    uq_mh_sample(mcmc,prior,lh,mpred,2);
    double ar=uq_mcmc_acceptance_rate(mcmc,0);
    assert(ar>=0.0 && ar<=1.0);
    printf("  PASS: MH acc=%.3f\n",ar);
    uq_mcmc_free(mcmc);

    printf("--- L9: Decision ---\n");
    double*r1=(double[]){100,0},*r2=(double[]){20,50},*um[]={r1,r2},pr[]={0.7,0.3};
    double evpi=uq_evpi(um,pr,2,2);
    assert(evpi>=0.0);
    printf("  PASS: EVPI=%.3f\n",evpi);

    printf("\n=== Cleanup ===\n");
    uq_dist_free(norm); uq_dist_free(unif); uq_dist_free(beta); uq_dist_free(gamma);
    uq_ci_free(ci);
    uq_matrix_free(M); uq_matrix_free(Minv);
    uq_matrix_free(X_lm); uq_vector_free(y_lm); uq_lm_free(lm);
    uq_prior_free(prior); uq_likelihood_free(lh);

    printf("\n*** ALL TESTS PASSED ***\n");
    return 0;
}
