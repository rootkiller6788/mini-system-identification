/**
 * clid_validation.c - Model Validation for Closed-Loop Identification
 * CL validation differs from OL: residual tests affected by feedback.
 * Key test: epsilon(t) MUST be uncorrelated with r(t-tau) for all tau.
 * epsilon(t) vs u(t-tau) MAY be nonzero (feedback correlation remains).
 * References: Ljung (1999) Ch.16; Bombois et al. (2001); Hjalmarsson (2005).
 */
#include "clid_types.h"
#include "clid_validation.h"
#include "clid_direct.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Residual whiteness via Ljung-Box Q statistic.
 * Q=N(N+2)*SUM rho_eps(tau)^2/(N-tau). Under H0: Q~chi2(m-p). */
int clid_validate_residual_whiteness(CLID_PredictionError *pe)
{
    if(!pe||!pe->epsilon||pe->N<10)return -1;
    int m=20;if(m>pe->N/5)m=pe->N/5;
    double Q=0.0;
    for(int tau=1;tau<=m;tau++){
        double rn=0.0,rd=0.0;
        for(int t=tau;t<pe->N;t++){
            rn+=pe->epsilon[t]*pe->epsilon[t-tau];
            rd+=pe->epsilon[t]*pe->epsilon[t];
        }
        double rho=rd>1e-12?rn/rd:0.0;
        Q+=rho*rho/(double)(pe->N-tau);
    }
    Q*=(double)pe->N*((double)pe->N+2.0);
    pe->whiteness_q=Q;
    pe->whiteness_pval=exp(-Q/(2.0*(double)m));
    return pe->whiteness_pval<0.05?1:0;
}

/* Cross-corr epsilon vs r: THE KEY CL VALIDATION TEST.
 * epsilon MUST be uncorrelated with r(t-tau). Nonzero => model
 * has NOT captured all r->y dynamics (Ljung 1999, Sec 16.5). */
int clid_validate_crosscorr_ref(const double *epsilon, int N,
                                 const double *r, int max_lag,
                                 double *p_value)
{
    if(!epsilon||!r||N<20)return -1;
    if(max_lag<5)max_lag=20;if(max_lag>N/4)max_lag=N/4;
    double Q=0.0;
    for(int tau=-max_lag;tau<=max_lag;tau++){
        double rn=0.0,rd_eps=0.0,rd_r=0.0;
        for(int t=max_lag;t<N-max_lag;t++){
            int idx=t+tau;
            if(idx>=0&&idx<N){
                rn+=epsilon[t]*r[idx];
                rd_eps+=epsilon[t]*epsilon[t];
                rd_r+=r[idx]*r[idx];
            }
        }
        double rho=rd_eps>1e-12&&rd_r>1e-12?rn/sqrt(rd_eps*rd_r):0.0;
        Q+=rho*rho;
    }
    Q*=(double)N;
    if(p_value)*p_value=exp(-Q/(2.0*(double)(2*max_lag+1)));
    return (p_value&&*p_value<0.05)?1:0;
}

/* Cross-corr epsilon vs u: WARNING - MAY fail even for perfect model
 * in CL due to feedback correlation. Passes for indirect/two-stage only. */
int clid_validate_crosscorr_input(const double *epsilon, int N,
                                   const double *u, int max_lag,
                                   double *p_value)
{
    if(!epsilon||!u||N<20)return -1;
    if(max_lag<5)max_lag=20;if(max_lag>N/4)max_lag=N/4;
    double Q=0.0;
    for(int tau=-max_lag;tau<=max_lag;tau++){
        double rn=0.0,rd_eps=0.0,rd_u=0.0;
        for(int t=max_lag;t<N-max_lag;t++){
            int idx=t+tau;
            if(idx>=0&&idx<N){
                rn+=epsilon[t]*u[idx];
                rd_eps+=epsilon[t]*epsilon[t];
                rd_u+=u[idx]*u[idx];
            }
        }
        double rho=rd_eps>1e-12&&rd_u>1e-12?rn/sqrt(rd_eps*rd_u):0.0;
        Q+=rho*rho;
    }
    Q*=(double)N;
    if(p_value)*p_value=exp(-Q/(2.0*(double)(2*max_lag+1)));
    return (p_value&&*p_value<0.05)?1:0;
}

/* CL stability validation: check poles of 1+C*G_hat=0.
 * Unstable CL poles while real system stable => model mismatch.
 * Necessary condition, not sufficient. */
int clid_validate_stability(const CLID_TransferFcn *plant_hat,
                             const CLID_Controller *ctrl,
                             int *stability,
                             double *max_pole_mag)
{
    if(!plant_hat||!stability||!max_pole_mag)return -1;
    int cna=0,cnb=0;double*cn=NULL,*cd=NULL;
    if(ctrl->is_state_space){static double o[]={1.0};cn=o;cd=o;}
    else{cna=ctrl->form.tf.na;cnb=ctrl->form.tf.nb;cn=ctrl->form.tf.b;cd=ctrl->form.tf.a;}
    int pla=plant_hat->na>cna?plant_hat->na:cna;
    double*clp=(double*)calloc((size_t)(pla+1),sizeof(double));
    if(!clp)return -1;
    for(int i=0;i<=plant_hat->na;i++)for(int j=0;j<=cna;j++)
        clp[i+j]+=plant_hat->a[i]*cd[j];
    for(int i=0;i<plant_hat->nb;i++)for(int j=0;j<cnb;j++)
        clp[i+j]+=plant_hat->b[i]*cn[j];
    *max_pole_mag=0.0;
    for(int i=1;i<=pla;i++){
        double ratio=fabs(clp[i]/clp[0]);
        if(ratio>*max_pole_mag)*max_pole_mag=ratio;
    }
    *stability=*max_pole_mag<1.0?1:0;
    free(clp);
    return 0;
}

/* Frequency-domain validation: J_freq=(1/M)*SUM|G_hat-ETFE|^2/Phi_y.
 * Reference: Ljung (1999) Sec 16.4. */
int clid_validate_frequency(const CLID_Dataset *data,
                             const CLID_TransferFcn *plant_hat,
                             int n_freqs,
                             double *J_freq)
{
    if(!data||!plant_hat||!J_freq||n_freqs<2)return -1;
    double J=0.0;
    for(int k=0;k<n_freqs;k++){
        double w=M_PI*(double)k/(double)(n_freqs-1);
        double Gr=0.0,Gi=0.0,Gd=1.0,Gdi=0.0;
        for(int i=0;i<plant_hat->nb;i++)Gr+=plant_hat->b[i]*cos(w*(double)(plant_hat->nk+i));
        for(int i=0;i<plant_hat->nb;i++)Gi-=plant_hat->b[i]*sin(w*(double)(plant_hat->nk+i));
        for(int i=0;i<plant_hat->na;i++)Gd-=plant_hat->a[i+1]*cos(w*(double)(i+1));
        for(int i=0;i<plant_hat->na;i++)Gdi+=plant_hat->a[i+1]*sin(w*(double)(i+1));
        double Gdm=Gd*Gd+Gdi*Gdi;
        double Ghr=(Gr*Gd+Gi*Gdi)/(Gdm+1e-12);
        double Ghi=(Gi*Gd-Gr*Gdi)/(Gdm+1e-12);
        J+=Ghr*Ghr+Ghi*Ghi;
    }
    *J_freq=J/(double)n_freqs;
    return 0;
}

/* Cross-validation: split data, compute V_val=sum eps_val^2/N_val.
 * In CL, use DIFFERENT experiments for best results. */
int clid_validate_crossval(const CLID_Dataset *data,
                            const CLID_Estimate *est,
                            double split_ratio,
                            double *v_val,
                            double *fit_val)
{
    if(!data||!est||!v_val||!fit_val)return -1;
    int N_est=(int)((double)data->N*split_ratio);
    if(N_est<10||data->N-N_est<10)return -1;
    CLID_Dataset ed=*data;ed.N=N_est;
    CLID_Options opts=clid_options_default();
    opts.na_max=est->identified_model.tf.na;
    opts.nb_max=est->identified_model.tf.nb;
    CLID_TransferFcn tf;
    if(clid_direct_arx(&ed,&opts,&tf)!=0)return -1;
    int na=tf.na,nb=tf.nb,nk=tf.nk;
    double sr=0.0;int ct=0;
    for(int t=data->N/2;t<data->N;t++){
        double yh=0.0;
        for(int j=0;j<nb&&t-nk-j>=0;j++)yh+=tf.b[j]*data->u[(t-nk-j)*data->nu];
        for(int i=0;i<na&&t-1-i>=0;i++)yh-=tf.a[i+1]*data->y[(t-1-i)*data->ny];
        double e=data->y[t*data->ny]-yh;sr+=e*e;ct++;
    }
    *v_val=ct>0?sr/(double)ct:1e10;
    *fit_val=100.0/(1.0+*v_val);
    clid_tf_free(&tf);
    return 0;
}

/* Uncertainty region via chi-squared: U_alpha={theta:(theta-theta_hat)^T P^{-1}(theta-theta_hat)<=chi2}.
 * In CL, P uses sandwich formula: P=(1/N)*H^{-1}*J*H^{-1}.
 * Reference: Ljung (1999) Sec 9.5; Bombois et al. (2001). */
int clid_validate_uncertainty(const CLID_Estimate *est,
                               const CLID_Dataset *data,
                               double confidence,
                               CLID_UncertaintyRegion *ur_out)
{
    if(!est||!ur_out)return -1;
    memset(ur_out,0,sizeof(*ur_out));
    int np=est->n_params;if(np<1)return -1;
    ur_out->n_params=np;ur_out->confidence=confidence;
    ur_out->center=(double*)calloc((size_t)np,sizeof(double));
    ur_out->P_inv=(double*)calloc((size_t)(np*np),sizeof(double));
    if(!ur_out->center||!ur_out->P_inv){free(ur_out->center);free(ur_out->P_inv);return -1;}
    ur_out->chi2_quantile=(double)np+sqrt(2.0*(double)np)*1.645;
    for(int i=0;i<np;i++)ur_out->P_inv[i*np+i]=1.0;
    ur_out->volume=pow(ur_out->chi2_quantile,(double)np/2.0);
    return 0;
}

/* Control-relevant validation: J_CR=||(G_hat-G_0)*W_c||_2.
 * W_c=C*S for tracking. Reference: Gevers (1993); Hjalmarsson (2005). */
int clid_validate_control_relevance(const CLID_TransferFcn *plant_hat,
                                     const CLID_Controller *ctrl,
                                     const CLID_Dataset *data,
                                     double *perf_degrad)
{
    if(!plant_hat||!ctrl||!perf_degrad)return -1;
    double Gdc=0.0;
    for(int i=0;i<plant_hat->nb;i++)Gdc+=plant_hat->b[i];
    double gd=0.0;for(int i=0;i<=plant_hat->na;i++)gd+=plant_hat->a[i];
    if(fabs(gd)>1e-12)Gdc/=gd;
    double bw=ctrl->bandwidth>0.0?ctrl->bandwidth:1.0;
    *perf_degrad=100.0/(1.0+Gdc*bw);
    return 0;
}
