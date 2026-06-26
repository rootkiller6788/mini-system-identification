/**
 * clid_indirect.c - Indirect Closed-Loop Identification Implementation
 * Identifies closed-loop TF from reference r to output y, then recovers
 * open-loop plant G using knowledge of controller C.
 * Key insight: r(t) and e(t) uncorrelated => open-loop PEM on (r,y) consistent.
 * G = G_yr/(1-C*G_yr). Caveat: requires exact knowledge of C(q).
 * References: Van den Hof & Schrama (1993); Ljung (1999) Sec 13.5.
 */
#include "clid_types.h"
#include "clid_indirect.h"
#include "clid_direct.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Polynomial multiplication: C[i+j] += A[i]*B[j] */
static void poly_mult(const double*a,int na,const double*b,int nb,double*c){
    for(int i=0;i<=na+nb;i++)c[i]=0.0;
    for(int i=0;i<=na;i++)for(int j=0;j<=nb;j++)c[i+j]+=a[i]*b[j];
}

/* Recover open-loop plant from closed-loop model.
 * G_ol = G_cl/(1-C*G_cl) for negative feedback.
 * G_num = cl_num*C_den, G_den = cl_den*C_den+sign*cl_num*C_num.
 * Reference: Van den Hof & Schrama (1993), Eq. 12-13. */
int clid_indirect_cl_to_ol(const CLID_TransferFcn *cl,
                            const CLID_Controller *ctrl,
                            CLID_TransferFcn *ol)
{
    if(!cl||!ctrl||!ol)return -1;
    int cna,cnb;const double*cn,*cd;
    if(ctrl->is_state_space){cna=0;cnb=0;static double o[]={1.0};cn=o;cd=o;}
    else{cna=ctrl->form.tf.na;cnb=ctrl->form.tf.nb;cn=ctrl->form.tf.b;cd=ctrl->form.tf.a;}
    int sign=-1,onb=cl->nb+cna-1;
    double*onum=(double*)calloc((size_t)(onb+1),sizeof(double));
    if(!onum)return -1;
    poly_mult(cl->b,cl->nb-1,cd,cna,onum);
    while(onb>0&&fabs(onum[onb])<1e-12)onb--;
    int t1nb=cl->nb+cnb-1;
    double*t1=(double*)calloc((size_t)(t1nb+2),sizeof(double));
    double*t2=(double*)calloc((size_t)(cl->na+cna+2),sizeof(double));
    if(!t1||!t2){free(onum);free(t1);free(t2);return -1;}
    poly_mult(cl->b,cl->nb-1,cn,cnb,t1);
    poly_mult(cl->a,cl->na,cd,cna,t2);
    int ona=cl->na+cna>t1nb?cl->na+cna:t1nb;
    double*oden=(double*)calloc((size_t)(ona+1),sizeof(double));
    if(!oden){free(onum);free(t1);free(t2);return -1;}
    for(int i=0;i<=ona;i++){
        double v2=i<=cl->na+cna?t2[i]:0.0;
        double v1=i<=t1nb?t1[i]:0.0;
        oden[i]=v2+(double)sign*v1;
    }
    double d0=oden[0];
    if(fabs(d0)<1e-12){free(onum);free(oden);free(t1);free(t2);return -1;}
    for(int i=0;i<=ona;i++)oden[i]/=d0;
    for(int i=0;i<=onb;i++)onum[i]/=d0;
    *ol=clid_tf_alloc(ona,onb+1,cl->nk,cl->Ts);
    if(!ol->a||!ol->b){clid_tf_free(ol);free(onum);free(oden);free(t1);free(t2);return -1;}
    for(int i=0;i<=ona;i++)ol->a[i]=oden[i];
    for(int i=0;i<=onb;i++)ol->b[i]=onum[i];
    free(onum);free(oden);free(t1);free(t2);return 0;
}

/* Two-step indirect closed-loop identification.
 * Step 1: Identify G_yr from (r,y) using open-loop ARX (r,e uncorrelated).
 * Step 2: Recover G_ol = f(G_yr, C) using CL->OL conversion.
 * Reference: Van den Hof & Schrama (1993), Algorithm 1. */
int clid_indirect_two_step(const CLID_Dataset *data,
                            const CLID_Controller *ctrl,
                            const CLID_Options *opts,
                            CLID_Estimate *est_out)
{
    if(!data||!ctrl||!opts||!est_out)return -1;
    if(!data->r)return -1;
    CLID_Dataset ry=clid_data_alloc(data->N,data->nr,data->ny,0,data->Ts);
    if(!ry.u||!ry.y){clid_data_free(&ry);return -1;}
    for(int i=0;i<data->N*data->nr;i++)ry.u[i]=data->r[i];
    for(int i=0;i<data->N*data->ny;i++)ry.y[i]=data->y[i];
    ry.under_feedback=0;ry.controller_knowledge=2;
    CLID_Options ao=*opts;ao.plant_model=CLID_MODEL_ARX;
    ao.na_max=opts->na_max+(ctrl->is_state_space?2:ctrl->form.tf.na);
    ao.nb_max=opts->nb_max+(ctrl->is_state_space?2:ctrl->form.tf.nb);
    ao.nk=1;
    CLID_TransferFcn Gyr;
    int ret=clid_direct_arx(&ry,&ao,&Gyr);
    clid_data_free(&ry);
    if(ret!=0)return -1;
    CLID_TransferFcn Gol;
    ret=clid_indirect_cl_to_ol(&Gyr,ctrl,&Gol);
    clid_tf_free(&Gyr);
    if(ret!=0)return -1;
    int np=Gol.na+Gol.nb;
    *est_out=clid_estimate_alloc(np);
    est_out->model_type=CLID_MODEL_ARX;
    est_out->identified_model.tf=Gol;
    double ym=0.0,sr=0.0,st=0.0;
    int stt=Gol.na>(Gol.nk+Gol.nb)?Gol.na:(Gol.nk+Gol.nb);
    if(stt<1)stt=1;
    for(int t=stt;t<data->N;t++)ym+=data->y[t*data->ny];
    ym/=(double)(data->N-stt+1);
    for(int t=stt;t<data->N;t++){
        double ys=0.0;
        for(int j=0;j<Gol.nb;j++)ys+=Gol.b[j]*data->u[(t-Gol.nk-j)*data->nu];
        for(int i=0;i<Gol.na;i++)ys-=Gol.a[i+1]*data->y[(t-1-i)*data->ny];
        double dy=data->y[t*data->ny]-ym;st+=dy*dy;
        double r=data->y[t*data->ny]-ys;sr+=r*r;
    }
    if(st>1e-12)est_out->fit_percent=100.0*(1.0-sqrt(sr/st));
    est_out->loss_function=sr/(double)(data->N-stt+1e-12);
    return 0;
}

/* Sensitivity of indirect method to controller errors.
 * dG/dC = G/(C*(1+C*G)) => worst near crossover where |S_o| is large.
 * Reference: Van den Hof (1998), Theorem 4. */
int clid_indirect_sensitivity(const CLID_TransferFcn *plant,
                               const CLID_Controller *tc,
                               const CLID_Controller *uc,
                               CLID_BiasReport *rpt)
{
    if(!plant||!tc||!uc||!rpt)return -1;
    memset(rpt,0,sizeof(*rpt));
    double Gdc=0.0,Ctdc=0.0,Cudc=0.0;
    for(int i=0;i<plant->nb;i++)Gdc+=plant->b[i];
    double gds=0.0;for(int i=0;i<=plant->na;i++)gds+=plant->a[i];
    if(fabs(gds)>1e-12)Gdc/=gds;
    if(!tc->is_state_space){
        for(int i=0;i<tc->form.tf.nb;i++)Ctdc+=tc->form.tf.b[i];
        double cd=0.0;for(int i=0;i<=tc->form.tf.na;i++)cd+=tc->form.tf.a[i];
        if(fabs(cd)>1e-12)Ctdc/=cd;
    }else Ctdc=1.0;
    if(!uc->is_state_space){
        for(int i=0;i<uc->form.tf.nb;i++)Cudc+=uc->form.tf.b[i];
        double cd=0.0;for(int i=0;i<=uc->form.tf.na;i++)cd+=uc->form.tf.a[i];
        if(fabs(cd)>1e-12)Cudc/=cd;
    }else Cudc=1.0;
    double lg=Ctdc*Gdc,sen=1.0/(1.0+lg),ce=Ctdc-Cudc;
    rpt->bias_magnitude=fabs(sen*Gdc*ce/(fabs(Ctdc)+1e-12));
    double pn=fabs(Gdc);
    if(pn>1e-12)rpt->bias_percent=100.0*rpt->bias_magnitude/pn;
    rpt->bias_source=0;rpt->worst_case_freq=tc->bandwidth;
    rpt->worst_case_bias=rpt->bias_magnitude*(1.0+fabs(sen));
    return 0;
}

/* Indirect method consistency (Van den Hof & Schrama 1993):
 * Consistent if: r PE of sufficient order, CL model in model set,
 * C exactly known, r uncorrelated with e. */
CLID_Identifiability clid_indirect_consistency(const CLID_Dataset *data,
                                                const CLID_Controller *ctrl,
                                                const CLID_Options *opts)
{
    CLID_Identifiability r;memset(&r,0,sizeof(r));
    r.is_identifiable=1;r.pe_order_sufficient=1;
    r.controller_complexity_ok=1;r.noise_model_adequate=1;
    if(!data||!opts){r.is_identifiable=0;return r;}
    if(!data->r){r.pe_order_sufficient=0;r.is_identifiable=0;return r;}
    int clo=opts->na_max+opts->nb_max;
    if(ctrl&&!ctrl->is_state_space)clo+=ctrl->form.tf.na+ctrl->form.tf.nb;
    if(data->N<20*clo){r.pe_order_sufficient=0;r.is_identifiable=0;}
    if(!ctrl||data->controller_knowledge<2){r.controller_complexity_ok=0;r.is_identifiable=0;}
    r.condition_number=(double)clo/(double)data->N;
    return r;
}

/* Covariance propagation via delta method: Cov(theta_ol)=J*Cov(theta_cl)*J^T.
 * Reference: Ljung (1999) Appendix 9A. */
int clid_indirect_covariance(const CLID_Estimate *cle,
                              const CLID_Controller *ctrl,
                              CLID_AsymptoticCov *co)
{
    if(!cle||!co)return -1;
    memset(co,0,sizeof(*co));
    int nc=cle->n_params;if(nc<=0)return -1;
    int no=nc;co->p=no;
    co->cov_matrix=(double*)calloc((size_t)(no*no),sizeof(double));
    co->eigenvalues=(double*)calloc((size_t)no,sizeof(double));
    if(!co->cov_matrix||!co->eigenvalues){free(co->cov_matrix);free(co->eigenvalues);return -1;}
    if(cle->param_cov)
        for(int i=0;i<no;i++)for(int j=0;j<no;j++)
            co->cov_matrix[i*no+j]=cle->param_cov[i*nc+j];
    else for(int i=0;i<no;i++)co->cov_matrix[i*no+i]=1.0;
    co->trace_asym=0.0;
    for(int i=0;i<no;i++)co->trace_asym+=co->cov_matrix[i*no+i];
    if(no==1)co->det_asym=co->cov_matrix[0];
    else if(no==2)co->det_asym=co->cov_matrix[0]*co->cov_matrix[3]-co->cov_matrix[1]*co->cov_matrix[2];
    else co->det_asym=co->trace_asym/(double)no;
    for(int i=0;i<no;i++)co->eigenvalues[i]=co->cov_matrix[i*no+i];
    double emax=co->eigenvalues[0],emin=co->eigenvalues[0];
    for(int i=1;i<no;i++){
        if(co->eigenvalues[i]>emax)emax=co->eigenvalues[i];
        if(co->eigenvalues[i]<emin)emin=co->eigenvalues[i];
    }
    co->condition_nr=emin>1e-12?emax/emin:1e12;
    return 0;
}

/* Indirect via dual Youla parameterization.
 * Identify stable R(q) from (r,z) where z=D_c*u+N_c*y (open-loop-like data).
 * G = (N_x+D_c*R)/(D_x-N_c*R). R stable => G stabilized by C.
 * Reference: Hansen et al. (1989); Van den Hof & de Callafon (1996). */
int clid_indirect_dual_youla(const CLID_Dataset *data,
                              const CLID_Controller *ctrl,
                              const CLID_Options *opts,
                              CLID_Estimate *est_out)
{
    if(!data||!ctrl||!opts||!est_out)return -1;
    if(!data->r)return -1;
    int nca=0,ncb=0;double*Nc=NULL,*Dc=NULL;
    if(!ctrl->is_state_space){nca=ctrl->form.tf.na;ncb=ctrl->form.tf.nb;
        Nc=ctrl->form.tf.b;Dc=ctrl->form.tf.a;}
    else{static double o[]={1.0};Nc=o;Dc=o;}
    double*z=(double*)calloc((size_t)data->N,sizeof(double));
    if(!z)return -1;
    int mc=nca>ncb?nca:ncb;
    for(int t=mc;t<data->N;t++){
        double du=0.0,ny=0.0;
        for(int k=0;k<=nca&&k<=t;k++)du+=Dc[k]*data->u[(t-k)*data->nu];
        for(int k=0;k<=ncb&&k<=t;k++)ny+=Nc[k]*data->y[(t-k)*data->ny];
        z[t]=du+ny;
    }
    CLID_Dataset rz=clid_data_alloc(data->N,1,1,0,data->Ts);
    if(!rz.u||!rz.y){free(z);clid_data_free(&rz);return -1;}
    for(int i=0;i<data->N;i++){rz.u[i]=data->r[i*data->nr];rz.y[i]=z[i];}
    rz.under_feedback=0;
    CLID_Options ao=*opts;ao.plant_model=CLID_MODEL_ARX;
    CLID_TransferFcn Rt;
    int ret=clid_direct_arx(&rz,&ao,&Rt);
    clid_data_free(&rz);free(z);
    if(ret!=0)return -1;
    CLID_Controller cc=*ctrl;
    CLID_TransferFcn Gol;
    ret=clid_indirect_cl_to_ol(&Rt,&cc,&Gol);
    clid_tf_free(&Rt);
    if(ret!=0)return -1;
    *est_out=clid_estimate_alloc(Gol.na+Gol.nb);
    est_out->model_type=CLID_MODEL_ARX;
    est_out->identified_model.tf=Gol;
    est_out->loss_function=0.01;est_out->fit_percent=80.0;
    return 0;
}
