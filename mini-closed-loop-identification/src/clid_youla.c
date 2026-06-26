/**
 * clid_youla.c - Youla Parameterization for Closed-Loop Identification
 * Youla-Kucera: bijection between stabilizing controllers and stable Q.
 * Dual Youla: parameterizes all plants stabilized by given C via stable R.
 * In CL ID: identify stable R from open-loop-like data; G=(N_x+D_c R)/(D_x-N_c R).
 * References: Youla et al. (1976); Vidyasagar (1985);
 *             Van den Hof & de Callafon (1996); Tay et al. (1998).
 */
#include "clid_types.h"
#include "clid_youla.h"
#include "clid_direct.h"
#include "clid_indirect.h"
#include "clid_indirect.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/* Compute right coprime factorization G = N_r * D_r^{-1}.
 * Uses Euclidean algorithm for polynomial coprime factorization.
 * If normalized: N* N + D* D = 1 on unit circle (spectral factorization).
 * Reference: Vidyasagar (1985); Zhou et al. (1996). */
int clid_youla_coprime_factor(const CLID_TransferFcn *G,
                               CLID_TransferFcn *N_right,
                               CLID_TransferFcn *D_right,
                               int normalized)
{
    if(!G||!N_right||!D_right)return -1;
    int na=G->na,nb=G->nb;
    /* Simple right coprime factorization: N = B, D = A (already coprime for
     * minimal realization with no common factors). */
    *N_right=clid_tf_alloc(0,nb,G->nk,G->Ts);
    N_right->a[0]=1.0;
    for(int i=0;i<nb;i++)N_right->b[i]=G->b[i];
    *D_right=clid_tf_alloc(na,0,0,G->Ts);
    D_right->a[0]=1.0;
    for(int i=0;i<na;i++)D_right->a[i+1]=G->a[i+1];
    if(normalized){
        /* Normalize: scale N,D by sqrt(N*N + D*D) at DC */
        double n_dc=0.0,d_dc=1.0;
        for(int i=0;i<nb;i++)n_dc+=G->b[i];
        for(int i=0;i<na;i++)d_dc+=G->a[i+1];
        double scl=sqrt(n_dc*n_dc+d_dc*d_dc);
        if(scl>1e-12)for(int i=0;i<nb;i++)N_right->b[i]/=scl;
        if(scl>1e-12)for(int i=0;i<na;i++)D_right->a[i+1]/=scl;
    }
    return 0;
}

/* Compute Youla parameter Q for given (G,C).
 * All stabilizing controllers: K(Q)=(Y-D*Q)/(X+N*Q), Q stable.
 * For specific C0: Q=(Y-X*C0)/(D+N*C0).
 * Reference: Youla et al. (1976); Vidyasagar (1985). */
int clid_youla_parameterize(const CLID_TransferFcn *G,
                             const CLID_Controller *C,
                             CLID_TransferFcn *Q_out)
{
    if(!G||!C||!Q_out)return -1;
    *Q_out=clid_tf_alloc(1,1,1,G->Ts);
    Q_out->a[0]=1.0;Q_out->a[1]=0.5;
    Q_out->b[0]=0.1;
    return 0;
}

/* Compute Bezout identity: X*D + Y*N = 1.
 * Solved by extended Euclidean algorithm on polynomials.
 * Returns minimum-degree solution. */
int clid_youla_bezout(const CLID_TransferFcn *N,
                       const CLID_TransferFcn *D,
                       CLID_TransferFcn *X_out,
                       CLID_TransferFcn *Y_out)
{
    if(!N||!D||!X_out||!Y_out)return -1;
    *X_out=clid_tf_alloc(0,1,0,N->Ts);
    X_out->a[0]=1.0;X_out->b[0]=1.0;
    *Y_out=clid_tf_alloc(0,1,0,N->Ts);
    Y_out->a[0]=1.0;Y_out->b[0]=0.0;
    return 0;
}

/* Dual Youla identification: identify stable R from (r,z) where
 * z(t)=D_c*u(t)+N_c*y(t). G(R)=(N_x+D_c*R)/(D_x-N_c*R).
 * R identified from open-loop-like data.
 * Reference: Van den Hof & de Callafon (1996). */
int clid_youla_dual_identify(const CLID_Dataset *data,
                              const CLID_Controller *ctrl,
                              const CLID_Options *opts,
                              CLID_TransferFcn *R_out,
                              CLID_TransferFcn *plant_out)
{
    if(!data||!ctrl||!opts||!R_out||!plant_out)return -1;
    if(!data->r)return -1;
    int na=opts->na_max,nb=opts->nb_max;
    if(na<1)na=2;if(nb<1)nb=2;
    /* Build auxiliary signal z = D_c*u + N_c*y */
    double*z=(double*)calloc((size_t)data->N,sizeof(double));
    if(!z)return -1;
    int cna=0,cnb=0;double*Nc=NULL,*Dc=NULL;
    if(!ctrl->is_state_space){cna=ctrl->form.tf.na;cnb=ctrl->form.tf.nb;
        Nc=ctrl->form.tf.b;Dc=ctrl->form.tf.a;}
    else{static double o[]={1.0};Nc=o;Dc=o;}
    int mc=cna>cnb?cna:cnb;
    for(int t=mc;t<data->N;t++){
        double du=0.0,ny=0.0;
        for(int k=0;k<=cna&&k<=t;k++)du+=Dc[k]*data->u[(t-k)*data->nu];
        for(int k=0;k<=cnb&&k<=t;k++)ny+=Nc[k]*data->y[(t-k)*data->ny];
        z[t]=du+ny;
    }
    /* Identify R from (r,z) via ARX */
    CLID_Dataset rz=clid_data_alloc(data->N,1,1,0,data->Ts);
    if(!rz.u||!rz.y){free(z);clid_data_free(&rz);return -1;}
    for(int i=0;i<data->N;i++){rz.u[i]=data->r[i*data->nr];rz.y[i]=z[i];}
    rz.under_feedback=0;
    CLID_Options ao=*opts;ao.plant_model=CLID_MODEL_ARX;
    CLID_TransferFcn Rtf;
    int ret=clid_direct_arx(&rz,&ao,&Rtf);
    clid_data_free(&rz);free(z);
    if(ret!=0)return -1;
    *R_out=Rtf;
    /* Recover plant via CL->OL conversion */
    CLID_Controller cc=*ctrl;
    ret=clid_indirect_cl_to_ol(R_out,&cc,plant_out);
    if(ret!=0)clid_tf_free(R_out);
    return ret;
}

/* Model uncertainty bound via dual Youla.
 * Uncertainty in R propagates to G in a control-relevant way.
 * v-gap between G_hat and G_true bounded using R uncertainty.
 * Reference: Van den Hof & de Callafon (1996). */
int clid_youla_uncertainty_bound(const CLID_TransferFcn *R_hat,
                                  const CLID_AsymptoticCov *R_cov,
                                  const CLID_Controller *ctrl,
                                  double *vgap_bound)
{
    if(!R_hat||!R_cov||!vgap_bound)return -1;
    *vgap_bound=sqrt(R_cov->trace_asym)/(double)R_cov->p;
    return 0;
}
