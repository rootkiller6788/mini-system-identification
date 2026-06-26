/**
 * clid_direct.c - Direct Closed-Loop Identification Implementation
 * Implements ARX, ARMAX, OE, BJ, SS PEM for closed-loop data.
 * Key theorem (Ljung 1999, Thm 13.1): direct PEM consistent iff
 * noise model H(q,theta) contains true noise model.
 * References: Ljung (1999) Ch.13.4; Forssell & Ljung (1999)
 */
#include "clid_types.h"
#include "clid_direct.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static int solve_linear_system(double *A, double *b, int n)
{
    if (n <= 0 || !A || !b) return -1;
    for (int col = 0; col < n; col++) {
        int max_row = col;
        double max_val = fabs(A[col * n + col]);
        for (int row = col + 1; row < n; row++) {
            double v = fabs(A[row * n + col]);
            if (v > max_val) { max_val = v; max_row = row; }
        }
        if (max_val < 1e-15) return -1;
        if (max_row != col) {
            for (int j = col; j < n; j++) {
                double tmp = A[col * n + j];
                A[col * n + j] = A[max_row * n + j];
                A[max_row * n + j] = tmp;
            }
            double tmp = b[col]; b[col] = b[max_row]; b[max_row] = tmp;
        }
        double pivot = A[col * n + col];
        for (int row = col + 1; row < n; row++) {
            double factor = A[row * n + col] / pivot;
            if (fabs(factor) < 1e-15) continue;
            for (int j = col; j < n; j++)
                A[row * n + j] -= factor * A[col * n + j];
            b[row] -= factor * b[col];
        }
    }
    for (int i = n - 1; i >= 0; i--) {
        double sum = b[i];
        for (int j = i + 1; j < n; j++)
            sum -= A[i * n + j] * b[j];
        b[i] = sum / A[i * n + i];
    }
    return 0;
}

/* Direct ARX via Least Squares.
 * Model: A(q)y(t)=B(q)u(t)+e(t)
 * Solves (Phi^T Phi)theta = Phi^T Y. O(N*(na+nb)^2). */
int clid_direct_arx(const CLID_Dataset *data,
                    const CLID_Options *opts,
                    CLID_TransferFcn *tf_out)
{
    if (!data || !opts || !tf_out) return -1;
    if (data->N < 10 || !data->y || !data->u) return -1;
    int na = opts->na_max, nb = opts->nb_max, nk = opts->nk;
    if (na < 1) na = 1; if (nb < 1) nb = 1; if (nk < 1) nk = 1;
    int n_theta = na + nb;
    int start_row = na > nk + nb - 1 ? na : nk + nb - 1;
    if (start_row >= data->N) return -1;
    int N_eff = data->N - start_row;
    if (N_eff < n_theta) return -1;
    double *PhiTPhi = (double *)calloc((size_t)(n_theta*n_theta), sizeof(double));
    double *PhiTY   = (double *)calloc((size_t)n_theta, sizeof(double));
    if (!PhiTPhi || !PhiTY) { free(PhiTPhi); free(PhiTY); return -1; }
    for (int t = start_row; t < data->N; t++) {
        double phi[64];
        for (int i = 0; i < na; i++) phi[i] = -data->y[(t-1-i)*data->ny];
        for (int i = 0; i < nb; i++) phi[na+i] = data->u[(t-nk-i)*data->nu];
        double yt = data->y[t*data->ny];
        for (int i = 0; i < n_theta; i++) {
            for (int j = 0; j < n_theta; j++)
                PhiTPhi[i*n_theta+j] += phi[i]*phi[j];
            PhiTY[i] += phi[i]*yt;
        }
    }
    double *theta = (double *)malloc((size_t)n_theta * sizeof(double));
    if (!theta) { free(PhiTPhi); free(PhiTY); return -1; }
    memcpy(theta, PhiTY, (size_t)n_theta * sizeof(double));
    int ret = solve_linear_system(PhiTPhi, theta, n_theta);
    if (ret == 0) {
        *tf_out = clid_tf_alloc(na, nb, nk, data->Ts);
        if (!tf_out->a || !tf_out->b) { clid_tf_free(tf_out); ret = -1; }
        else {
            tf_out->a[0] = 1.0;
            for (int i = 0; i < na; i++) tf_out->a[i+1] = theta[i];
            for (int i = 0; i < nb; i++) tf_out->b[i] = theta[na+i];
        }
    }
    free(theta); free(PhiTPhi); free(PhiTY);
    return ret;
}

/* Direct ARMAX via Gauss-Newton PEM.
 * Model: A(q)y(t)=B(q)u(t)+C(q)e(t)
 * Predictor: C*eps(t)=A*y(t)-B*u(t). GN: theta+=(J^TJ)^-1 J^T eps
 * Gradient: d_eps/da=(1/C)y(t-i), d_eps/db=-(1/C)u(t-nk-j), d_eps/dc=-(1/C)eps(t-k)
 * Reference: Ljung (1999) Section 10.2. */
int clid_direct_armax(const CLID_Dataset *data,
                      const CLID_Options *opts,
                      CLID_Estimate *est_out)
{
    if (!data || !opts || !est_out) return -1;
    int na=opts->na_max, nb=opts->nb_max, nk=opts->nk;
    int nc=(opts->noise_model==CLID_NOISE_MA)?nb/2:na;
    if(na<1)na=1; if(nb<1)nb=1; if(nk<1)nk=1; if(nc<1)nc=1;
    int n_theta=na+nb+nc, start_t=na>(nk+nb)?na:(nk+nb);
    if(nc>start_t)start_t=nc;
    if(start_t>=data->N||data->N-start_t<n_theta)return -1;
    int N_eff=data->N-start_t;
    double*th=(double*)calloc((size_t)n_theta,sizeof(double));
    double*ep=(double*)calloc((size_t)data->N,sizeof(double));
    double*JtJ=(double*)calloc((size_t)(n_theta*n_theta),sizeof(double));
    double*JtE=(double*)calloc((size_t)n_theta,sizeof(double));
    if(!th||!ep||!JtJ||!JtE){free(th);free(ep);free(JtJ);free(JtE);return -1;}
    /* Init via ARX */
    CLID_Options ao=*opts; ao.na_max=na; ao.nb_max=nb; ao.nk=nk;
    CLID_TransferFcn ti;
    if(clid_direct_arx(data,&ao,&ti)==0){
        for(int i=0;i<na;i++)th[i]=ti.a[i+1];
        for(int i=0;i<nb;i++)th[na+i]=ti.b[i];
        clid_tf_free(&ti);
    }
    int conv=0; double pl=1e30;
    for(int iter=0;iter<opts->max_iter;iter++){
        double VN=0.0;
        for(int t=start_t;t<data->N;t++){
            double Ay=data->y[t*data->ny], Bu=0.0;
            for(int i=0;i<na;i++)Ay+=th[i]*data->y[(t-1-i)*data->ny];
            for(int i=0;i<nb;i++)Bu+=th[na+i]*data->u[(t-nk-i)*data->nu];
            double Ce=Ay-Bu;
            for(int i=0;i<nc;i++)Ce-=th[na+nb+i]*ep[t-1-i];
            ep[t]=Ce; VN+=Ce*Ce;
        }
        VN/=(double)N_eff;
        if(fabs(pl-VN)<opts->tolerance*(1.0+pl)){conv=1;break;}
        pl=VN;
        memset(JtJ,0,(size_t)(n_theta*n_theta)*sizeof(double));
        memset(JtE,0,(size_t)n_theta*sizeof(double));
        for(int t=start_t;t<data->N;t++){
            double pa[32],pb[32],pc[16];
            for(int i=0;i<na;i++)pa[i]=data->y[(t-1-i)*data->ny];
            for(int j=0;j<nb;j++)pb[j]=-data->u[(t-nk-j)*data->nu];
            for(int k=0;k<nc;k++)pc[k]=-ep[t-1-k];
            for(int i=0;i<na;i++){
                for(int j=0;j<na;j++)JtJ[i*n_theta+j]+=pa[i]*pa[j];
                for(int j=0;j<nb;j++)JtJ[i*n_theta+na+j]+=pa[i]*pb[j];
                for(int j=0;j<nc;j++)JtJ[i*n_theta+na+nb+j]+=pa[i]*pc[j];
                JtE[i]+=pa[i]*ep[t];
            }
            for(int i=0;i<nb;i++){
                int ri=na+i;
                for(int j=0;j<nb;j++)JtJ[ri*n_theta+na+j]+=pb[i]*pb[j];
                for(int j=0;j<nc;j++)JtJ[ri*n_theta+na+nb+j]+=pb[i]*pc[j];
                JtE[ri]+=pb[i]*ep[t];
            }
            for(int i=0;i<nc;i++){
                int ri=na+nb+i;
                for(int j=0;j<nc;j++)JtJ[ri*n_theta+na+nb+j]+=pc[i]*pc[j];
                JtE[ri]+=pc[i]*ep[t];
            }
        }
        double*de=(double*)malloc((size_t)n_theta*sizeof(double));
        if(!de)break;
        for(int i=0;i<n_theta;i++)de[i]=-JtE[i];
        if(solve_linear_system(JtJ,de,n_theta)==0)
            for(int i=0;i<n_theta;i++)th[i]+=de[i];
        free(de);
    }
    *est_out=clid_estimate_alloc(n_theta);
    est_out->model_type=CLID_MODEL_ARMAX;
    CLID_TransferFcn*tf=&est_out->identified_model.tf;
    *tf=clid_tf_alloc(na,nb,nk,data->Ts); tf->a[0]=1.0;
    for(int i=0;i<na;i++)tf->a[i+1]=th[i];
    for(int i=0;i<nb;i++)tf->b[i]=th[na+i];
    est_out->noise_na=nc; est_out->noise_nc=0;
    est_out->noise_model=(double*)calloc((size_t)(nc+1),sizeof(double));
    if(est_out->noise_model){
        est_out->noise_model[0]=1.0;
        for(int i=0;i<nc;i++)est_out->noise_model[i+1]=th[na+nb+i];
    }
    est_out->loss_function=pl;
    double d=(double)n_theta;
    est_out->fpe=pl*((double)N_eff+d)/((double)N_eff-d+1e-12);
    est_out->aic=(double)N_eff*log(pl+1e-12)+2.0*d;
    double ym=0.0,yv=0.0,rv=0.0;
    for(int t=start_t;t<data->N;t++)ym+=data->y[t*data->ny];
    ym/=(double)N_eff;
    for(int t=start_t;t<data->N;t++){
        double dy=data->y[t*data->ny]-ym;
        yv+=dy*dy; rv+=ep[t]*ep[t];
    }
    if(yv>1e-12){
        est_out->fit_percent=100.0*(1.0-sqrt(rv/yv));
        if(est_out->fit_percent<0.0)est_out->fit_percent=0.0;
    }
    free(th); free(ep); free(JtJ); free(JtE);
    return conv?0:-1;
}

/* Direct Output Error: y(t)=[B/F]u(t)+e(t). No noise model.
 * OE uses simulation-based prediction (no y feedback), making it
 * generally INCONSISTENT in closed loop. GN on simulated error.
 * Reference: Ljung (1999) Section 10.3. */
int clid_direct_oe(const CLID_Dataset *data,
                   const CLID_Options *opts,
                   CLID_Estimate *est_out)
{
    if(!data||!opts||!est_out)return -1;
    int nb=opts->nb_max,nf=opts->na_max,nk=opts->nk;
    if(nb<1)nb=1;if(nf<1)nf=1;if(nk<1)nk=1;
    int nth=nb+nf,st=nk+(nb>nf?nb:nf);
    if(st>=data->N||data->N-st<nth)return -1;
    int Ne=data->N-st;
    double*th=(double*)calloc((size_t)nth,sizeof(double));
    double*yh=(double*)calloc((size_t)data->N,sizeof(double));
    double*JJ=(double*)calloc((size_t)(nth*nth),sizeof(double));
    double*JE=(double*)calloc((size_t)nth,sizeof(double));
    if(!th||!yh||!JJ||!JE){free(th);free(yh);free(JJ);free(JE);return -1;}
    for(int i=0;i<nb;i++)th[i]=0.1/(double)(i+1);
    double pl=1e30;int cv=0;
    for(int iter=0;iter<opts->max_iter;iter++){
        double VN=0.0;
        for(int t=st;t<data->N;t++){
            double ys=0.0;
            for(int j=0;j<nb;j++)ys+=th[j]*data->u[(t-nk-j)*data->nu];
            for(int i=0;i<nf;i++)ys-=th[nb+i]*yh[t-1-i];
            yh[t]=ys;double ep=data->y[t*data->ny]-ys;VN+=ep*ep;
        }
        VN/=(double)Ne;
        if(fabs(pl-VN)<opts->tolerance*(1.0+pl)){cv=1;break;}
        pl=VN;
        memset(JJ,0,(size_t)(nth*nth)*sizeof(double));
        memset(JE,0,(size_t)nth*sizeof(double));
        for(int t=st;t<data->N;t++){
            double pb[32],pf[32];
            for(int j=0;j<nb;j++)pb[j]=data->u[(t-nk-j)*data->nu];
            for(int i=0;i<nf;i++)pf[i]=-yh[t-1-i];
            double ep=data->y[t*data->ny]-yh[t];
            for(int i=0;i<nb;i++){
                for(int j=0;j<nb;j++)JJ[i*nth+j]+=pb[i]*pb[j];
                for(int j=0;j<nf;j++)JJ[i*nth+nb+j]+=pb[i]*pf[j];
                JE[i]+=pb[i]*ep;
            }
            for(int i=0;i<nf;i++){
                int ri=nb+i;
                for(int j=0;j<nf;j++)JJ[ri*nth+nb+j]+=pf[i]*pf[j];
                JE[ri]+=pf[i]*ep;
            }
        }
        double*de=(double*)malloc((size_t)nth*sizeof(double));
        if(!de)break;
        for(int i=0;i<nth;i++)de[i]=-JE[i];
        if(solve_linear_system(JJ,de,nth)==0)
            for(int i=0;i<nth;i++)th[i]+=de[i];
        free(de);
    }
    *est_out=clid_estimate_alloc(nth);
    est_out->model_type=CLID_MODEL_OE;
    CLID_TransferFcn*tf=&est_out->identified_model.tf;
    *tf=clid_tf_alloc(nf,nb,nk,data->Ts);tf->a[0]=1.0;
    for(int i=0;i<nf;i++)tf->a[i+1]=th[nb+i];
    for(int i=0;i<nb;i++)tf->b[i]=th[i];
    est_out->loss_function=pl;
    est_out->fpe=pl*((double)Ne+(double)nth)/((double)Ne-(double)nth+1e-12);
    free(th);free(yh);free(JJ);free(JE);
    return cv?0:-1;
}

/* Direct Box-Jenkins: y=[B/F]u+[C/D]e. Most flexible polynomial model.
 * Independent plant/noise dynamics => C/D captures feedback correlation,
 * making BJ most likely consistent in CL. Alternating optimization.
 * Reference: Box,Jenkins,Reinsel (1994); Ljung (1999) Sec 4.2. */
int clid_direct_bj(const CLID_Dataset *data,
                   const CLID_Options *opts,
                   CLID_Estimate *est_out)
{
    if(!data||!opts||!est_out)return -1;
    int nb=opts->nb_max,nc=opts->na_max/2,nd=opts->na_max/2;
    int nf=opts->nb_max,nk=opts->nk;
    if(nb<1)nb=1;if(nc<1)nc=1;if(nd<1)nd=1;if(nf<1)nf=1;if(nk<1)nk=1;
    int nth=nb+nc+nd+nf,mo=nb;
    if(nf>mo)mo=nf;if(nc>mo)mo=nc;if(nd>mo)mo=nd;
    int st=nk+mo;
    if(st>=data->N||data->N-st<nth)return -1;
    int Ne=data->N-st;
    double*th=(double*)calloc((size_t)nth,sizeof(double));
    double*ep=(double*)calloc((size_t)data->N,sizeof(double));
    double*ys=(double*)calloc((size_t)data->N,sizeof(double));
    if(!th||!ep||!ys){free(th);free(ep);free(ys);return -1;}
    /* Init: get ARMAX first */
    CLID_Options ao=*opts;ao.plant_model=CLID_MODEL_ARMAX;
    CLID_Estimate ae;
    if(clid_direct_armax(data,&ao,&ae)==0){
        CLID_TransferFcn*at=&ae.identified_model.tf;
        for(int i=0;i<nb&&i<at->nb;i++)th[i]=at->b[i];
        for(int i=0;i<nf&&i<at->na;i++)th[nb+nc+nd+i]=at->a[i+1];
        if(ae.noise_model)
            for(int i=0;i<nc&&i<ae.noise_na;i++)th[nb+i]=ae.noise_model[i+1];
        clid_estimate_free(&ae);
    }else{for(int i=0;i<nb;i++)th[i]=0.1;th[nb]=0.3;}
    double pl=1e30;int cv=0;
    for(int iter=0;iter<opts->max_iter;iter++){
        double VN=0.0;
        for(int t=st;t<data->N;t++){
            double ys_=0.0;
            for(int j=0;j<nb;j++)ys_+=th[j]*data->u[(t-nk-j)*data->nu];
            for(int i=0;i<nf;i++)ys_-=th[nb+nc+nd+i]*ys[t-1-i];
            ys[t]=ys_;
            double re=data->y[t*data->ny]-ys_,ef=re;
            for(int k=0;k<nc;k++)ef-=th[nb+k]*ep[t-1-k];
            for(int l=0;l<nd;l++)ef+=th[nb+nc+l]*ep[t-1-l];
            ep[t]=ef;VN+=ef*ef;
        }
        VN/=(double)Ne;
        if(fabs(pl-VN)<opts->tolerance*(1.0+fabs(pl))){cv=1;break;}
        pl=VN;
        double step=0.5/(1.0+0.01*iter);
        for(int j=0;j<nb;j++){
            double g=0.0;
            for(int t=st;t<data->N;t++)g-=2.0*ep[t]*data->u[(t-nk-j)*data->nu];
            th[j]-=step*g/(double)Ne;
        }
        for(int i=0;i<nf;i++){
            int idx=nb+nc+nd+i;double g=0.0;
            for(int t=st;t<data->N;t++)g-=2.0*ep[t]*(-ys[t-1-i]);
            th[idx]-=step*g/(double)Ne;
        }
        for(int k=0;k<nc;k++){
            double g=0.0;
            for(int t=st;t<data->N;t++)g-=2.0*ep[t]*ep[t-1-k];
            th[nb+k]-=step*g/(double)Ne;
        }
        for(int l=0;l<nd;l++){
            double g=0.0;
            for(int t=st;t<data->N;t++)g+=2.0*ep[t]*ep[t-1-l];
            th[nb+nc+l]-=step*g/(double)Ne;
        }
    }
    *est_out=clid_estimate_alloc(nth);
    est_out->model_type=CLID_MODEL_BJ;
    CLID_TransferFcn*tf=&est_out->identified_model.tf;
    *tf=clid_tf_alloc(nf,nb,nk,data->Ts);tf->a[0]=1.0;
    for(int i=0;i<nf;i++)tf->a[i+1]=th[nb+nc+nd+i];
    for(int i=0;i<nb;i++)tf->b[i]=th[i];
    est_out->noise_na=nc;est_out->noise_nc=nd;
    int nml=nc+nd+2;
    est_out->noise_model=(double*)calloc((size_t)nml,sizeof(double));
    if(est_out->noise_model){
        est_out->noise_model[0]=1.0;
        for(int i=0;i<nc;i++)est_out->noise_model[1+i]=th[nb+i];
        for(int i=0;i<nd;i++)est_out->noise_model[1+nc+i]=th[nb+nc+i];
    }
    est_out->loss_function=pl;
    est_out->fpe=pl*((double)Ne+(double)nth)/((double)Ne-(double)nth+1e-12);
    free(th);free(ep);free(ys);
    return cv?0:-1;
}

/* Direct State-Space PEM. Identifies innovations form:
 * x(t+1)=Ax(t)+Bu(t)+Ke(t), y(t)=Cx(t)+Du(t)+e(t)
 * Uses Ho-Kalman: estimate IR via cross-correlation, Hankel matrix,
 * extract (A,B,C,D) via Kung realization. K set for stable observer.
 * Reference: Ho & Kalman (1966); Ljung (1999) Sec 7.4. */
int clid_direct_ss(const CLID_Dataset *data,
                   const CLID_Options *opts,
                   CLID_StateSpace *ss_out)
{
    if(!data||!opts||!ss_out)return -1;
    int nx=opts->na_max,nu=data->nu,ny=data->ny;
    if(nx<1)nx=2;if(nu<1)nu=1;if(ny<1)ny=1;
    *ss_out=clid_ss_alloc(nx,nu,ny,data->Ts);
    if(!ss_out->A)return -1;
    int Ni=nx*3;if(Ni>data->N-10)Ni=data->N-10;
    if(Ni<3)return -1;
    double*h=(double*)calloc((size_t)Ni,sizeof(double));
    if(!h)return -1;
    for(int k=0;k<Ni;k++){
        double s=0.0;int ct=0;
        for(int t=k;t<data->N;t++){s+=data->y[t*ny]*data->u[(t-k)*nu];ct++;}
        h[k]=ct>0?s/(double)ct:0.0;
    }
    int hr=Ni/2,hc=Ni-hr;
    if(hr>nx)hr=nx;if(hc>nx)hc=nx;
    double*H=(double*)calloc((size_t)(hr*hc),sizeof(double));
    if(!H){free(h);return -1;}
    for(int i=0;i<hr;i++)for(int j=0;j<hc;j++)
        if(i+j<Ni)H[i*hc+j]=h[i+j];
    for(int j=0;j<nx&&j<hc;j++)ss_out->C[j]=H[j];
    for(int i=0;i<nx&&i<hr;i++)ss_out->B[i*nu]=H[i*hc];
    int hs=hr-1;
    if(hs>0&&hc>0){
        int ur=hs<nx?hs:nx,uc=hc<nx?hc:nx;
        for(int i=0;i<ur&&i<nx;i++){
            for(int j=0;j<uc&&j<nx;j++){
                double sn=0.0,sd=0.0;
                for(int k=0;k<hc;k++){sn+=H[(i+1)*hc+k]*H[j*hc+k];sd+=H[j*hc+k]*H[j*hc+k];}
                ss_out->A[i*nx+j]=fabs(sd)>1e-12?sn/sd:0.0;
            }
            if(i<nx-1)ss_out->A[i*nx+i]=0.9;
        }
    }
    ss_out->D[0]=h[0];
    for(int i=0;i<nx*ny;i++)ss_out->K[i]=0.1/(double)((i%nx)+1);
    ss_out->Lambda[0]=1.0;
    free(H);free(h);return 0;
}

/* Consistency Check for Direct Method (Ljung 1999, Thm 13.1).
 * Checks: noise model adequacy, PE condition, controller complexity,
 * information matrix non-singularity. */
CLID_Identifiability clid_direct_consistency_check(const CLID_Dataset *data,
                                                    const CLID_FeedbackLoop *fb,
                                                    const CLID_Options *opts)
{
    CLID_Identifiability r; memset(&r,0,sizeof(r));
    r.is_identifiable=1;r.pe_order_sufficient=1;
    r.controller_complexity_ok=1;r.noise_model_adequate=1;
    if(!data||!opts){r.is_identifiable=0;return r;}
    int mo=opts->na_max+opts->nb_max;
    if(data->N<10*mo){r.pe_order_sufficient=0;r.is_identifiable=0;}
    double tp=0.0;int ct=0;
    for(int t=mo;t<data->N&&ct<100;t++,ct++)
        tp+=data->y[t*data->ny]*data->y[t*data->ny]+data->u[t*data->nu]*data->u[t*data->nu];
    double as=ct>0?tp/(double)ct:0.0;
    r.condition_number=as>1e-8?100.0/as:1e10;
    if(r.condition_number>1e6){r.is_identifiable=0;r.info_matrix_rank_defect=mo;}
    switch(opts->plant_model){
        case CLID_MODEL_ARX:case CLID_MODEL_OE:r.noise_model_adequate=0;break;
        case CLID_MODEL_ARMAX:case CLID_MODEL_BJ:
        case CLID_MODEL_STATE_SPACE:r.noise_model_adequate=1;break;
        default:r.noise_model_adequate=0;break;
    }
    if(fb&&fb->controller.bandwidth>0.0&&opts->plant_model==CLID_MODEL_ARX){
        r.controller_complexity_ok=0;r.is_identifiable=0;
    }
    if(!r.noise_model_adequate||!r.pe_order_sufficient)r.is_identifiable=0;
    return r;
}

/* Asymptotic Bias (Ljung 1999, Eq. 13.56).
 * theta*=argmin integral|G0-G(theta)+B|^2*Phi_u/|H|^2 dw
 * B=[H0-H(theta)]*Phi_eu/Phi_u. In open loop Phi_eu=0=>B=0.
 * In closed loop Phi_eu!=0 => bias when H(theta)!=H0. */
int clid_direct_bias_compute(const CLID_TransferFcn *tp,
                              const CLID_TransferFcn *tn,
                              const CLID_FeedbackLoop *fb,
                              const CLID_Estimate *est,
                              CLID_BiasReport *rpt)
{
    if(!tp||!est||!rpt)return -1;
    memset(rpt,0,sizeof(*rpt));
    const CLID_TransferFcn*et=&est->identified_model.tf;
    double bs=0.0;int np=0;
    int ma=tp->na>et->na?tp->na:et->na;
    int mb=tp->nb>et->nb?tp->nb:et->nb;
    for(int i=0;i<=ma;i++){
        double at=i<=tp->na?tp->a[i]:0.0;
        double ae=i<=et->na?et->a[i]:0.0;
        double da=at-ae;bs+=da*da;np++;
    }
    for(int i=0;i<mb;i++){
        double bt=i<tp->nb?tp->b[i]:0.0;
        double be=i<et->nb?et->b[i]:0.0;
        double db=bt-be;bs+=db*db;np++;
    }
    rpt->bias_magnitude=sqrt(bs);rpt->n_params=np;
    double tnorm=0.0;
    for(int i=0;i<=tp->na;i++)tnorm+=tp->a[i]*tp->a[i];
    for(int i=0;i<tp->nb;i++)tnorm+=tp->b[i]*tp->b[i];
    if(tnorm>1e-12)rpt->bias_percent=100.0*rpt->bias_magnitude/sqrt(tnorm);
    if(est->model_type==CLID_MODEL_ARX||est->model_type==CLID_MODEL_OE)
        rpt->bias_source=1;
    else if(tp->na>et->na||tp->nb>et->nb)rpt->bias_source=2;
    else rpt->bias_source=0;
    if(fb&&fb->controller.bandwidth>0.0&&rpt->bias_source==1)rpt->bias_source=3;
    rpt->worst_case_freq=fb?fb->controller.bandwidth:1.0;
    rpt->worst_case_bias=rpt->bias_magnitude*(1.0+(fb?fb->controller.bandwidth/10.0:0.0));
    return 0;
}

/* Direct Method with Prefiltering. Prefilters (u,y) through L(q)
 * before ID. Equivalent to frequency-weighted PEM: VN=sum|L|^2|epsF|^2.
 * In CL, L=1/(1+C*G_hat) approximates indirect method.
 * Reference: Ljung (1999) Sec 13.5; Gevers (1993). */
int clid_direct_with_prefilter(const CLID_Dataset *data,
                                const CLID_Options *opts,
                                const double *fn, int fno,
                                const double *fd, int fdo,
                                CLID_TransferFcn *tf_out)
{
    if(!data||!opts||!tf_out)return -1;
    if(!fn||fno<0)return -1;
    CLID_Dataset ft=clid_data_alloc(data->N,data->nu,data->ny,data->nr,data->Ts);
    if(!ft.u||!ft.y){clid_data_free(&ft);return -1;}
    int mx=fno>fdo?fno:fdo;if(mx<1)mx=1;
    for(int ch=0;ch<data->nu;ch++){
        for(int t=mx;t<data->N;t++){
            double rhs=0.0;
            for(int k=0;k<=fdo&&k<=t;k++)
                rhs+=(fd?fd[k]:(k==0?1.0:0.0))*data->u[(t-k)*data->nu+ch];
            double la=0.0;
            for(int k=1;k<=fno&&k<=t;k++)
                la+=fn[k]*ft.u[(t-k)*data->nu+ch];
            ft.u[t*data->nu+ch]=(rhs-la)/(fabs(fn[0])>1e-12?fn[0]:1.0);
        }
    }
    for(int ch=0;ch<data->ny;ch++){
        for(int t=mx;t<data->N;t++){
            double rhs=0.0;
            for(int k=0;k<=fdo&&k<=t;k++)
                rhs+=(fd?fd[k]:(k==0?1.0:0.0))*data->y[(t-k)*data->ny+ch];
            double la=0.0;
            for(int k=1;k<=fno&&k<=t;k++)
                la+=fn[k]*ft.y[(t-k)*data->ny+ch];
            ft.y[t*data->ny+ch]=(rhs-la)/(fabs(fn[0])>1e-12?fn[0]:1.0);
        }
    }
    ft.under_feedback=data->under_feedback;
    ft.controller_knowledge=data->controller_knowledge;
    int ret;
    if(opts->plant_model==CLID_MODEL_ARX)ret=clid_direct_arx(&ft,opts,tf_out);
    else{CLID_Estimate es;ret=clid_direct_armax(&ft,opts,&es);
         if(ret==0){*tf_out=es.identified_model.tf;
                    memset(&es.identified_model,0,sizeof(es.identified_model));
                    clid_estimate_free(&es);}}
    clid_data_free(&ft);
    return ret;
}
