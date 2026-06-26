#include "nlsid_validation.h"
#include "nlsid_models.h"
#include "nlsid_algorithms.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

void nlsid_residual_statistics(const double* r, int n, double* mean, double* var, double* skew, double* kurt) {
    if (!r || n <= 0) return;
    double m = 0.0; for (int i = 0; i < n; i++) m += r[i]; m /= n;
    if (mean) *mean = m;
    double v = 0.0; for (int i = 0; i < n; i++) { double d = r[i] - m; v += d * d; } v /= (n - 1);
    if (var) *var = v;
    if (v < 1e-15) { if (skew) *skew = 0.0; if (kurt) *kurt = 3.0; return; }
    double s = sqrt(v), sk = 0.0, kt = 0.0;
    for (int i = 0; i < n; i++) { double z = (r[i] - m) / s; sk += z * z * z; kt += z * z * z * z; }
    if (skew) *skew = sk / n; if (kurt) *kurt = kt / n;
}

void nlsid_autocorrelation_test(const double* e, int n, int ml, double* re, double* cb, bool* iw, double* qs) {
    if (!e || n <= 1) return;
    double em = 0.0; for (int i = 0; i < n; i++) em += e[i]; em /= n;
    double ev = 0.0; for (int i = 0; i < n; i++) { double d = e[i] - em; ev += d * d; } ev /= n;
    if (ev < 1e-15) ev = 1e-15;
    double conf = 1.96 / sqrt((double)n); if (cb) *cb = conf;
    double Q = 0.0; bool wh = true;
    for (int lag = 0; lag <= ml && lag < n; lag++) {
        double cov = 0.0; for (int t = 0; t < n - lag; t++) cov += (e[t] - em) * (e[t + lag] - em);
        cov /= n; double acf = cov / ev; if (re) re[lag] = acf;
        if (lag > 0) { Q += acf * acf / (n - lag); if (fabs(acf) > conf) wh = false; }
    }
    Q *= n * (n + 2.0); if (qs) *qs = Q; if (iw) *iw = wh;
}

void nlsid_crosscorrelation_test(const double* e, const double* u, int n, int ml, double* re, double* cb, bool* ii) {
    if (!e || !u || n <= 1) return;
    double em = 0, um = 0; for (int i = 0; i < n; i++) { em += e[i]; um += u[i]; } em /= n; um /= n;
    double ev = 0, uv = 0; for (int i = 0; i < n; i++) { ev += (e[i]-em)*(e[i]-em); uv += (u[i]-um)*(u[i]-um); } ev /= n; uv /= n;
    double dn = sqrt(ev * uv); if (dn < 1e-15) dn = 1e-15;
    double conf = 1.96 / sqrt((double)n); if (cb) *cb = conf;
    bool ind = true;
    for (int lag = -ml; lag <= ml; lag++) {
        double cov = 0.0; for (int t = 0; t < n; t++) { int idx = t + lag; if (idx < 0 || idx >= n) continue; cov += (e[t] - em) * (u[idx] - um); }
        cov /= n; double ccf = cov / dn; int li = lag + ml; if (re && li >= 0 && li < 2*ml+1) re[li] = ccf;
        if (fabs(ccf) > conf) ind = false;
    }
    if (ii) *ii = ind;
}

double nlsid_ljung_box_test(const double* r, int n, int ml, double* qs) {
    if (!r || n <= ml) return 1.0;
    double* acf = (double*)calloc(ml + 1, sizeof(double));
    double m = 0, v = 0; for (int i = 0; i < n; i++) m += r[i]; m /= n; for (int i = 0; i < n; i++) { double d = r[i] - m; v += d * d; } v /= n;
    if (v < 1e-15) { free(acf); return 1.0; }
    for (int lg = 0; lg <= ml; lg++) { double cv = 0; for (int t = 0; t < n - lg; t++) cv += (r[t] - m) * (r[t + lg] - m); acf[lg] = (cv / n) / v; }
    double Q = 0.0; for (int k = 1; k <= ml; k++) Q += acf[k] * acf[k] / (n - k); Q *= n * (n + 2.0); if (qs) *qs = Q;
    double p = exp(-Q / (2.0 * ml)); if (p > 1.0) p = 1.0; free(acf); return p;
}

double nlsid_aic(double ve, int N, int d) {
    if (N <= 0 || ve <= 0.0) return 1e100;
    return (double)N * log(ve) + 2.0 * (double)d;
}
double nlsid_aicc(double ve, int N, int d) {
    if (N <= d + 1) return 1e100;
    double a = nlsid_aic(ve, N, d);
    return a + 2.0 * (double)d * ((double)d + 1.0) / ((double)N - (double)d - 1.0);
}
double nlsid_bic(double ve, int N, int d) {
    if (N <= 0 || ve <= 0.0) return 1e100;
    return (double)N * log(ve) + (double)d * log((double)N);
}
double nlsid_mdl(double ve, int N, int d) {
    if (N <= 0 || ve <= 0.0) return 1e100;
    return -log(ve) * N / 2.0 + (double)d * log((double)N) / 2.0;
}
double nlsid_fpe(double ve, int N, int d) {
    if (N <= d || N <= 0) return 1e100;
    double dn = (double)d, DN = (double)N;
    return ve * (1.0 + dn / DN) / (1.0 - dn / DN);
}
double nlsid_hannan_quinn(double ve, int N, int d) {
    if (N <= 0 || ve <= 0.0) return 1e100;
    return (double)N * log(ve) + 2.0 * (double)d * log(log((double)N));
}
double nlsid_f_test(double vr, double vf, int dr, int df, int N, double* pv) {
    if (df <= dr || N <= df) { if (pv) *pv = 1.0; return 0.0; }
    double num = (vr - vf) / (double)(df - dr);
    double den = vf / (double)(N - df);
    if (den < 1e-15) { if (pv) *pv = 0.0; return 1e100; }
    double F = num / den; if (pv) *pv = exp(-F * 0.5); return F;
}
int nlsid_select_narx_order(const NLSIDDataset* ds, int nym, int num, int nkm,
    BasisExpansion* te, NLSIDConfig* cf, int cr, int* bny, int* bnu, int* bnk, double* bcv) {
    if (!ds || !bny || !bnu || !bnk || !bcv) return -1;
    double best = 1e100; *bny = 1; *bnu = 1; *bnk = 1;
    for (int ny = 1; ny <= nym; ny++)
        for (int nu = 1; nu <= num; nu++)
            for (int nk = 1; nk <= nkm; nk++) {
                NARXModel* narx = narx_create(ny, nu, nk, 1, 1);
                if (!narx) continue;
                BasisExpansion* exp = basis_expansion_create(te->input_dim, te->n_bases);
                narx_set_expansion(narx, exp);
                double c = (cr == 0) ? nlsid_aic(1.0, ds->n_samples, narx->n_params)
                                     : nlsid_bic(1.0, ds->n_samples, narx->n_params);
                if (c < best) { best = c; *bny = ny; *bnu = nu; *bnk = nk; }
                narx_free(narx);
            }
    *bcv = best; return 0;
}
double nlsid_structural_risk(double er, int d, int N, double cf) {
    (void)cf; double DN = (double)N, h = (double)d;
    if (DN <= h) return 1e100;
    return er + sqrt((h * (log(2.0 * DN / h) + 1.0)) / DN);
}
void nlsid_residual_acf_pacf(const double* e, int n, int ml, double* acf, double* pacf) {
    if (!e || n <= 1) return;
    double m = 0.0, v = 0.0; for (int i = 0; i < n; i++) m += e[i]; m /= n;
    for (int i = 0; i < n; i++) { double d = e[i] - m; v += d * d; } v /= n;
    for (int lg = 0; lg <= ml; lg++) {
        double cv = 0.0; for (int t = 0; t < n - lg; t++) cv += (e[t] - m) * (e[t + lg] - m);
        if (acf) acf[lg] = (cv / n) / (v > 1e-15 ? v : 1.0);
    }
    if (pacf) {
        pacf[0] = 1.0;
        for (int k = 1; k <= ml; k++) {
            double num = acf[k], den = 1.0;
            for (int j = 1; j < k; j++) { num -= pacf[j] * acf[k - j]; den -= pacf[j] * acf[j]; }
            pacf[k] = (fabs(den) > 1e-15) ? num / den : 0.0;
        }
    }
}

bool nlsid_narmax_validation_tests(const double* e, const double* u, int n, int ml, bool* tr) {
    if (!e || !u || n <= ml) return false;
    bool all = true; int L = ml;
    { double p[101], c; bool pa; nlsid_test_phi_ee(e,n,L,p,&c,&pa); if(tr)tr[0]=pa; if(!pa)all=false; }
    { double p[201], c; bool pa; nlsid_test_phi_eu(e,u,n,L,p,&c,&pa); if(tr)tr[1]=pa; if(!pa)all=false; }
    { double p[101], c; bool pa; nlsid_test_phi_e_ue(e,u,n,L,p,&c,&pa); if(tr)tr[2]=pa; if(!pa)all=false; }
    { double p[101], c; bool pa; nlsid_test_phi_u2_e2(e,u,n,L,p,&c,&pa); if(tr)tr[3]=pa; if(!pa)all=false; }
    { double p[101], c; bool pa; nlsid_test_phi_e_eu(e,u,n,L,p,&c,&pa); if(tr)tr[4]=pa; if(!pa)all=false; }
    return all;
}

void nlsid_test_phi_ee(const double* e, int n, int L, double* p, double* c, bool* pa) {
    double cb; bool w; nlsid_autocorrelation_test(e,n,L,p,&cb,&w,NULL);
    if(c)*c=cb; bool ok=true;
    for(int lg=1;lg<=L&&ok;lg++)if(fabs(p[lg])>cb)ok=false;
    if(pa)*pa=ok;
}

void nlsid_test_phi_eu(const double* e, const double* u, int n, int L, double* p, double* c, bool* pa) {
    bool ind; nlsid_crosscorrelation_test(e,u,n,L,p,c,&ind); if(pa)*pa=ind;
}

void nlsid_test_phi_e_ue(const double* e, const double* u, int n, int L, double* p, double* c, bool* pa) {
    double* ue=(double*)malloc((size_t)n*sizeof(double));
    for(int t=0;t<n;t++)ue[t]=u[t]*e[t];
    bool ind; nlsid_crosscorrelation_test(e,ue,n,L,p,c,&ind);
    bool ok=true; double cb=c?*c:0.0;
    for(int t=0;t<=L&&ok;t++){int li=t+L;if(p&&li<2*L+1&&fabs(p[li])>cb)ok=false;}
    if(pa)*pa=ok; free(ue);
}

void nlsid_test_phi_u2_e2(const double* e, const double* u, int n, int L, double* p, double* c, bool* pa) {
    double* u2=(double*)malloc((size_t)n*sizeof(double));
    double* e2=(double*)malloc((size_t)n*sizeof(double));
    double um=0,em=0;for(int t=0;t<n;t++){u2[t]=u[t]*u[t];um+=u2[t];e2[t]=e[t]*e[t];em+=e2[t];}
    um/=n;em/=n;for(int t=0;t<n;t++){u2[t]-=um;e2[t]-=em;}
    bool ind; nlsid_crosscorrelation_test(u2,e2,n,L,p,c,&ind);
    bool ok=true;double cb=c?*c:0.0;
    for(int t=0;t<=L&&ok;t++){int li=t+L;if(p&&li<2*L+1&&fabs(p[li])>cb)ok=false;}
    if(pa)*pa=ok;free(u2);free(e2);
}

void nlsid_test_phi_e_eu(const double* e, const double* u, int n, int L, double* p, double* c, bool* pa) {
    double* eu=(double*)malloc((size_t)n*sizeof(double));
    for(int t=0;t<n;t++)eu[t]=e[t]*u[t];
    bool ind; nlsid_crosscorrelation_test(e,eu,n,L,p,c,&ind);
    bool ok=true;double cb=c?*c:0.0;
    for(int t=0;t<=L&&ok;t++){int li=t+L;if(p&&li<2*L+1&&fabs(p[li])>cb)ok=false;}
    if(pa)*pa=ok;free(eu);
}

void nlsid_k_step_ahead_validation(const NLSIDModel* m, const NLSIDDataset* ds, int mk, double* fvk) {
    if(!m||!ds||!fvk||mk<1)return;
    const double*u=ds->input->channels[0]->data,*y=ds->output->channels[0]->data;int N=ds->n_samples;
    for(int k=1;k<=mk;k++){double*yh=(double*)malloc((size_t)N*sizeof(double));
    for(int t=0;t<N;t++){double yhv;if(t>=k){m->predict_one_step((NLSIDModel*)m,u,y,t,&yhv);yh[t]=yhv;}else yh[t]=y[t];}
    fvk[k-1]=nlsid_compute_fit(y,yh,N);free(yh);}
}

double nlsid_simulation_stability_test(const NLSIDModel* m,const double* ut,int nt,int ni,unsigned int* seed){
    if(!m||!ut||nt<=0||ni<=0)return 0.0;
    int st=0;unsigned int s=seed?*seed:1;
    for(int tr=0;tr<ni;tr++){s=s*1103515245+12345;double y0=((s&0xFFFF)/65535.0-0.5)*2.0,yp=y0;bool bd=true;
    for(int t=0;t<nt&&bd;t++){double uv=t<nt?ut[t]:0.0,yh;m->predict_one_step((NLSIDModel*)m,&uv,&yp,1,&yh);if(fabs(yh)>1e6)bd=false;yp=yh;}
    if(bd)st++;}
    if(seed)*seed=s;return 100.0*(double)st/(double)ni;
}

double nlsid_output_error(const NLSIDModel* m,const NLSIDDataset* ds,double* ys){
    if(!m||!ds)return 0.0;
    const double*u=ds->input->channels[0]->data,*y=ds->output->channels[0]->data;int N=ds->n_samples;
    if(m->simulate){m->simulate((NLSIDModel*)m,u,N,ys);}
    else{for(int t=0;t<N;t++){double yh;m->predict_one_step((NLSIDModel*)m,u,y,t,&yh);ys[t]=yh;}}
    return nlsid_compute_mse(y,ys,N);
}

int nlsid_fit_linear_arx(const NLSIDDataset* ds,int na,int nb,int nk,double* a,double* b,double* ft,double* ms){
    if(!ds||na<=0||nb<=0)return -1;
    const double*u=ds->input->channels[0]->data,*y=ds->output->channels[0]->data;int N=ds->n_samples;
    int nr=na+nb,stt=(na>nb+nk)?na:nb+nk,ne=N-stt;if(ne<nr)return -1;
    double*Phi=(double*)calloc((size_t)(ne*nr),sizeof(double)),*Y=(double*)calloc((size_t)ne,sizeof(double));
    for(int t=stt;t<N;t++){int r=t-stt;for(int i=0;i<na;i++)Phi[r*nr+i]=y[t-1-i];
    for(int i=0;i<nb;i++){int idx=t-nk-i;Phi[r*nr+na+i]=(idx>=0)?u[idx]:0.0;}Y[r]=y[t];}
    double*PT=(double*)calloc((size_t)(nr*nr),sizeof(double)),*PY=(double*)calloc((size_t)nr,sizeof(double));
    for(int i=0;i<nr;i++){for(int t=0;t<ne;t++)PY[i]+=Phi[t*nr+i]*Y[t];
    for(int j=0;j<nr;j++)for(int t=0;t<ne;t++)PT[i*nr+j]+=Phi[t*nr+i]*Phi[t*nr+j];}
    double*th=(double*)calloc((size_t)nr,sizeof(double));nlsid_solve_linear_system(PT,PY,nr,th);
    double*yh=(double*)calloc((size_t)N,sizeof(double));
    for(int t=0;t<N;t++){double s=0;for(int i=0;i<na&&t-1-i>=0;i++)s+=th[i]*y[t-1-i];
    for(int i=0;i<nb&&t-nk-i>=0;i++)s+=th[na+i]*u[t-nk-i];yh[t]=s;}
    if(ft)*ft=nlsid_compute_fit(y,yh,N);if(ms)*ms=nlsid_compute_mse(y,yh,N);
    if(a)memcpy(a,th,(size_t)na*sizeof(double));if(b)memcpy(b,th+na,(size_t)nb*sizeof(double));
    free(Phi);free(Y);free(PT);free(PY);free(th);free(yh);return 0;
}

bool nlsid_test_nonlinear_significance(const NLSIDDataset* ds,int na,int nb,int nk,const NLSIDModel* nm,double* pv){
    double la[10],lb[10],lf,lm;nlsid_fit_linear_arx(ds,na,nb,nk,la,lb,&lf,&lm);
    const double*u=ds->input->channels[0]->data,*y=ds->output->channels[0]->data;int N=ds->n_samples;
    double*yh=(double*)malloc((size_t)N*sizeof(double));
    for(int t=0;t<N;t++){double yhv;nm->predict_one_step((NLSIDModel*)nm,u,y,t,&yhv);yh[t]=yhv;}
    double nmse=nlsid_compute_mse(y,yh,N);free(yh);
    double f=nlsid_f_test(lm*N,nmse*N,na+nb,nm->n_params,N,pv);return f>3.0;
}

double nlsid_nonlinearity_contribution_ratio(const NLSIDDataset* ds,int na,int nb,int nk,const NLSIDModel* nm){
    double la[10],lb[10],lf,lm;nlsid_fit_linear_arx(ds,na,nb,nk,la,lb,&lf,&lm);
    const double*u=ds->input->channels[0]->data,*y=ds->output->channels[0]->data;int N=ds->n_samples;
    double*yh=(double*)malloc((size_t)N*sizeof(double));
    for(int t=0;t<N;t++){double yhv;nm->predict_one_step((NLSIDModel*)nm,u,y,t,&yhv);yh[t]=yhv;}
    double nmse=nlsid_compute_mse(y,yh,N);free(yh);if(lm<1e-15)return 0.0;
    double eta=(lm-nmse)/lm;return(eta>0.0)?eta:0.0;
}

void nlsid_parameter_standard_errors(const NLSIDModel* m,const NLSIDDataset* ds,double rv,double* se){
    if(!m||!ds||!se)return;int np=m->n_params;if(np<=0)return;
    double**J=NULL;int nr,nc;if(nlsid_compute_jacobian(m,ds,&J,&nr,&nc)!=0){for(int i=0;i<np;i++)se[i]=sqrt(rv);return;}
    double*H=(double*)calloc((size_t)(np*np),sizeof(double));
    for(int i=0;i<np;i++)for(int j=0;j<np;j++)for(int t=0;t<nr;t++)H[i*np+j]+=J[t][i]*J[t][j];
    double*Hi=(double*)malloc((size_t)(np*np)*sizeof(double));
    if(nlsid_matrix_inverse(H,np,Hi)==0){for(int i=0;i<np;i++)se[i]=sqrt(rv*fabs(Hi[i*np+i]));}
    else{for(int i=0;i<np;i++)se[i]=sqrt(rv);}
    for(int t=0;t<nr;t++)free(J[t]);free(J);free(H);free(Hi);
}

void nlsid_parameter_confidence_intervals(const double* th,const double* se,int np,double* lo,double* hi){
    if(!th||!se||!lo||!hi)return;
    for(int i=0;i<np;i++){lo[i]=th[i]-1.96*se[i];hi[i]=th[i]+1.96*se[i];}
}

void nlsid_parameter_t_test(const double* th,const double* se,int np,int nd,double* pv){
    (void)nd;if(!th||!se||!pv)return;
    for(int i=0;i<np;i++){double s=se[i];if(s<1e-15){pv[i]=0.0;continue;}
    double t=fabs(th[i])/s;pv[i]=2.0*(1.0-0.5*(1.0+tanh(t/sqrt(2.0))));
    if(pv[i]<0.0)pv[i]=0.0;if(pv[i]>1.0)pv[i]=1.0;}
}

NLSIDValidationReport* nlsid_validate(const NLSIDModel* m,const NLSIDDataset* est,const NLSIDDataset* val,int ml){
    if(!m||!est)return NULL;
    NLSIDValidationReport* rpt=(NLSIDValidationReport*)calloc(1,sizeof(NLSIDValidationReport));if(!rpt)return NULL;
    const double*u=est->input->channels[0]->data,*y=est->output->channels[0]->data;int N=est->n_samples;
    double*yh=(double*)malloc((size_t)N*sizeof(double));
    for(int t=0;t<N;t++){double yhv;m->predict_one_step((NLSIDModel*)m,u,y,t,&yhv);yh[t]=yhv;}
    rpt->fit_estimation=nlsid_compute_fit(y,yh,N);double mse=nlsid_compute_mse(y,yh,N);
    double*e=(double*)malloc((size_t)N*sizeof(double));for(int t=0;t<N;t++)e[t]=y[t]-yh[t];
    nlsid_residual_statistics(e,N,&rpt->residual_mean,&rpt->residual_variance,NULL,NULL);
    double qs;double pw=nlsid_ljung_box_test(e,N,ml,&qs);
    rpt->residual_whiteness_q=qs;rpt->residual_whiteness_pvalue=pw;rpt->residual_is_white=(pw>0.05);
    bool ind;nlsid_crosscorrelation_test(e,u,N,ml,NULL,NULL,&ind);rpt->crosscorr_independent=ind;
    rpt->aic=nlsid_aic(mse,N,m->n_params);rpt->bic=nlsid_bic(mse,N,m->n_params);
    rpt->aicc=nlsid_aicc(mse,N,m->n_params);rpt->mdl=nlsid_mdl(mse,N,m->n_params);
    rpt->fpe=nlsid_fpe(mse,N,m->n_params);rpt->hq=nlsid_hannan_quinn(mse,N,m->n_params);
    if(val&&val->n_samples>0){const double*uv=val->input->channels[0]->data,*yv=val->output->channels[0]->data;int Nv=val->n_samples;
    double*yhv2=(double*)malloc((size_t)Nv*sizeof(double));
    for(int t=0;t<Nv;t++){double yhv2v;m->predict_one_step((NLSIDModel*)m,uv,yv,t,&yhv2v);yhv2[t]=yhv2v;}
    rpt->fit_validation=nlsid_compute_fit(yv,yhv2,Nv);free(yhv2);}
    bool tr[5];rpt->all_narmax_passed=nlsid_narmax_validation_tests(e,u,N,ml,tr);
    rpt->test1_residual_white=tr[0];rpt->test2_eu_independent=tr[1];
    rpt->test3_e_ue=tr[2];rpt->test4_u2_e2=tr[3];rpt->test5_e_eu=tr[4];
    double sc=0;if(rpt->residual_is_white)sc+=0.25;if(rpt->crosscorr_independent)sc+=0.25;
    if(rpt->fit_estimation>50.0)sc+=0.25;if(rpt->all_narmax_passed)sc+=0.25;
    rpt->overall_score=sc;rpt->model_accepted=(sc>=0.5);free(yh);free(e);return rpt;
}

void nlsid_validation_report_free(NLSIDValidationReport* r){free(r);}

void nlsid_validation_report_print(const NLSIDValidationReport* r){
    if(!r){printf("ValidationReport: NULL\n");return;}
    printf("\n===== Model Validation Report =====\n");
    printf("  Accepted=%s score=%.2f\n",r->model_accepted?"YES":"NO",r->overall_score);
    printf("  Fit: est=%.1f%% val=%.1f%%\n",r->fit_estimation,r->fit_validation);
    printf("  AIC=%.1f BIC=%.1f\n",r->aic,r->bic);
    printf("  NARMAX all=%s\n",r->all_narmax_passed?"YES":"NO");
    printf("====================================\n");
}
