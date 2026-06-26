/**
 * example_joint_io.c - Joint IO identification of a process control loop
 * L7 Application: Industrial process control (ISO standard).
 */
#include "clid_types.h"
#include "clid_joint_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
int main(void){
    printf("=== Joint IO CL ID: Process Control Loop ===

");
    int N=400;
    CLID_Dataset d=clid_data_alloc(N,1,1,1,0.1);
    if(!d.u||!d.y||!d.r){printf("Alloc failed
");return 1;}
    d.under_feedback=1;d.controller_knowledge=0;
    /* Process: G(s)=exp(-s)/(5s+1) discrete approx: y(t)=0.8y(t-1)+0.15u(t-2)+noise */
    double y=0.0,u=0.0,v=0.0;
    for(int t=0;t<N;t++){
        double r=((double)rand()/(double)RAND_MAX-0.5)*1.0;
        double e=((double)rand()/(double)RAND_MAX-0.5)*0.1;
        v=0.3*v+e;
        if(t==0){u=r;y=0.0;}
        else{
            u=r-0.4*y;
            if(t>=2)y=0.8*d.y[t-1]+0.15*d.u[t-2]+v;
            else y=0.8*d.y[t-1]+0.15*u+v;
        }
        d.r[t]=r;d.u[t]=u;d.y[t]=y;
    }
    printf("Data generated: N=%d, Ts=0.1s (process with delay)
",N);
    printf("Controller unknown -> Joint IO needed

");

    CLID_Options opts=clid_options_default();
    opts.na_max=2;opts.nb_max=5;
    CLID_Estimate est;
    if(clid_joint_io_spectral(&d,&opts,&est)==0){
        printf("Joint IO Spectral Results:
");
        printf("  a1=%.4f
",est.identified_model.tf.a[1]);
        printf("  b0=%.4f b1=%.4f
",est.identified_model.tf.b[0],est.identified_model.tf.b[1]);
        printf("  Fit=%.2f%%
",est.fit_percent);
        clid_estimate_free(&est);
    }else{printf("Spectral method failed
");}
    clid_data_free(&d);
    return 0;
}
