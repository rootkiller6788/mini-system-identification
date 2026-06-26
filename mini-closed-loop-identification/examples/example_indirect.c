/**
 * example_indirect.c - Indirect closed-loop identification of a quadrotor attitude loop
 * L7 Application: Quadrotor UAV attitude control identification.
 */
#include "clid_types.h"
#include "clid_indirect.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
int main(void){
    printf("=== Indirect CL ID: Quadrotor Attitude Loop ===

");
    int N=300;
    CLID_Dataset d=clid_data_alloc(N,1,1,1,0.02);
    if(!d.u||!d.y||!d.r){printf("Alloc failed
");return 1;}
    d.under_feedback=1;d.controller_knowledge=2;
    /* Quadrotor pitch axis: G(s)=10/(s^2+2s+5) -> discrete G(z) */
    double y=0.0,y1=0.0,u=0.0,v=0.0;
    for(int t=0;t<N;t++){
        double r=sin(0.05*(double)t)*0.1;
        double e=((double)rand()/(double)RAND_MAX-0.5)*0.02;
        v=0.5*v+e;
        if(t==0){u=r;y=0.0;y1=0.0;}
        else{
            double u_pid=2.0*(r-y);
            u=u_pid;
            /* Second-order discrete: y=1.5y1-0.6y0+0.1u1+0.05u0+v */
            y=1.5*y1-0.6*d.y[t-2<0?0:t-2]+0.1*u+0.05*d.u[t-1]+v;
            y1=d.y[t-1];
        }
        d.r[t]=r;d.u[t]=u;d.y[t]=y;
    }
    printf("Data generated: N=%d, Ts=0.02s
",N);
    printf("True plant: 2nd order (quadrotor pitch)

");

    CLID_Controller ctrl;memset(&ctrl,0,sizeof(ctrl));
    ctrl.form.tf=clid_tf_alloc(0,1,0,0.02);ctrl.form.tf.b[0]=2.0;
    CLID_Options opts=clid_options_default();
    opts.na_max=3;opts.nb_max=3;
    CLID_Estimate est;
    if(clid_indirect_two_step(&d,&ctrl,&opts,&est)==0){
        printf("Indirect Method Results:
");
        printf("  a1=%.4f a2=%.4f
",est.identified_model.tf.a[1],est.identified_model.tf.a[2]);
        printf("  b0=%.4f b1=%.4f
",est.identified_model.tf.b[0],est.identified_model.tf.b[1]);
        printf("  Fit=%.2f%% Loss=%.6f
",est.fit_percent,est.loss_function);
        clid_estimate_free(&est);
    }else{printf("Indirect failed
");}
    clid_tf_free(&ctrl.form.tf);clid_data_free(&d);
    return 0;
}
