/**
 * example_direct.c - Direct closed-loop identification of a DC motor
 * Demonstrates direct ARMAX identification on closed-loop data.
 * L7 Application: DC motor servo control identification.
 */
#include "clid_types.h"
#include "clid_direct.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
int main(void){
    printf("=== Direct Closed-Loop ID: DC Motor Example ===

");
    int N=200;
    CLID_Dataset d=clid_data_alloc(N,1,1,1,0.01);
    if(!d.u||!d.y||!d.r){printf("Alloc failed
");return 1;}
    d.under_feedback=1;d.controller_knowledge=2;
    /* DC motor: G(s)=K/(tau*s+1) discretized: G(z)=0.1/(1-0.9z^{-1}) */
    /* Controller: PI with Kp=0.5, Ki=0.1 */
    double y=0.0,u=0.0,v=0.0,y_int=0.0;
    for(int t=0;t<N;t++){
        double r=sin(0.1*(double)t);
        double e_noise=((double)rand()/(double)RAND_MAX-0.5)*0.05;
        v=0.2*v+e_noise;
        if(t==0){u=r;y=0.0;}
        else{
            y_int+=0.01*(r-y);
            u=0.5*(r-y)+0.1*y_int;
            y=0.9*d.y[t-1]+0.1*d.u[t-1]+v;
        }
        d.r[t]=r;d.u[t]=u;d.y[t]=y;
    }
    printf("Data generated: N=%d samples, Ts=0.01s
",N);
    printf("True plant: G(z)=0.1/(1-0.9z^{-1})
");
    printf("Controller: PI (Kp=0.5, Ki=0.1)

");

    CLID_Options opts=clid_options_default();
    opts.plant_model=CLID_MODEL_ARMAX;
    opts.na_max=2;opts.nb_max=2;opts.nk=1;
    CLID_Estimate est;
    if(clid_direct_armax(&d,&opts,&est)==0){
        printf("Direct ARMAX Results:
");
        printf("  a1 = %.4f (true: -0.9)
",est.identified_model.tf.a[1]);
        printf("  b0 = %.4f (true: 0.1)
",est.identified_model.tf.b[0]);
        printf("  Loss = %.6f
",est.loss_function);
        printf("  Fit  = %.2f%%
",est.fit_percent);
        printf("  FPE  = %.6f
",est.fpe);
        clid_estimate_free(&est);
    }else{printf("ARMAX failed
");}
    clid_data_free(&d);
    return 0;
}
