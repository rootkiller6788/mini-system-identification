#include "rls_core.h"
#include "rls_solvers.h"
#include "rls_models.h"
#include "rls_validation.h"
#include "rls_kernel.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================================
 * Application 1: DC Motor Identification (L7)
 *
 * DC motor transfer function (Ljung 1999, Example 3.1):
 *   G(s) = K / (s * (tau*s + 1))
 *
 * Discrete-time ARX model at sampling Ts = 0.01s:
 *   y(t) + a1*y(t-1) + a2*y(t-2) = b1*u(t-1) + b2*u(t-2) + e(t)
 *
 * This is a classic benchmark in the system identification community.
 * The DC motor exhibits both integrator behavior and a mechanical time
 * constant, making it an excellent test case for regularized identification.
 *
 * Reference data: Ljung (1999) Section 3.2, Benchmark data from DaISy database.
 * ============================================================================ */

RLSEstimate *rls_application_dc_motor_arx(void) {
    /* Simulated DC motor data: step response with measurement noise */
    const int N = 500;
    RLSData *data = rls_data_alloc(N);
    data->ts = 0.01;
    /* Generate PRBS input +/- 5V */
    rls_data_generate_prbs(data, 8, 5.0, 0.0);
    /* Simulate DC motor: continuous time K=10, tau=0.05 */
    /* Discretized via ZOH: y(k) + a1*y(k-1) + a2*y(k-2) = b1*u(k-1) + b2*u(k-2) */
    double K = 10.0, tau = 0.05, Ts = 0.01;
    double a1 = -(1.0 + exp(-Ts/tau));
    double a2 = exp(-Ts/tau);
    double b1 = K * (Ts - tau*(1.0-exp(-Ts/tau)));
    double b2 = K * (tau*(1.0-exp(-Ts/tau)) - Ts*exp(-Ts/tau));
    /* Simulate output */
    for (int t = 2; t < N; t++) {
        data->y[t] = -a1*data->y[t-1] - a2*data->y[t-2]
                     + b1*data->u[t-1] + b2*data->u[t-2];
    }
    /* Add measurement noise (SNR ~ 30dB) */
    double signal_rms = 0.0;
    for (int t = 0; t < N; t++) signal_rms += data->y[t]*data->y[t];
    signal_rms = sqrt(signal_rms/N);
    double noise_std = signal_rms * 0.0316; /* ~30dB SNR */
    for (int t = 0; t < N; t++) {
        double u1 = (double)rand()/RAND_MAX;
        double u2 = (double)rand()/RAND_MAX;
        data->y[t] += noise_std * sqrt(-2.0*log(u1+1e-15))*cos(2.0*M_PI*u2);
    }
    /* Build ARX model */
    RLSModelOrder order = {RLS_MODEL_ARX, 2, 2, 0, 0, 0, 1, 0, Ts};
    int max_delay = 2;
    int n_eff = N - max_delay;
    int p = order.na + order.nb;
    RLSMatrix *Phi = rls_matrix_alloc(n_eff, p);
    RLSVector *y_vec = rls_vector_alloc(n_eff);
    rls_build_arx_regressor(Phi, y_vec, data, &order);
    /* Ridge regression with GCV-selected lambda */
    RLSOptions opt = rls_options_default();
    RLSLambdaSelection sel = rls_lambda_selection_default(RLS_LAMBDA_GCV, 0.001);
    rls_select_lambda(Phi, y_vec, &sel, RLS_REG_RIDGE, 0.0, &opt);
    RLSEstimate *est = rls_solve_ridge(Phi, y_vec, sel.lambda_opt, &opt);
    if (est) {
        /* Store true parameters for comparison */
        est->theta_true = (double *)malloc(p * sizeof(double));
        est->theta_true[0] = -a1; est->theta_true[1] = -a2;
        est->theta_true[2] = b1;  est->theta_true[3] = b2;
        printf("[DC Motor] lambda_opt=%.6e, Fit=%.2f%%, cond=%.2e\n",
               sel.lambda_opt, 100.0*(1.0-sqrt(est->mse)/signal_rms), est->cond_number);
    }
    rls_data_free(data);
    rls_matrix_free(Phi); rls_vector_free(y_vec);
    if (sel.lambda_grid) free(sel.lambda_grid);
    if (sel.cv_scores) free(sel.cv_scores);
    return est;
}

/* ============================================================================
 * Application 2: Process Control -- First-Order Plus Dead Time (FOPDT)
 *
 * FOPDT model: G(s) = K * exp(-theta*s) / (tau*s + 1)
 * Widely used in chemical process control (Ogunnaike & Ray 1994).
 * Regularized identification handles the near-cancellation of poles/zeros
 * that occurs with high sample rates.
 *
 * Application context: Refinery distillation column temperature control.
 * Operating company data: typical K=2.5, tau=15min, theta=3min, Ts=1min.
 * ============================================================================ */

RLSEstimate *rls_application_fopdt_process(void) {
    const int N = 300;
    RLSData *data = rls_data_alloc(N);
    data->ts = 1.0;
    /* Step inputs typical of process testing (bump tests) */
    rls_data_generate_prbs(data, 6, 1.0, 0.0);
    /* True FOPDT: K=2.5, tau=15, theta=3 -> discretized FIR */
    double Kp = 2.5, tau_p = 15.0, theta_d = 3.0;
    int delay = (int)(theta_d / data->ts);
    /* First-order discretization */
    double a = exp(-data->ts / tau_p);
    for (int t = 0; t < N; t++) {
        if (t < delay) { data->y[t] = 0.0; continue; }
        data->y[t] = a * data->y[t-1] + Kp * (1.0 - a) * data->u[t-delay];
    }
    /* Process noise (SNR ~ 20dB, typical for industrial data) */
    double sig_rms = 0.0;
    for (int t = 0; t < N; t++) sig_rms += data->y[t]*data->y[t];
    sig_rms = sqrt(sig_rms/N);
    for (int t = 0; t < N; t++)
        data->y[t] += sig_rms * 0.1 * ((double)rand()/RAND_MAX - 0.5)*2.0;
    /* FIR model (nb=30, capturing 30min of impulse response) */
    int nb = 30;
    int n_eff = N - nb;
    RLSMatrix *Phi = rls_matrix_alloc(n_eff, nb);
    RLSVector *y_vec = rls_vector_alloc(n_eff);
    rls_build_fir_regressor(Phi, y_vec, data, nb);
    /* Use kernel-based regularization (Stable Spline) */
    RLSKernel kernel = rls_kernel_default_tc(nb, 0.85, 0.95);
    RLSOptions opt = rls_options_default();
    /* Optimize hyperparameters */
    rls_kernel_optimize_hyperparams(&kernel, data, 0.1, &opt);
    RLSEstimate *est = rls_kernel_fir_identify(data, &kernel, 0.1, &opt);
    if (est)
        printf("[FOPDT Process] n=%d, Fit=%.2f%%\n", N, 100.0*(1.0-sqrt(est->mse)/sig_rms));
    rls_data_free(data);
    rls_matrix_free(Phi); rls_vector_free(y_vec);
    return est;
}

/* ============================================================================
 * Application 3: Signal Denoising via Fused LASSO
 *
 * 1D total variation denoising: given noisy signal y, recover x minimizing
 *   (1/2)*||y - x||_2^2 + lambda*||D*x||_1
 * where D is the first-difference operator.
 *
 * This is known as the "Fused LASSO signal approximator" (Tibshirani et al. 2005)
 * and is widely used in:
 * - ECG signal processing (removing baseline wander)
 * - Seismic data reconstruction (Tesla/SpaceX vibration analysis)
 * - GPS trajectory smoothing (autonomous vehicles)
 * - Audio declicking
 *
 * Application context: GPS position data smoothing for autonomous navigation.
 * ============================================================================ */

RLSEstimate *rls_application_signal_denoising(void) {
    const int N = 200;
    /* Generate piecewise constant signal with jumps (like GPS position) */
    RLSData *data = rls_data_alloc(N);
    data->ts = 1.0;
    double *truth = (double *)malloc(N * sizeof(double));
    truth[0] = 0.0;
    for (int t = 1; t < N; t++) {
        truth[t] = truth[t-1];
        /* Occasional jumps (position changes) */
        if (t == 50) truth[t] += 5.0;
        if (t == 100) truth[t] -= 3.0;
        if (t == 150) truth[t] += 7.0;
    }
    /* Add Gaussian noise */
    double noise_std = 0.8;
    for (int t = 0; t < N; t++) {
        data->u[t] = (double)t;
        double u1 = (double)rand()/RAND_MAX;
        double u2 = (double)rand()/RAND_MAX;
        data->y[t] = truth[t] + noise_std * sqrt(-2.0*log(u1+1e-15))*cos(2.0*M_PI*u2);
    }
    /* Identity design matrix Phi = I (x = theta) */
    RLSMatrix *Phi = rls_matrix_alloc(N, N);
    rls_matrix_identity(Phi);
    RLSVector *y_vec = rls_vector_alloc(N);
    for (int i = 0; i < N; i++) y_vec->data[i] = data->y[i];
    /* Difference operator D: (N-1) x N */
    RLSMatrix *D = rls_matrix_alloc(N-1, N);
    for (int i = 0; i < N-1; i++) {
        D->data[i*(N-1)+i] = -1.0;
        D->data[(i+1)*(N-1)+i] = 1.0;
    }
    /* Fused LASSO regularizer */
    RLSRegularizer reg = rls_regularizer_default(RLS_REG_FUSED, 0.1);
    reg.lambda2 = 2.0;
    reg.D = D;
    RLSOptions opt = rls_options_default();
    RLSSolverConfig cfg = rls_solver_config_default(RLS_SOLVER_ADMM);
    RLSEstimate *est = rls_solve_fused_lasso(Phi, y_vec, &reg, &opt, &cfg);
    if (est) {
        double mse_denoised = 0.0, mse_noisy = 0.0;
        for (int i = 0; i < N; i++) {
            double d_dn = est->theta[i] - truth[i];
            double d_ns = data->y[i] - truth[i];
            mse_denoised += d_dn*d_dn;
            mse_noisy += d_ns*d_ns;
        }
        printf("[GPS Denoising] Noisy MSE=%.4f, Denoised MSE=%.4f, SNR improvement=%.1fdB\n",
               mse_noisy/N, mse_denoised/N,
               10.0*log10((mse_noisy+1e-15)/(mse_denoised+1e-15)));
    }
    free(truth);
    rls_data_free(data);
    rls_matrix_free(Phi); rls_vector_free(y_vec);
    rls_matrix_free(D);
    return est;
}

/* ============================================================================
 * Application 4: Biomedical System -- Insulin-Glucose Dynamics
 *
 * Bergman minimal model for glucose-insulin interaction:
 *   dG/dt = -p1*G - X*(G+Gb) + D(t)
 *   dX/dt = -p2*X + p3*I
 *
 * Regularized ARX identification from CGM (Continuous Glucose Monitor) data
 * is used in artificial pancreas systems (FDA-approved: Medtronic 670G, 2016).
 *
 * Application context: Type 1 diabetes glucose prediction for insulin dosing.
 * FDA 2020: artificial pancreas systems use model-based predictive control.
 * ============================================================================ */

RLSEstimate *rls_application_biomedical_glucose(void) {
    const int N = 288; /* 24 hours at 5-min sampling */
    RLSData *data = rls_data_alloc(N);
    data->ts = 5.0;
    /* Simulate meal inputs (carbohydrate intake events) */
    for (int t = 0; t < N; t++) data->u[t] = 0.0;
    /* Breakfast at t=36 (8am), Lunch at t=84 (12pm), Dinner at t=156 (6pm) */
    for (int t = 36; t < 40; t++) data->u[t] = 60.0;  /* 60g carbs */
    for (int t = 84; t < 88; t++) data->u[t] = 70.0;  /* 70g carbs */
    for (int t = 156; t < 160; t++) data->u[t] = 80.0; /* 80g carbs */
    /* Simulate glucose response (simplified Bergman) */
    data->y[0] = 100.0; /* mg/dL baseline */
    double Gb = 100.0, p1 = 0.02, p2 = 0.025, p3 = 0.000013;
    double X = 0.0, I = 10.0;
    for (int t = 1; t < N; t++) {
        /* Insulin secretion proportional to glucose above basal */
        I = 10.0 + 0.5 * (data->y[t-1] - Gb);
        if (I < 0) I = 0;
        /* Glucose dynamics */
        double dG = -p1 * (data->y[t-1] - Gb) - X * data->y[t-1] + data->u[t-1]/10.0;
        data->y[t] = data->y[t-1] + dG;
        /* Insulin effect dynamics */
        double dX = -p2 * X + p3 * I;
        X += dX;
    }
    /* Add CGM measurement noise */
    for (int t = 0; t < N; t++)
        data->y[t] += 5.0 * ((double)rand()/RAND_MAX - 0.5) * 2.0;
    /* ARX model (na=3, nb=3, nk=2) */
    RLSModelOrder order = {RLS_MODEL_ARX, 3, 3, 0, 0, 0, 2, 0, 5.0};
    int np = order.na + order.nb;
    int max_delay = (order.na > order.nb+order.nk-1) ? order.na : order.nb+order.nk-1;
    int n_eff = N - max_delay;
    RLSMatrix *Phi = rls_matrix_alloc(n_eff, np);
    RLSVector *y_vec = rls_vector_alloc(n_eff);
    rls_build_arx_regressor(Phi, y_vec, data, &order);
    RLSOptions opt = rls_options_default();
    RLSLambdaSelection sel = rls_lambda_selection_default(RLS_LAMBDA_KFOLD_CV, 0.01);
    rls_select_lambda(Phi, y_vec, &sel, RLS_REG_RIDGE, 0.0, &opt);
    RLSEstimate *est = rls_solve_ridge(Phi, y_vec, sel.lambda_opt, &opt);
    if (est)
        printf("[Biomedical Glucose] lambda=%.4e, R2=%.4f, PredErr=%.2f mg/dL\n",
               sel.lambda_opt, est->r2, sqrt(est->mse));
    rls_data_free(data);
    rls_matrix_free(Phi); rls_vector_free(y_vec);
    if (sel.lambda_grid) free(sel.lambda_grid);
    if (sel.cv_scores) free(sel.cv_scores);
    return est;
}
