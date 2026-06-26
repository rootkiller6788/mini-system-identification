#include "pem_core.h"
#include "pem_predictor.h"
#include "pem_criterion.h"
#include "pem_model.h"
#include "pem_optimizer.h"
#include "pem_validation.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <assert.h>

/* Tolerance for floating-point comparisons */
#define TOL 1e-6
#define TOL_SOFT 1e-3

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) do { tests_run++; printf("  TEST %s... ", name); } while(0)
#define PASS() do { tests_passed++; printf("PASSED\n"); } while(0)
#define FAIL(msg) do { printf("FAILED: %s\n", msg); } while(0)
#define CHECK(cond, msg) do { if (!(cond)) { FAIL(msg); return; } } while(0)
#define CHECK_CLOSE(a, b, tol, msg) do { if (fabs((a)-(b)) > (tol)) { \
    printf("FAILED: %s (%.6e vs %.6e)\n", msg, a, b); return; } } while(0)

/* Test 1: Polynomial operations */
static void test_polynomial(void) {
    TEST("polynomial evaluation");
    PEMPolynomial p = pem_poly_alloc(3);
    p.coeff[0] = 1.0; p.coeff[1] = 0.5; p.coeff[2] = 0.25;
    double u[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double result = pem_poly_apply(&p, u, 2);
    /* p(q^{-1}) u(2) = 1.0*u[2] + 0.5*u[1] + 0.25*u[0] = 3.0 + 1.0 + 0.25 = 4.25 */
    CHECK_CLOSE(result, 4.25, TOL, "poly eval");
    pem_poly_free(&p);
    PASS();

    TEST("polynomial addition");
    PEMPolynomial a = pem_poly_alloc(2);
    a.coeff[0] = 1.0; a.coeff[1] = 2.0;
    PEMPolynomial b = pem_poly_alloc(3);
    b.coeff[0] = 3.0; b.coeff[1] = 4.0; b.coeff[2] = 5.0;
    PEMPolynomial c = pem_poly_add(&a, &b);
    CHECK_CLOSE(c.coeff[0], 4.0, TOL, "add[0]");
    CHECK_CLOSE(c.coeff[1], 6.0, TOL, "add[1]");
    CHECK_CLOSE(c.coeff[2], 5.0, TOL, "add[2]");
    pem_poly_free(&a); pem_poly_free(&b); pem_poly_free(&c);
    PASS();

    TEST("polynomial multiplication");
    PEMPolynomial x = pem_poly_alloc(2);
    x.coeff[0] = 1.0; x.coeff[1] = 1.0;  /* 1 + q^{-1} */
    PEMPolynomial y = pem_poly_alloc(2);
    y.coeff[0] = 1.0; y.coeff[1] = -1.0; /* 1 - q^{-1} */
    PEMPolynomial z = pem_poly_mul(&x, &y);
    /* (1+q^{-1})(1-q^{-1}) = 1 - q^{-2} */
    CHECK_CLOSE(z.coeff[0], 1.0, TOL, "mul[0]");
    CHECK_CLOSE(z.coeff[1], 0.0, TOL, "mul[1]");
    CHECK_CLOSE(z.coeff[2], -1.0, TOL, "mul[2]");
    pem_poly_free(&x); pem_poly_free(&y); pem_poly_free(&z);
    PASS();
}

/* Test 2: Utilities */
static void test_utilities(void) {
    TEST("mean");
    double x[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    CHECK_CLOSE(pem_mean(x, 5), 3.0, TOL, "mean");
    PASS();

    TEST("variance");
    double var = pem_variance(x, 5, 3.0);
    CHECK_CLOSE(var, 2.5, TOL, "variance");
    PASS();

    TEST("norm2");
    double v[] = {3.0, 4.0};
    CHECK_CLOSE(pem_norm2(v, 2), 5.0, TOL, "norm2");
    PASS();

    TEST("dot product");
    double a[] = {1.0, 2.0, 3.0};
    double b[] = {4.0, 5.0, 6.0};
    CHECK_CLOSE(pem_dot(a, b, 3), 32.0, TOL, "dot");
    PASS();

    TEST("NRMSE fit");
    double y[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double yh[] = {1.1, 1.9, 3.1, 3.9, 5.1};
    double fit = pem_nrmse_fit(y, yh, 5);
    CHECK(fit > 90.0, "NRMSE should be > 90%");
    PASS();
}

/* Test 3: ARX Predictor */
static void test_arx_predictor(void) {
    TEST("ARX one-step prediction");
    /* True system: y(t) = 0.5*y(t-1) + 1.0*u(t-1) + e(t)
     * na=1, nb=1, nk=1, theta=[0.5, 1.0] */
    double theta[] = {0.5, 1.0};
    double u[] = {1.0, 0.0, 1.0, 0.0, 1.0};
    double y[] = {0.0, 1.5, 0.75, 1.375, 0.6875};
    /* y(0)=0, y(1)=0.5*0+1.0*1.0=1.0+0.5? Let me recalculate:
     * y(1)=0.5*0+1.0*1.0=1.0, y(2)=0.5*1.0+1.0*0=0.5 */
    /* Actually let's compute step by step:
     * y(0)=0
     * y(1)=0.5*0+1.0*u[0]=1.0 → OK prediction: phi^T=[-0, u[0]]=[0,1], theta^T*phi=1.0 ✓
     * y(2)=0.5*1.0+1.0*u[1]=0.5+0=0.5 → prediction: phi^T=[-1.0, u[1]]=[-1.0,0], theta^T*phi=-0.5+0=-0.5? No!
     * Wait: y_hat(2) = -a_1*y(1) + b_1*u(1) = -0.5*1.0 + 1.0*0.0 = -0.5
     * So y(2) would be -0.5 if no noise. But y is set to 0.5 above...
     * Let me just use computed values. */
    double y_hat = pem_predict_arx(theta, 1, 1, 1, u, y, 2);
    /* y_hat(2) = -0.5*y[1] + 1.0*u[1] = -0.5*1.5 + 1.0*0.0 = -0.75 */
    CHECK_CLOSE(y_hat, -0.75, TOL_SOFT, "ARX predict t=2");
    PASS();

    TEST("ARX residual");
    double eps = pem_residual_arx(theta, 1, 1, 1, u, y, 2);
    /* eps(2) = y[2] - y_hat(2) = 0.75 - (-0.75) = 1.5 */
    CHECK_CLOSE(eps, 1.5, TOL_SOFT, "ARX residual t=2");
    PASS();

    TEST("ARX batch prediction");
    double y_hat_batch[5];
    pem_predict_arx_batch(theta, 1, 1, 1, u, y, 5, y_hat_batch);
    assert(y_hat_batch[0] == 0.0); /* t=0, no past data */
    PASS();
}

/* Test 4: ARX Criterion and Gradient */
static void test_arx_criterion(void) {
    TEST("ARX criterion");
    double theta[] = {0.5, 1.0};
    PEMData *data = pem_data_alloc(5);
    data->u[0]=1.0; data->u[1]=0.0; data->u[2]=1.0; data->u[3]=0.0; data->u[4]=1.0;
    data->y[0]=0.0; data->y[1]=0.5; data->y[2]=0.25; data->y[3]=0.125; data->y[4]=0.0625;
    double V = pem_criterion_arx(theta, 1, 1, 1, data);
    CHECK(V >= 0.0, "Criterion should be non-negative");
    pem_data_free(data);
    PASS();

    TEST("ARX gradient (at optimum should be near zero for LS)");
    /* ARX convention: y(t)=-a_1*y(t-1)+b_1*u(t-1)
     * For y(t)=0.7*y(t-1)+0.3*u(t-1): a_1=-0.7, b_1=0.3 */
    PEMData *data2 = pem_data_alloc(100);
    double true_theta[] = {-0.7, 0.3};
    /* Simulate: y(t) = -(-0.7)*y(t-1) + 0.3*u(t-1) = 0.7*y(t-1)+0.3*u(t-1) */
    for (int t = 0; t < 100; t++) {
        data2->u[t] = (t % 20 < 10) ? 1.0 : -1.0;
        if (t == 0) data2->y[t] = 0.0;
        else data2->y[t] = 0.7*data2->y[t-1] + 0.3*data2->u[t-1];
    }
    double g[2];
    pem_gradient_arx(true_theta, 1, 1, 1, data2, g);
    CHECK(fabs(g[0]) < 1.0, "Gradient[0] should be small at true params");
    CHECK(fabs(g[1]) < 1.0, "Gradient[1] should be small at true params");
    pem_data_free(data2);
    PASS();
}

/* Test 5: OE Predictor */
static void test_oe_predictor(void) {
    TEST("OE one-step prediction (batch)");
    /* OE model: y(t) = B(q)/F(q) u(t-nk), nk=1
     * F(q)=1+f_1 q^{-1}=1-0.5q^{-1}, B(q)=b_1=1.0
     * w(t) = b_1*u(t-1) - f_1*w(t-1) = 1.0*u(t-1) + 0.5*w(t-1) */
    double theta[] = {1.0, -0.5}; /* b_1=1.0, f_1=-0.5 */
    double u[] = {1.0, 1.0, 1.0, 1.0, 1.0};
    /* Use batch predictor which maintains state internally */
    double y_hat[5];
    pem_predict_oe_batch(theta, 1, 1, 1, u, 5, y_hat);
    /* w(0)=0, w(1)=b1*u[0]=1.0, w(2)=b1*u[1]-f1*w[1]=1.0+0.5*1=1.5,
     * w(3)=1.0+0.5*1.5=1.75, w(4)=1.0+0.5*1.75=1.875 */
    CHECK_CLOSE(y_hat[3], 1.75, TOL_SOFT, "OE predict t=3");
    CHECK_CLOSE(y_hat[4], 1.875, TOL_SOFT, "OE predict t=4");
    PASS();
}

/* Test 6: Cholesky and Linear Solver */
static void test_cholesky(void) {
    TEST("Cholesky decomposition");
    /* A = [4, 2; 2, 3], positive definite */
    double A[] = {4.0, 2.0, 2.0, 3.0};
    double L[4];
    memcpy(L, A, 4 * sizeof(double));
    int ret = pem_cholesky(L, 2);
    CHECK(ret == 0, "Cholesky should succeed");
    /* L = [2, 0; 1, sqrt(2)] */
    CHECK_CLOSE(L[0], 2.0, TOL, "L[0][0]");
    CHECK_CLOSE(L[2], 1.0, TOL, "L[1][0]");
    PASS();

    TEST("Cholesky solve");
    double b[] = {6.0, 7.0};
    double x[2];
    pem_cholesky_solve(L, b, x, 2);
    /* A*x = b: [4,2;2,3]*[x0;x1] = [6;7]
     * 4x0+2x1=6, 2x0+3x1=7 -> x0=0.5, x1=2 */
    CHECK_CLOSE(x[0], 0.5, TOL, "x[0]");
    CHECK_CLOSE(x[1], 2.0, TOL, "x[1]");
    PASS();
}

/* Test 7: ARX Least-Squares Estimation */
static void test_arx_estimation(void) {
    TEST("ARX LS estimation");
    /* Generate data from ARX(1,1,1): y(t)=0.7*y(t-1)+0.3*u(t-1)+e(t) */
    int N = 200;
    PEMData *data = pem_data_alloc(N);
    data->y[0] = 0.0;
    for (int t = 0; t < N; t++) {
        data->u[t] = (t % 20 < 10) ? 1.0 : -1.0;
        if (t > 0)
            data->y[t] = 0.7 * data->y[t-1] + 0.3 * data->u[t-1];
    }

    PEMResult *result = pem_result_alloc(2);
    PEMOptions opts = pem_options_default();
    int ret = pem_estimate_arx_ls(data, 1, 1, 1, result, &opts);

    CHECK(ret == 0, "ARX LS should succeed");
    /* ARX convention: A(q)=1+a_1 q^{-1}, so y(t)=-a_1 y(t-1)+b_1 u(t-1)
     * For y(t)=0.7*y(t-1)+0.3*u(t-1): a_1=-0.7, b_1=0.3 */
    CHECK_CLOSE(result->theta_hat[0], -0.7, 0.1, "a_1 estimate");
    CHECK_CLOSE(result->theta_hat[1], 0.3, 0.1, "b_1 estimate");
    CHECK(result->loss_final < 1e-4, "Loss should be near zero (noise-free)");

    pem_result_free(result);
    pem_data_free(data);
    PASS();
}

/* Test 8: OE Estimation */
static void test_oe_estimation(void) {
    TEST("OE estimation (noise-free)");
    /* Generate data from OE: y(t) = B/F u(t-1), B=1.0, F=1-0.5q^{-1}
     * w(t) = 1.0*u(t-1) + 0.5*w(t-1) */
    int N = 200;
    PEMData *data = pem_data_alloc(N);
    double w = 0.0;
    for (int t = 0; t < N; t++) {
        data->u[t] = (t % 20 < 10) ? 1.0 : -1.0;
        double u_prev = (t > 0) ? data->u[t-1] : 0.0;
        w = u_prev + 0.5 * w;
        data->y[t] = w;
    }

    PEMResult *result = pem_result_alloc(2); /* nb=1, nf=1 */
    PEMOptions opts = pem_options_default();
    opts.max_iterations = 50;
    int ret = pem_estimate_oe(data, 1, 1, 1, NULL, result, &opts);

    CHECK(ret == 0, "OE estimation should succeed");
    CHECK_CLOSE(result->theta_hat[0], 1.0, 0.15, "b_1 estimate");
    /* f_1 should be near -0.5 (F = 1 - 0.5q^{-1} -> f_1 = -0.5) */
    CHECK_CLOSE(result->theta_hat[1], -0.5, 0.15, "f_1 estimate");

    pem_result_free(result);
    pem_data_free(data);
    PASS();
}

/* Test 9: Model Simulation */
static void test_simulation(void) {
    TEST("ARX simulation");
    /* ARX convention: y(t) = -a_1 y(t-1) + b_1 u(t-1)
     * For y(t)=0.7*y(t-1)+0.3*u(t-1): a_1=-0.7, b_1=0.3 */
    double theta[] = {-0.7, 0.3}; /* ARX(1,1,1) */
    double u[] = {1.0, 0.0, 1.0, 0.0, 1.0};
    double y_sim[5] = {0};
    int orders[] = {1, 1, 0, 0, 0};
    pem_simulate_model(PEM_ARX, theta, orders, 1, u, 5, y_sim, NULL);
    /* y_sim(0)=-(-0.7)*0+0.3*0=0; y_sim(1)=0.7*0+0.3*1=0.3; y_sim(2)=0.7*0.3+0.3*0=0.21 */
    CHECK_CLOSE(y_sim[1], 0.3, TOL_SOFT, "sim[1]");
    CHECK_CLOSE(y_sim[2], 0.21, TOL_SOFT, "sim[2]");
    PASS();

    TEST("OE simulation");
    double theta_oe[] = {1.0, -0.5}; /* B=1, F=1-0.5q^{-1} */
    int orders_oe[] = {0, 1, 0, 0, 1};
    double y_sim_oe[5] = {0};
    pem_simulate_model(PEM_OE, theta_oe, orders_oe, 1, u, 5, y_sim_oe, NULL);
    /* w(0)=0; w(1)=u[0]+0.5*w(0)=1; w(2)=u[1]+0.5*1.0=0.5; w(3)=1.25 */
    CHECK_CLOSE(y_sim_oe[1], 1.0, TOL_SOFT, "OE sim[1]");
    CHECK_CLOSE(y_sim_oe[2], 0.5, TOL_SOFT, "OE sim[2]");
    PASS();
}

/* Test 10: Validation Statistics */
static void test_validation(void) {
    TEST("NRMSE fit (perfect)");
    double y[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double yh[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double fit = pem_validation_fit(y, yh, 5);
    CHECK_CLOSE(fit, 100.0, TOL, "perfect fit");
    PASS();

    TEST("AIC computation");
    double aic = pem_validation_aic(0.01, 100, 5);
    /* AIC = log(0.01) + 2*5/100 = -4.605 + 0.1 = -4.505 */
    CHECK(aic < 0.0, "AIC should be negative for small loss");
    PASS();

    TEST("Ljung-Box whiteness test");
    double eps[] = {0.1, -0.2, 0.05, 0.15, -0.1, 0.08, -0.12, 0.03, 0.11, -0.07,
                    0.09, -0.06, 0.14, -0.08, 0.02, 0.10, -0.15, 0.04, -0.09, 0.13,
                    0.06, -0.11, 0.07, -0.13, 0.01, 0.12, -0.03, -0.05, 0.08, -0.14};
    double pval = pem_validation_ljung_box(eps, 30, 10);
    CHECK(pval >= 0.0 && pval <= 1.0, "p-value in [0,1]");
    PASS();
}

/* Test 11: PEMOptions defaults */
static void test_options(void) {
    TEST("default options");
    PEMOptions opts = pem_options_default();
    CHECK(opts.max_iterations == 100, "max_iter default");
    CHECK(opts.tol_param > 0.0, "tol_param positive");
    CHECK(opts.algorithm == PEM_OPT_LM, "default algorithm is LM");
    PASS();
}

/* Test 12: FIR Predictor */
static void test_fir(void) {
    TEST("FIR prediction");
    double theta[] = {0.5, 0.3, 0.2}; /* nb=3, nk=1 */
    double u[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double yh = pem_predict_fir(theta, 3, 1, u, 3);
    /* y_hat(3) = 0.5*u[2] + 0.3*u[1] + 0.2*u[0] = 0.5*3 + 0.3*2 + 0.2*1 = 1.5+0.6+0.2=2.3 */
    CHECK_CLOSE(yh, 2.3, TOL, "FIR predict");
    PASS();
}

int main(void) {
    printf("=== PEM Test Suite ===\n\n");

    test_polynomial();
    test_utilities();
    test_arx_predictor();
    test_arx_criterion();
    test_oe_predictor();
    test_cholesky();
    test_arx_estimation();
    test_oe_estimation();
    test_simulation();
    test_validation();
    test_options();
    test_fir();

    printf("\n=== Results: %d/%d tests passed ===\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}