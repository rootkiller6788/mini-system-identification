/**
 * test_wh_model.c ? Comprehensive Test Suite for Wiener-Hammerstein Module
 *
 * Tests cover all core APIs across model creation, linear blocks,
 * nonlinearities, simulation, identification, signal generation, and validation.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "wh_model.h"
#include "wh_linear.h"
#include "wh_nonlinear.h"
#include "wh_identification.h"
#include "wh_simulation.h"
#include "wh_signal.h"
#include "wh_validation.h"

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) do { \
    printf("  TEST: %s ... ", name); \
    fflush(stdout); \
} while(0)

#define PASS() do { \
    printf("PASS\n"); \
    tests_passed++; \
} while(0)

#define FAIL(msg) do { \
    printf("FAIL: %s\n", msg); \
    tests_failed++; \
} while(0)

#define CHECK(cond, msg) do { \
    if (!(cond)) { FAIL(msg); return; } \
} while(0)

#define CHECK_EQ(a, b, msg) do { \
    if ((a) != (b)) { FAIL(msg); return; } \
} while(0)

#define CHECK_DBL_EQ(a, b, tol, msg) do { \
    if (fabs((a) - (b)) > (tol)) { \
        printf("FAIL: %s (%.6f vs %.6f)\n", msg, a, b); \
        tests_failed++; return; \
    } \
} while(0)

/* ??? Test: Model creation ??????????????????????????????????????????????? */

static void test_model_create(void) {
    TEST("WH_Model create");
    WH_Model* m = wh_model_create();
    CHECK(m != NULL, "creation failed");
    CHECK_EQ(m->L1.type, WH_LIN_FIR, "L1 type");
    CHECK_EQ(m->N.type, WH_NL_POLYNOMIAL, "N type");
    CHECK_EQ(m->L2.type, WH_LIN_FIR, "L2 type");
    CHECK_EQ(m->is_identified, 0, "not identified");
    wh_model_free(m);
    PASS();
}

/* ??? Test: Model evaluation ????????????????????????????????????????????? */

static void test_model_evaluate_identity(void) {
    TEST("WH_Model evaluate identity (u?u)");
    WH_Model* m = wh_model_create();
    CHECK(m != NULL, "creation failed");

    /* Default model is identity: L1=N=L2=1 */
    double y = wh_model_evaluate(m, 3.5);
    CHECK_DBL_EQ(y, 3.5, 1e-10, "identity 3.5");

    y = wh_model_evaluate(m, -2.0);
    CHECK_DBL_EQ(y, -2.0, 1e-10, "identity -2.0");

    y = wh_model_evaluate(m, 0.0);
    CHECK_DBL_EQ(y, 0.0, 1e-10, "identity 0.0");

    wh_model_free(m);
    PASS();
}

/* ??? Test: Linear block FIR ????????????????????????????????????????????? */

static void test_linear_fir(void) {
    TEST("Linear block FIR [1.0, 0.5]");
    WH_LinearBlock blk;
    double b[] = {1.0, 0.5};
    int ret = wh_linear_init_fir(&blk, b, 2, 1.0);
    CHECK_EQ(ret, 0, "init_fir");

    /* First sample */
    double y0 = wh_linear_evaluate(&blk, 1.0);
    CHECK_DBL_EQ(y0, 1.0, 1e-10, "FIR y[0] with u[0]=1");

    /* Second sample: u[1]=2, and delayed u[0]=1 contributes 0.5*1.0 */
    double y1 = wh_linear_evaluate(&blk, 2.0);
    CHECK_DBL_EQ(y1, 2.0 * 1.0 + 0.5 * 1.0, 1e-10, "FIR y[1] with u[1]=2, u[0]=1");

    /* Third sample */
    double y2 = wh_linear_evaluate(&blk, 0.0);
    CHECK_DBL_EQ(y2, 0.5 * 2.0, 1e-10, "FIR y[2] with u[2]=0");

    PASS();
}

/* ??? Test: Linear block IIR ????????????????????????????????????????????? */

static void test_linear_iir(void) {
    TEST("Linear block IIR y[k]=u[k]+0.5*y[k-1]");
    WH_LinearBlock blk;
    double b[] = {1.0};
    double a[] = {1.0, -0.5}; /* a1 = -0.5 ? y[k] = u[k] + 0.5*y[k-1] */
    int ret = wh_linear_init_iir(&blk, b, 1, a, 2, 1.0);
    CHECK_EQ(ret, 0, "init_iir");

    double y0 = wh_linear_evaluate(&blk, 1.0);
    CHECK_DBL_EQ(y0, 1.0, 1e-10, "IIR y[0]=1");

    double y1 = wh_linear_evaluate(&blk, 0.0);
    CHECK_DBL_EQ(y1, 0.5, 1e-10, "IIR y[1]=0.5");

    double y2 = wh_linear_evaluate(&blk, 0.0);
    CHECK_DBL_EQ(y2, 0.25, 1e-10, "IIR y[2]=0.25");

    PASS();
}

/* ??? Test: Linear block DC gain ????????????????????????????????????????? */

static void test_linear_dc_gain(void) {
    TEST("Linear block DC gain");
    WH_LinearBlock blk;
    double b[] = {0.2, 0.3, 0.4};
    wh_linear_init_fir(&blk, b, 3, 1.0);
    double dc = wh_linear_get_dc_gain(&blk);
    CHECK_DBL_EQ(dc, 0.9, 1e-10, "FIR DC gain = 0.2+0.3+0.4");

    /* IIR: H(z) = [2.0 * z^-1] / [1 - 0.8 * z^-1], DC = 2/(1-0.8) = 10 */
    WH_LinearBlock blk2;
    double b2[] = {2.0};
    double a2[] = {1.0, -0.8};
    wh_linear_init_iir(&blk2, b2, 1, a2, 2, 1.0);
    dc = wh_linear_get_dc_gain(&blk2);
    CHECK_DBL_EQ(dc, 10.0, 1e-6, "IIR DC gain = 2/(1-0.8)");

    PASS();
}

/* ??? Test: Nonlinearity polynomial ?????????????????????????????????????? */

static void test_nl_polynomial(void) {
    TEST("Nonlinearity polynomial f(x)=1+2x+3x^2");
    WH_Nonlinearity nl;
    double coeffs[] = {1.0, 2.0, 3.0};
    int ret = wh_nl_init_polynomial(&nl, coeffs, 2);
    CHECK_EQ(ret, 0, "init_poly");
    CHECK_EQ(nl.n_params, 3, "n_params=3");

    CHECK_DBL_EQ(wh_nl_evaluate(&nl, 0.0), 1.0, 1e-10, "f(0)=1");
    CHECK_DBL_EQ(wh_nl_evaluate(&nl, 1.0), 6.0, 1e-10, "f(1)=6");
    CHECK_DBL_EQ(wh_nl_evaluate(&nl, 2.0), 17.0, 1e-10, "f(2)=17");
    CHECK_DBL_EQ(wh_nl_evaluate(&nl, -1.0), 2.0, 1e-10, "f(-1)=2");

    /* Derivative: f'(x) = 2 + 6x */
    CHECK_DBL_EQ(wh_nl_derivative(&nl, 0.0), 2.0, 1e-10, "f'(0)=2");
    CHECK_DBL_EQ(wh_nl_derivative(&nl, 1.0), 8.0, 1e-10, "f'(1)=8");

    PASS();
}

/* ??? Test: Nonlinearity saturation ?????????????????????????????????????? */

static void test_nl_saturation(void) {
    TEST("Nonlinearity saturation K=2 L=1");
    WH_Nonlinearity nl;
    wh_nl_init_saturation(&nl, 2.0, 1.0);

    CHECK_DBL_EQ(wh_nl_evaluate(&nl, 0.0), 0.0, 1e-10, "sat(0)=0");
    CHECK_DBL_EQ(wh_nl_evaluate(&nl, 0.5), 1.0, 1e-10, "sat(0.5)=1");
    CHECK_DBL_EQ(wh_nl_evaluate(&nl, 1.5), 2.0, 1e-10, "sat(2)=2 (K*L)");
    CHECK_DBL_EQ(wh_nl_evaluate(&nl, -1.5), -2.0, 1e-10, "sat(-2)=-2");
    CHECK_DBL_EQ(wh_nl_derivative(&nl, 3.0), 0.0, 1e-10, "sat'(3)=0");

    PASS();
}

/* ??? Test: Nonlinearity sigmoid ????????????????????????????????????????? */

static void test_nl_sigmoid(void) {
    TEST("Nonlinearity sigmoid a=1 b=1 c=0");
    WH_Nonlinearity nl;
    wh_nl_init_sigmoid(&nl, 1.0, 1.0, 0.0);

    CHECK_DBL_EQ(wh_nl_evaluate(&nl, 0.0), 0.5, 1e-6, "sig(0)=0.5");
    CHECK(wh_nl_evaluate(&nl, 100.0) > 0.99, "sig(100)?1");
    CHECK(wh_nl_evaluate(&nl, -100.0) < 0.01, "sig(-100)?0");

    PASS();
}

/* ??? Test: WH model with gains ?????????????????????????????????????????? */

static void test_wh_model_gain(void) {
    TEST("WH model L1=gain2, N=saturate, L2=gain3");
    WH_Model* m = wh_model_create();
    CHECK(m != NULL, "creation");

    /* L1: gain 2.0 */
    double b1[] = {2.0};
    wh_linear_init_fir(&m->L1, b1, 1, 1.0);

    /* N: saturation K=1, L=5 */
    wh_nl_init_saturation(&m->N, 1.0, 5.0);

    /* L2: gain 3.0 */
    double b2[] = {3.0};
    wh_linear_init_fir(&m->L2, b2, 1, 1.0);

    wh_model_reset(m);

    /* u=1 ? x=2 ? saturated at 2 (<5) ? w=2 ? y=6 */
    double y = wh_model_evaluate(m, 1.0);
    CHECK_DBL_EQ(y, 6.0, 1e-10, "small input: 1?2?1*2?3*2=6");

    /* u=3 ? x=6 ? saturated at 5 ? w=5 ? y=15 */
    y = wh_model_evaluate(m, 3.0);
    CHECK_DBL_EQ(y, 15.0, 1e-10, "large input: 3?6?5?15");

    wh_model_free(m);
    PASS();
}

/* ??? Test: Model simulation batch ??????????????????????????????????????? */

static void test_model_simulate(void) {
    TEST("WH model batch simulation");
    WH_Model* m = wh_model_create();
    double u[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double y[5];
    int ret = wh_model_simulate(m, u, y, 5);
    CHECK_EQ(ret, WH_STATUS_OK, "simulate return");
    /* Identity model */
    for (int i = 0; i < 5; i++) {
        CHECK_DBL_EQ(y[i], u[i], 1e-10, "identity batch");
    }
    wh_model_free(m);
    PASS();
}

/* ??? Test: Signal generation ???????????????????????????????????????????? */

static void test_signal_sine(void) {
    TEST("Signal sine gen f=1Hz fs=100Hz");
    double s[100];
    wh_signal_sine(s, 100, 100.0, 1.0, 1.0, 0.0, 0.0);
    CHECK_DBL_EQ(s[0], 0.0, 1e-10, "sine[0]=0");
    CHECK_DBL_EQ(s[25], 1.0, 1e-6, "sine[25]=1 (quarter period)");
    PASS();
}

static void test_signal_step(void) {
    TEST("Signal step");
    double s[10];
    wh_signal_step(s, 10, 5, 0.0, 1.0);
    CHECK_DBL_EQ(s[0], 0.0, 1e-10, "step before");
    CHECK_DBL_EQ(s[5], 1.0, 1e-10, "step at");
    CHECK_DBL_EQ(s[9], 1.0, 1e-10, "step after");
    PASS();
}

static void test_signal_prbs(void) {
    TEST("Signal PRBS 7-stage");
    double s[127];
    int ret = wh_signal_prbs(s, 127, 1.0, 7, 1);
    CHECK_EQ(ret, 0, "PRBS gen");
    /* Check values are ?1 */
    int ones = 0, minus_ones = 0;
    for (int i = 0; i < 127; i++) {
        if (fabs(s[i] - 1.0) < 1e-10) ones++;
        if (fabs(s[i] + 1.0) < 1e-10) minus_ones++;
    }
    CHECK(ones > 30, "PRBS has +1");
    CHECK(minus_ones > 30, "PRBS has -1");
    PASS();
}

static void test_signal_statistics(void) {
    TEST("Signal statistics");
    double x[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double mean = wh_signal_mean(x, 5);
    CHECK_DBL_EQ(mean, 3.0, 1e-10, "mean=3");
    double var = wh_signal_variance(x, 5);
    CHECK_DBL_EQ(var, 2.5, 1e-10, "var=2.5");
    double rms = wh_signal_rms(x, 5);
    CHECK_DBL_EQ(rms, sqrt(11.0), 1e-10, "rms=sqrt(11)");
    PASS();
}

/* ??? Test: Simulation metrics ??????????????????????????????????????????? */

static void test_sim_metrics(void) {
    TEST("Simulation FIT/MSE metrics");
    double y_ref[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double y_sim[] = {1.1, 1.9, 3.0, 4.1, 4.9};
    double mse = wh_sim_compute_mse(y_ref, y_sim, 5);
    CHECK(mse < 0.1, "MSE small");
    double fit = wh_sim_compute_fit(y_ref, y_sim, 5);
    CHECK(fit > 90.0, "FIT > 90%");
    double mae = wh_sim_compute_mae(y_ref, y_sim, 5);
    CHECK(mae < 0.2, "MAE small");
    PASS();
}

/* ??? Test: Model validation quick ??????????????????????????????????????? */

static void test_validate_quick(void) {
    TEST("Model validation basic checks");
    WH_Model* m = wh_model_create();
    CHECK(m != NULL, "create");

    /* Stability */
    CHECK_EQ(wh_validate_stability(m), 1, "identity stable");

    /* Delay */
    CHECK_EQ(wh_validate_delay(m, 5), 1, "delay ? 5");

    /* Monotonicity */
    CHECK_EQ(wh_validate_monotonic(m), 1, "linear N monotonic");

    wh_model_free(m);
    PASS();
}

/* ??? Test: Model copy ??????????????????????????????????????????????????? */

static void test_model_copy(void) {
    TEST("WH model deep copy");
    WH_Model* m = wh_model_create();
    double b[] = {2.0, 1.0};
    wh_linear_init_fir(&m->L1, b, 2, 1.0);
    m->N.params[1] = 3.0;

    WH_Model* m2 = wh_model_copy(m);
    CHECK(m2 != NULL, "copy created");
    CHECK_DBL_EQ(m2->L1.b[0], 2.0, 1e-10, "L1.b[0] copied");
    CHECK_DBL_EQ(m2->L1.b[1], 1.0, 1e-10, "L1.b[1] copied");
    CHECK_DBL_EQ(m2->N.params[1], 3.0, 1e-10, "N copied");

    wh_model_free(m);
    wh_model_free(m2);
    PASS();
}

/* ??? Test: Information criteria ????????????????????????????????????????? */

static void test_info_criteria(void) {
    TEST("AIC/BIC computation");
    double aic = wh_ident_compute_aic(0.01, 5, 500);
    double bic = wh_ident_compute_bic(0.01, 5, 500);
    CHECK(aic < bic, "AIC < BIC for same model (AIC penalty = 2k, BIC = k*lnN)");
    CHECK(aic > -1e6, "AIC finite");
    CHECK(bic > -1e6, "BIC finite");
    PASS();
}

/* ??? Test: Parameter counting ??????????????????????????????????????????? */

static void test_param_count(void) {
    TEST("Parameter counting");
    WH_Model* m = wh_model_create();
    /* Default: L1.nb=1, N.n_params=2, L2.nb=1, noise=1 */
    int np = wh_model_count_parameters(m);
    CHECK(np >= 3, "at least 3 params");
    CHECK(np <= 10, "reasonable count");

    /* Change L1 to 3-tap FIR */
    double b[] = {1.0, 0.5, 0.2};
    wh_linear_init_fir(&m->L1, b, 3, 1.0);
    np = wh_model_count_parameters(m);
    CHECK(np >= 5, "more params after L1 change");

    wh_model_free(m);
    PASS();
}

/* ??? Test: Model delay computation ?????????????????????????????????????? */

static void test_model_delay(void) {
    TEST("Model delay computation");
    WH_Model* m = wh_model_create();

    /* L1 with delay: b = [0, 0, 1] ? delay=2 */
    double b1[] = {0.0, 0.0, 1.0};
    wh_linear_init_fir(&m->L1, b1, 3, 1.0);

    /* L2 with delay: b = [0, 1] ? delay=1 */
    double b2[] = {0.0, 1.0};
    wh_linear_init_fir(&m->L2, b2, 2, 1.0);

    int total_delay = wh_model_get_delay(m);
    CHECK_EQ(total_delay, 3, "total delay = 2+1 = 3");

    wh_model_free(m);
    PASS();
}

/* ??? Test: Multisine generation ????????????????????????????????????????? */

static void test_signal_multisine(void) {
    TEST("Multisine signal generation");
    double s[1024];
    int n_harmonics = wh_signal_multisine(s, 1024, 1000.0, 10.0, 400.0, 10, 0.5, 42);
    CHECK(n_harmonics > 0, "multisine generated");

    double rms = wh_signal_rms(s, 1024);
    CHECK(rms > 0.0, "multisine has energy");
    CHECK(rms < 2.0, "multisine not too large");

    (void)wh_signal_peak(s, 1024); /* Peak used for crest factor internally */
    double cf = wh_signal_crest_factor(s, 1024);
    CHECK(cf >= 1.0, "crest factor ? 1");
    CHECK(cf < 10.0, "crest factor reasonable");

    PASS();
}

/* ??? Main ??????????????????????????????????????????????????????????????? */

int main(void) {
    printf("??? Wiener-Hammerstein Module Test Suite ?????????????????\n\n");

    /* Model tests */
    test_model_create();
    test_model_evaluate_identity();
    test_model_copy();
    test_wh_model_gain();
    test_model_simulate();
    test_model_delay();
    test_param_count();

    /* Linear block tests */
    test_linear_fir();
    test_linear_iir();
    test_linear_dc_gain();

    /* Nonlinearity tests */
    test_nl_polynomial();
    test_nl_saturation();
    test_nl_sigmoid();

    /* Signal tests */
    test_signal_sine();
    test_signal_step();
    test_signal_prbs();
    test_signal_statistics();
    test_signal_multisine();

    /* Simulation tests */
    test_sim_metrics();

    /* Validation tests */
    test_validate_quick();

    /* Info criteria tests */
    test_info_criteria();

    printf("\n????????????????????????????????????????????????????????????\n");
    printf("?  Results: %d passed, %d failed                          \n",
           tests_passed, tests_failed);
    printf("????????????????????????????????????????????????????????????\n");

    return tests_failed > 0 ? 1 : 0;
}
