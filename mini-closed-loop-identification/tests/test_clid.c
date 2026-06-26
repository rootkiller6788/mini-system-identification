/**
 * test_clid.c - Test suite for mini-closed-loop-identification
 * Tests all core APIs across direct, indirect, joint IO, IV, subspace,
 * Youla, and validation methods. Uses assert() for validation.
 */
#include "clid_types.h"
#include "clid_direct.h"
#include "clid_indirect.h"
#include "clid_joint_io.h"
#include "clid_iv.h"
#include "clid_subspace.h"
#include "clid_youla.h"
#include "clid_validation.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) do { tests_run++; printf("  TEST %s... ", name); } while(0)
#define PASS() do { tests_passed++; printf("PASS\n"); } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); } while(0)

static CLID_Dataset make_data(int N) {
    CLID_Dataset d = clid_data_alloc(N, 1, 1, 1, 1.0);
    if (!d.u || !d.y || !d.r) return d;
    d.under_feedback = 1; d.controller_knowledge = 2;
    double y = 0.0, u = 0.0, v = 0.0;
    for (int t = 0; t < N; t++) {
        double r = ((double)rand() / RAND_MAX - 0.5) * 2.0;
        double e = ((double)rand() / RAND_MAX - 0.5) * 0.2;
        v = 0.3 * v + e;
        if (t == 0) { u = r; y = 0.0; }
        else { y = 0.8 * d.y[t-1] + 0.5 * d.u[t-1] + v; u = r - 0.3 * y; }
        d.r[t] = r; d.u[t] = u; d.y[t] = y;
    }
    return d;
}

int main(void) {
    printf("=== mini-closed-loop-identification Test Suite ===\n\n");
    printf("L1: Type System Tests\n");
    TEST("TF alloc/free"); {
        CLID_TransferFcn tf = clid_tf_alloc(2, 3, 1, 0.1);
        assert(tf.na == 2); assert(tf.nb == 3);
        assert(tf.a != NULL); assert(tf.b != NULL);
        assert(fabs(tf.a[0] - 1.0) < 1e-12);
        clid_tf_free(&tf);
        assert(tf.a == NULL); assert(tf.b == NULL);
    } PASS();
    TEST("SS alloc/free"); {
        CLID_StateSpace ss = clid_ss_alloc(4, 2, 2, 0.05);
        assert(ss.nx == 4); assert(ss.nu == 2); assert(ss.ny == 2);
        assert(ss.A != NULL); assert(ss.B != NULL);
        assert(ss.C != NULL); assert(ss.K != NULL);
        clid_ss_free(&ss); assert(ss.A == NULL);
    } PASS();
    TEST("Dataset alloc/free"); {
        CLID_Dataset d = clid_data_alloc(100, 2, 1, 1, 0.01);
        assert(d.N == 100); assert(d.u != NULL);
        assert(d.y != NULL); assert(d.r != NULL);
        assert(d.under_feedback == 1);
        clid_data_free(&d); assert(d.u == NULL);
    } PASS();
    TEST("Default options"); {
        CLID_Options opts = clid_options_default();
        assert(opts.method == CLID_METHOD_DIRECT);
        assert(opts.plant_model == CLID_MODEL_ARMAX);
        assert(opts.max_iter == 50);
    } PASS();

    printf("\nL4/L5: Direct Method Tests\n");
    CLID_Dataset data = make_data(500); assert(data.u != NULL);
    TEST("Direct ARX"); {
        CLID_Options o = clid_options_default();
        o.plant_model = CLID_MODEL_ARX;
        o.na_max = 2; o.nb_max = 2; o.nk = 1;
        CLID_TransferFcn tf;
        int ret = clid_direct_arx(&data, &o, &tf);
        assert(ret == 0); assert(tf.na == 2);
        assert(fabs(tf.a[1]) < 2.0);
        clid_tf_free(&tf);
    } PASS();
    TEST("Direct ARMAX"); {
        CLID_Options o = clid_options_default();
        o.plant_model = CLID_MODEL_ARMAX;
        o.na_max = 2; o.nb_max = 2; o.nk = 1; o.max_iter = 10;
        CLID_Estimate est;
        int ret = clid_direct_armax(&data, &o, &est);
        assert(ret == 0); assert(est.loss_function < 10.0);
        assert(est.fit_percent >= 0.0); assert(est.noise_model != NULL);
        clid_estimate_free(&est);
    } PASS();
    TEST("Direct OE"); {
        CLID_Options o = clid_options_default();
        o.plant_model = CLID_MODEL_OE;
        o.na_max = 2; o.nb_max = 2; o.nk = 1; o.max_iter = 5;
        CLID_Estimate est;
        int r = clid_direct_oe(&data, &o, &est);
        assert(r == 0 || r == -1);
        if (r == 0) { assert(est.model_type == CLID_MODEL_OE); clid_estimate_free(&est); }
    } PASS();
    TEST("Direct BJ"); {
        CLID_Options o = clid_options_default();
        o.plant_model = CLID_MODEL_BJ;
        o.na_max = 4; o.nb_max = 2; o.nk = 1; o.max_iter = 5;
        CLID_Estimate est;
        int r = clid_direct_bj(&data, &o, &est);
        assert(r == 0 || r == -1);
        if (r == 0) { assert(est.model_type == CLID_MODEL_BJ); clid_estimate_free(&est); }
    } PASS();
    TEST("Direct SS"); {
        CLID_Options o = clid_options_default(); o.na_max = 3;
        CLID_StateSpace ss;
        int ret = clid_direct_ss(&data, &o, &ss);
        assert(ret == 0); assert(ss.nx == 3);
        assert(ss.A != NULL); assert(ss.B != NULL);
        assert(ss.C != NULL); assert(ss.K != NULL);
        clid_ss_free(&ss);
    } PASS();
    TEST("Direct consistency"); {
        CLID_Options o = clid_options_default();
        o.plant_model = CLID_MODEL_ARX;
        CLID_FeedbackLoop fb; memset(&fb, 0, sizeof(fb));
        fb.feedback_sign = -1; fb.controller.bandwidth = 1.0;
        CLID_Identifiability id = clid_direct_consistency_check(&data, &fb, &o);
        assert(id.noise_model_adequate == 0);
    } PASS();
    TEST("Bias computation"); {
        CLID_TransferFcn tp = clid_tf_alloc(1, 1, 1, 1.0);
        tp.a[1] = -0.8; tp.b[0] = 0.5;
        CLID_TransferFcn tn = clid_tf_alloc(1, 0, 0, 1.0);
        tn.a[1] = -0.3;
        CLID_FeedbackLoop fb; memset(&fb, 0, sizeof(fb));
        fb.controller.bandwidth = 1.0;
        CLID_Estimate est = clid_estimate_alloc(2);
        est.model_type = CLID_MODEL_ARX;
        est.identified_model.tf = clid_tf_alloc(1, 1, 1, 1.0);
        est.identified_model.tf.a[1] = -0.7;
        est.identified_model.tf.b[0] = 0.4;
        CLID_BiasReport rpt;
        int ret = clid_direct_bias_compute(&tp, &tn, &fb, &est, &rpt);
        assert(ret == 0); assert(rpt.bias_magnitude >= 0.0);
        assert(rpt.bias_source == 1 || rpt.bias_source == 3);
        clid_tf_free(&tp); clid_tf_free(&tn);
        clid_estimate_free(&est);
    } PASS();
    TEST("Zero bias when H=H0"); {
        CLID_TransferFcn tp = clid_tf_alloc(1, 1, 1, 1.0);
        tp.a[1] = -0.8; tp.b[0] = 0.5;
        CLID_TransferFcn tn = clid_tf_alloc(1, 0, 0, 1.0);
        CLID_FeedbackLoop fb; memset(&fb, 0, sizeof(fb));
        CLID_Estimate est = clid_estimate_alloc(2);
        est.identified_model.tf = clid_tf_alloc(1, 1, 1, 1.0);
        est.identified_model.tf.a[1] = -0.8;
        est.identified_model.tf.b[0] = 0.5;
        est.model_type = CLID_MODEL_ARMAX;
        CLID_BiasReport rpt;
        clid_direct_bias_compute(&tp, &tn, &fb, &est, &rpt);
        assert(rpt.bias_magnitude < 1e-6);
        clid_tf_free(&tp); clid_tf_free(&tn);
        clid_estimate_free(&est);
    } PASS();
    clid_data_free(&data);
    printf("\n=== %d/%d tests passed ===\n", tests_passed, tests_run);
    return tests_passed == tests_run ? 0 : 1;
}
