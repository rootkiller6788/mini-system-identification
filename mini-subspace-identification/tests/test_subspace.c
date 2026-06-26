#include "subspace_core.h"
#include "subspace_algorithms.h"
#include "subspace_linalg.h"
#include "subspace_hankel.h"
#include "subspace_order.h"
#include "subspace_validation.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

/* ============================================================================
 * Subspace Identification -- Test Suite
 * Tests core API functions, linear algebra, Hankel construction,
 * projections, order estimation, and the three main algorithms.
 * ============================================================================ */

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(cond, msg) do { \
    if (cond) { tests_passed++; } \
    else { tests_failed++; printf("FAIL: %s\n", msg); } \
} while(0)

#define TEST_FLOAT_EQ(a, b, tol, msg) \
    TEST(fabs((a)-(b)) < (tol), msg)

/* Test 1-5: Memory management */
static void test_memory_management(void) {
    printf("--- Memory Management ---\n");

    SubspaceData *data = subspace_data_alloc(100, 2, 1);
    TEST(data != NULL, "data alloc");
    TEST(data->N == 100, "data N");
    TEST(data->n_inputs == 2, "data n_inputs");
    TEST(data->u != NULL && data->y != NULL, "data buffers");
    subspace_data_free(data);

    SubspaceMatrix *mat = subspace_matrix_alloc(5, 3);
    TEST(mat != NULL, "matrix alloc");
    TEST(mat->rows == 5 && mat->cols == 3, "matrix dims");
    subspace_matrix_free(mat);

    SubspaceHankel *H = subspace_hankel_alloc(10, 50, 2);
    TEST(H != NULL, "hankel alloc");
    TEST(H->total_rows == 20, "hankel rows");
    subspace_hankel_free(H);

    SubspaceModel *model = subspace_model_alloc(4, 2, 1);
    TEST(model != NULL, "model alloc");
    TEST(model->A != NULL && model->B != NULL, "model matrices");
    subspace_model_free(model);

    SubspaceResult *result = subspace_result_alloc();
    TEST(result != NULL, "result alloc");
    subspace_result_free(result);

    SubspaceSVD *svd = subspace_svd_alloc(10, 5);
    TEST(svd != NULL, "svd alloc");
    subspace_svd_free(svd);

    SubspaceOptions opts = subspace_options_default();
    TEST(opts.i == 10, "default options i");
    TEST(opts.algorithm == SS_N4SID, "default algorithm");

    printf("Memory tests: %d passed, %d failed\n\n",
           tests_passed, tests_failed);
}

/* Test 6-10: Matrix operations */
static void test_matrix_operations(void) {
    int prev_pass = tests_passed, prev_fail = tests_failed;
    printf("--- Matrix Operations ---\n");

    SubspaceMatrix *A = subspace_matrix_alloc(3, 2);
    SubspaceMatrix *B = subspace_matrix_alloc(2, 3);
    SubspaceMatrix *C = subspace_matrix_alloc(3, 3);
    TEST(A && B && C, "matrix alloc for ops");

    subspace_matrix_set(A, 0, 0, 1.0);
    subspace_matrix_set(A, 1, 1, 2.0);
    TEST_FLOAT_EQ(subspace_matrix_get(A, 0, 0), 1.0, 1e-10, "matrix set/get");
    TEST_FLOAT_EQ(subspace_matrix_get(A, 1, 1), 2.0, 1e-10, "matrix set/get 2");

    subspace_matrix_fill(B, 1.0);
    subspace_matrix_multiply(A, B, C);
    /* A=[1 0;0 2;0 0], B=all ones, C[0][0]=1, C[1][0]=2, C[2][0]=0 */
    TEST_FLOAT_EQ(subspace_matrix_get(C, 0, 0), 1.0, 1e-10, "matrix multiply C00");
    TEST_FLOAT_EQ(subspace_matrix_get(C, 1, 0), 2.0, 1e-10, "matrix multiply C10");
    TEST_FLOAT_EQ(subspace_matrix_get(C, 2, 0), 0.0, 1e-10, "matrix multiply C20");

    SubspaceMatrix *I = subspace_matrix_alloc(4, 4);
    subspace_matrix_identity(I);
    double trace = subspace_matrix_trace(I);
    TEST_FLOAT_EQ(trace, 4.0, 1e-10, "identity trace");

    double frob = subspace_matrix_norm_frobenius(I);
    TEST_FLOAT_EQ(frob, 2.0, 1e-10, "frobenius norm of I_4");

    subspace_matrix_free(A); subspace_matrix_free(B);
    subspace_matrix_free(C); subspace_matrix_free(I);
    printf("Matrix tests: %d passed, %d failed\n\n",
           tests_passed - prev_pass, tests_failed - prev_fail);
}

/* Test 11-15: Linear algebra */
static void test_linear_algebra(void) {
    int prev_pass = tests_passed, prev_fail = tests_failed;
    printf("--- Linear Algebra ---\n");

    /* QR decomposition */
    SubspaceMatrix *A = subspace_matrix_alloc(6, 3);
    SubspaceMatrix *Q = subspace_matrix_alloc(6, 3);
    SubspaceMatrix *R = subspace_matrix_alloc(3, 3);
    TEST(A && Q && R, "qr alloc");

    /* Fill A with known values */
    for (int i = 0; i < 6; i++)
        for (int j = 0; j < 3; j++)
            subspace_matrix_set(A, i, j, (double)(i + j + 1));

    int qr_ret = subspace_qr_mgs(A, Q, R);
    TEST(qr_ret == 0, "qr success");

    /* Check R is upper triangular and diagonal positive */
    double r00 = subspace_matrix_get(R, 0, 0);
    TEST(r00 > 0, "R[0][0] positive");

    /* Verify orthogonality: Q^T * Q should approximate I */
    double max_off_diag = 0.0;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            double dot = 0.0;
            for (int k = 0; k < 6; k++)
                dot += subspace_matrix_get(Q, k, i) *
                       subspace_matrix_get(Q, k, j);
            if (i == j) {
                TEST_FLOAT_EQ(dot, 1.0, 1e-6, "Q orthonormal diag");
            } else {
                if (fabs(dot) > max_off_diag) max_off_diag = fabs(dot);
            }
        }
    TEST(max_off_diag < 1e-6, "Q orthonormal off-diag");

    /* SVD test */
    SubspaceMatrix *B = subspace_matrix_alloc(4, 3);
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 3; j++)
            subspace_matrix_set(B, i, j, (double)(i + 1) * (double)(j + 0.5));
    SubspaceSVD *svd = subspace_svd_alloc(4, 3);
    TEST(svd != NULL, "svd alloc");
    int svd_ret = subspace_svd_compute(B, svd);
    TEST(svd_ret >= 0, "svd convergence");
    TEST(svd->S[0] > 0, "svd first singular value positive");
    TEST(svd->S[0] >= svd->S[1], "svd descending order");

    subspace_matrix_free(A); subspace_matrix_free(Q);
    subspace_matrix_free(R); subspace_matrix_free(B);
    subspace_svd_free(svd);
    printf("Linalg tests: %d passed, %d failed\n\n",
           tests_passed - prev_pass, tests_failed - prev_fail);
}

/* Test 16-20: Hankel construction */
static void test_hankel(void) {
    int prev_pass = tests_passed, prev_fail = tests_failed;
    printf("--- Hankel Construction ---\n");

    /* Create simple IO data */
    int N = 100, r = 1, m = 1;
    SubspaceData *data = subspace_data_alloc(N, r, m);
    TEST(data != NULL, "data alloc for hankel");
    for (int k = 0; k < N; k++) {
        data->u[k] = sin(0.1 * (double)k);
        data->y[k] = 0.5 * sin(0.1 * (double)k) + 0.1 * ((double)rand() / RAND_MAX - 0.5);
    }

    int i = 5;
    int block_cols = N - 2*i + 1;
    SubspaceHankel *Up = subspace_hankel_alloc(i, block_cols, r);
    SubspaceHankel *Uf = subspace_hankel_alloc(i, block_cols, r);
    SubspaceHankel *Yp = subspace_hankel_alloc(i, block_cols, m);
    SubspaceHankel *Yf = subspace_hankel_alloc(i, block_cols, m);
    TEST(Up && Uf && Yp && Yf, "hankel alloc for io");

    subspace_hankel_from_io_data(data, i, Up, Uf, Yp, Yf);

    /* Check dimensions */
    TEST(Up->total_rows == i * r, "Up rows");
    TEST(Up->total_cols == block_cols, "Up cols");

    /* Check first element: U_p(0,0) = u(0) */
    TEST_FLOAT_EQ(Up->data[0], data->u[0], 1e-10, "Up first element");

    /* Check U_f(0,0) = u(i) */
    TEST_FLOAT_EQ(Uf->data[0], data->u[i], 1e-10, "Uf first element");

    subspace_hankel_free(Up); subspace_hankel_free(Uf);
    subspace_hankel_free(Yp); subspace_hankel_free(Yf);
    subspace_data_free(data);
    printf("Hankel tests: %d passed, %d failed\n\n",
           tests_passed - prev_pass, tests_failed - prev_fail);
}

/* Test 21-25: Order estimation */
static void test_order_estimation(void) {
    int prev_pass = tests_passed, prev_fail = tests_failed;
    printf("--- Order Estimation ---\n");

    /* Simulate singular values: clear gap at order 3 */
    double sv[8] = {100.0, 50.0, 25.0, 0.1, 0.05, 0.02, 0.01, 0.005};
    int n_sv = 8;

    int order_gap = subspace_order_svd_gap(sv, n_sv, 8);
    TEST(order_gap == 3, "svd gap detects order 3");

    int order_svc = subspace_order_svc(sv, n_sv, 0.99, 8);
    TEST(order_svc >= 1 && order_svc <= 8, "svc returns valid order");

    int order_aic = subspace_order_aic(sv, n_sv, 500, 1, 1, 8);
    TEST(order_aic >= 1 && order_aic <= 8, "aic returns valid order");

    int order_nic = subspace_order_nic(sv, n_sv, 500, 1, 1, 8);
    TEST(order_nic >= 1 && order_nic <= 8, "nic returns valid order");

    int order_consensus = subspace_order_consensus(sv, n_sv, 500, 1, 1, 8);
    TEST(order_consensus >= 1 && order_consensus <= 8, "consensus valid");

    printf("Order tests: %d passed, %d failed\n\n",
           tests_passed - prev_pass, tests_failed - prev_fail);
}

/* Test 26-30: Subspace identification algorithms (small data for speed) */
static void test_algorithms(void) {
    int prev_pass = tests_passed, prev_fail = tests_failed;
    printf("--- Identification Algorithms ---\n");

    /* Generate data from a known 2nd-order system. Use small N for fast SVD. */
    int N = 80, r = 1, m = 1;
    SubspaceData *data = subspace_data_alloc(N, r, m);
    TEST(data != NULL, "data alloc for algorithm test");

    /* True model: x(k+1) = [0.8 0.1; -0.1 0.7] x(k) + [0.5; 0.3] u(k)
     *              y(k) = [1.0 0.5] x(k) + 0.1 u(k) + e(k) */
    double A_true[4] = {0.8, 0.1, -0.1, 0.7};
    double B_true[2] = {0.5, 0.3};
    double C_true[2] = {1.0, 0.5};
    double D_true = 0.1;

    /* Generate random input */
    srand(12345);
    for (int k = 0; k < N; k++)
        data->u[k] = 2.0 * ((double)rand() / RAND_MAX - 0.5);

    /* Simulate */
    double x[2] = {0, 0};
    for (int k = 0; k < N; k++) {
        data->y[k] = C_true[0] * x[0] + C_true[1] * x[1] + D_true * data->u[k]
                     + 0.01 * ((double)rand() / RAND_MAX - 0.5);
        double x_new[2];
        x_new[0] = A_true[0] * x[0] + A_true[1] * x[1] + B_true[0] * data->u[k];
        x_new[1] = A_true[2] * x[0] + A_true[3] * x[1] + B_true[1] * data->u[k];
        x[0] = x_new[0]; x[1] = x_new[1];
    }

    SubspaceOptions opts = subspace_options_default();
    opts.i = 5;
    opts.max_order = 6;
    opts.algorithm = SS_N4SID;

    SubspaceResult *result = subspace_result_alloc();
    TEST(result != NULL, "result alloc for n4sid");

    int ret = subspace_n4sid(data, &opts, result);
    TEST(ret == 0, "n4sid returns success");
    TEST(result->model != NULL, "n4sid model exists");
    TEST(result->order_estimated > 0, "n4sid order positive");
    TEST(result->order_estimated <= 6, "n4sid order bounded");

    /* Also test MOESP */
    SubspaceResult *result2 = subspace_result_alloc();
    opts.algorithm = SS_MOESP;
    ret = subspace_moesp(data, &opts, result2);
    TEST(ret == 0, "moesp returns success");

    /* Test CVA */
    SubspaceResult *result3 = subspace_result_alloc();
    opts.algorithm = SS_CVA;
    ret = subspace_cva(data, &opts, result3);
    TEST(ret == 0, "cva returns success");

    subspace_result_free(result);
    subspace_result_free(result2);
    subspace_result_free(result3);
    subspace_data_free(data);
    printf("Algorithm tests: %d passed, %d failed\n\n",
           tests_passed - prev_pass, tests_failed - prev_fail);
}

/* Test 31-35: Validation */
static void test_validation(void) {
    int prev_pass = tests_passed, prev_fail = tests_failed;
    printf("--- Validation ---\n");

    SubspaceModel *model = subspace_model_alloc(2, 1, 1);
    TEST(model != NULL, "model for validation");
    model->A[0] = 0.8; model->A[1] = 0.1;
    model->A[2] = -0.1; model->A[3] = 0.7;
    model->B[0] = 0.5; model->B[1] = 0.3;
    model->C[0] = 1.0; model->C[1] = 0.5;
    model->D[0] = 0.1;

    double spec_rad = subspace_stability_check(model);
    TEST(spec_rad < 1.0, "stability check stable");

    double *impulse = (double*)calloc(50, sizeof(double));
    subspace_model_impulse_response(model, impulse, 50);
    TEST(impulse[0] == 0.1, "impulse response D term");
    free(impulse);

    /* NRMSE fit test */
    double y_true[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double y_pred[5] = {1.1, 1.9, 3.2, 3.8, 5.1};
    double fit = subspace_fit_percent(y_true, y_pred, 5);
    TEST(fit > 0 && fit <= 100, "fit percent valid range");
    TEST(fit > 80, "fit percent reasonable");

    subspace_model_free(model);
    printf("Validation tests: %d passed, %d failed\n\n",
           tests_passed - prev_pass, tests_failed - prev_fail);
}

/* Test 36-40: Utility functions */
static void test_utilities(void) {
    int prev_pass = tests_passed, prev_fail = tests_failed;
    printf("--- Utilities ---\n");

    double x[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double mu = subspace_mean(x, 5);
    TEST_FLOAT_EQ(mu, 3.0, 1e-10, "mean calculation");

    double var = subspace_variance(x, 5, mu);
    TEST_FLOAT_EQ(var, 2.5, 1e-10, "variance calculation");

    double dot = subspace_dot_product(x, x, 5);
    TEST_FLOAT_EQ(dot, 55.0, 1e-10, "dot product");

    double nrm = subspace_norm2(x, 5);
    TEST_FLOAT_EQ(nrm, sqrt(55.0), 1e-10, "norm2");

    printf("Utility tests: %d passed, %d failed\n\n",
           tests_passed - prev_pass, tests_failed - prev_fail);
}

int main(void) {
    printf("========== Subspace Identification Test Suite ==========\n\n");

    test_memory_management();
    test_matrix_operations();
    test_linear_algebra();
    test_hankel();
    test_order_estimation();
    test_algorithms();
    test_validation();
    test_utilities();

    printf("========== Results ==========\n");
    printf("Total: %d passed, %d failed\n", tests_passed, tests_failed);

    return tests_failed > 0 ? 1 : 0;
}
