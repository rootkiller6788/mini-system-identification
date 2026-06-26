#include "nlsid_core.h"
#include "nlsid_models.h"
#include "nlsid_algorithms.h"
#include "nlsid_validation.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static int tests_passed = 0;
static int tests_failed = 0;
#define TEST(cond, msg) do { \
    if (cond) { tests_passed++; } \
    else { tests_failed++; printf("FAIL: %s\n", msg); } \
} while(0)
#define TEST_EQ(a, b, tol, msg) TEST(fabs((a)-(b)) < (tol), msg)

/* Test 1: Signal creation and basic operations (L1) */
static void test_signal_operations(void) {
    Signal* sig = nlsid_signal_create(100, 0.01);
    TEST(sig != NULL, "Signal creation");
    TEST(sig->length == 100, "Signal length");
    TEST(fabs(sig->sample_time - 0.01) < 1e-9, "Signal sample time");

    nlsid_signal_set(sig, 0, 3.14);
    TEST_EQ(nlsid_signal_get(sig, 0), 3.14, 1e-9, "Signal set/get");

    nlsid_signal_fill(sig, 1.0);
    TEST_EQ(nlsid_signal_mean(sig), 1.0, 1e-9, "Signal mean after fill");
    TEST_EQ(nlsid_signal_variance(sig), 0.0, 1e-9, "Signal variance after fill");

    /* Add noise and check statistics */
    unsigned int seed = 42;
    nlsid_signal_add_noise(sig, 0.1, &seed);
    double noisy_mean = nlsid_signal_mean(sig);
    TEST(fabs(noisy_mean - 1.0) < 0.1, "Signal mean with small noise");

    nlsid_signal_free(sig);
}

/* Test 2: Dataset management (L1) */
static void test_dataset_management(void) {
    NLSIDDataset* ds = nlsid_dataset_create(2, 1, 500, 0.01);
    TEST(ds != NULL, "Dataset creation");
    TEST(ds->n_samples == 500, "Dataset samples");
    TEST(ds->input->n_channels == 2, "Dataset input channels");
    TEST(ds->output->n_channels == 1, "Dataset output channels");

    /* Fill with sine wave data */
    for (int t = 0; t < ds->n_samples; t++) {
        ds->input->channels[0]->data[t] = sin(2.0 * M_PI * 0.01 * t);
        ds->input->channels[1]->data[t] = cos(2.0 * M_PI * 0.01 * t);
        ds->output->channels[0]->data[t] = 0.7 * sin(2.0 * M_PI * 0.01 * t)
                                            + 0.3 * cos(2.0 * M_PI * 0.01 * t);
    }

    /* Split dataset */
    NLSIDDataset *est = NULL, *val = NULL;
    int rc = nlsid_dataset_split(ds, 0.7, &est, &val);
    TEST(rc == 0, "Dataset split");
    TEST(est != NULL, "Estimation dataset exists");
    TEST(val != NULL, "Validation dataset exists");
    TEST(est->n_samples + val->n_samples == ds->n_samples, "Split count");

    nlsid_dataset_free(ds);
    nlsid_dataset_free(est);
    nlsid_dataset_free(val);
}

/* Test 3: Basis functions (L3) */
static void test_basis_functions(void) {
    /* Polynomial basis */
    double p_params[3] = {2.0, 1.0, 0.5}; /* degree=2, coeffs=[1, 0.5] */
    double x[2] = {1.0, 2.0};
    double y_poly = basis_eval_polynomial(x, p_params, 3);
    /* Expected: (1*1 + 0.5*2)^2 = (2)^2 = 4 */
    TEST_EQ(y_poly, 4.0, 1e-6, "Polynomial basis evaluation");

    /* RBF basis */
    double rbf_p[3] = {0.5, 1.0, 2.0}; /* sigma=0.5, center=(1,2) */
    double y_rbf = basis_eval_rbf(x, rbf_p, 3);
    TEST_EQ(y_rbf, 1.0, 1e-6, "RBF at center evaluates to 1.0");

    double x_far[2] = {10.0, 10.0};
    double y_rbf_far = basis_eval_rbf(x_far, rbf_p, 3);
    TEST(y_rbf_far < 0.01, "RBF decays away from center");

    /* Sigmoid basis */
    double sig_p[3] = {1.0, 1.0, 0.0}; /* slope=1, direction=(1,0) */
    double y_sig = basis_eval_sigmoid(x, sig_p, 3);
    TEST(y_sig > 0.5, "Sigmoid at positive input");

    /* Basis expansion */
    BasisExpansion* be = basis_expansion_create(2, 3);
    TEST(be != NULL, "Basis expansion creation");
    basis_expansion_add_basis(be, BASIS_RBF, rbf_p, 3);
    basis_expansion_add_basis(be, BASIS_SIGMOID, sig_p, 3);
    basis_expansion_add_basis(be, BASIS_POLYNOMIAL, p_params, 3);
    TEST(be->n_bases == 3, "Basis expansion has 3 bases");
    TEST(basis_expansion_nparams(be) == 4, "Nparams = 1 offset + 3 weights");

    /* Set weights */
    be->offset = 0.5;
    be->weights[0] = 1.0;
    be->weights[1] = 2.0;
    be->weights[2] = 3.0;

    double y_exp = basis_expansion_eval(be, x);
    TEST(y_exp > 0.0, "Basis expansion evaluates positively");

    basis_expansion_free(be);
}

/* Test 4: NARX model (L3, L5) */
static void test_narx_model(void) {
    NARXModel* narx = narx_create(2, 2, 1, 1, 1);
    TEST(narx != NULL, "NARX creation");
    TEST(narx->regressor_dim == 4, "NARX regressor dim = ny+nu = 4");

    BasisExpansion* be = basis_expansion_polynomial(4, 2);
    narx_set_expansion(narx, be);
    TEST(narx->n_params > 0, "NARX has parameters after expansion set");

    /* Set some weights manually */
    narx->expansion->offset = 0.1;
    narx->expansion->weights[0] = 0.5;

    /* Predict one step */
    double y_hist[5] = {0.0, 0.1, 0.2, 0.3, 0.4};
    double u_hist[5] = {1.0, 1.0, 1.0, 1.0, 1.0};
    double pred = narx_predict_one_step(narx, y_hist, u_hist, 4);
    /* Should produce a finite value */
    TEST(isfinite(pred), "NARX one-step prediction is finite");

    narx_free(narx);
}

/* Test 5: Hammerstein model (L6) */
static void test_hammerstein_model(void) {
    HammersteinModel* hm = hammerstein_create(2, 2, 1);
    TEST(hm != NULL, "Hammerstein creation");
    TEST(hm->na == 2 && hm->nb == 2, "Hammerstein orders");

    /* Set linear part: y(t) = u(t-1) + 0.5*u(t-2) */
    double a[3] = {1.0, -0.5, 0.0}; /* A(q) = 1 - 0.5 q^{-1} */
    double b[3] = {1.0, 0.5, 0.0};  /* B(q) = 1 + 0.5 q^{-1} */
    hammerstein_set_linear_part(hm, a, b);

    /* Simulate */
    double input[100];
    for (int i = 0; i < 100; i++) input[i] = 1.0; /* Step input */
    double y_pred[100];
    int rc = hammerstein_simulate(hm, input, 100, NULL, y_pred);
    TEST(rc == 0, "Hammerstein simulation succeeds");
    TEST(isfinite(y_pred[50]), "Hammerstein output is finite");

    hammerstein_free(hm);
}

/* Test 6: Wiener model (L6) */
static void test_wiener_model(void) {
    WienerModel* wm = wiener_create(2, 2, 1);
    TEST(wm != NULL, "Wiener creation");

    double a[3] = {1.0, -0.3, 0.0};
    double b[3] = {0.5, 1.0, 0.0};
    wiener_set_linear_part(wm, a, b);

    double input[50];
    for (int i = 0; i < 50; i++) input[i] = sin(0.1 * (double)i);
    double y_pred[50];
    int rc = wiener_simulate(wm, input, 50, NULL, y_pred);
    TEST(rc == 0, "Wiener simulation succeeds");
    TEST(isfinite(y_pred[25]), "Wiener output is finite");

    wiener_free(wm);
}

/* Test 7: Performance metrics (L1) */
static void test_performance_metrics(void) {
    double y[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double y_hat[5] = {1.1, 2.2, 2.8, 4.1, 4.9};
    double mse = nlsid_compute_mse(y, y_hat, 5);
    TEST(mse > 0.0, "MSE is positive for non-perfect fit");
    TEST(mse < 1.0, "MSE is less than 1 (good fit)");

    double fit = nlsid_compute_fit(y, y_hat, 5);
    TEST(fit > 50.0, "Fit percentage > 50%% for good fit");
    TEST(fit < 100.0, "Fit percentage < 100%% (not perfect)");

    /* Perfect fit */
    double y_perf[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double yh_perf[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double fit_perfect = nlsid_compute_fit(y_perf, yh_perf, 5);
    TEST_EQ(fit_perfect, 100.0, 0.01, "Perfect fit gives 100%%");

    /* Information criteria */
    double aic = nlsid_compute_aic(mse, 100, 4);
    double bic = nlsid_compute_bic(mse, 100, 4);
    TEST(bic > aic, "BIC > AIC (heavier penalty for 100 samples with 4 params)");

    double fpe = nlsid_compute_fpe(mse, 100, 4);
    TEST(fpe > mse, "FPE > MSE (penalizes model complexity)");
}

/* Test 8: Matrix operations (L5) */
static void test_matrix_operations(void) {
    /* Solve 2x2 system */
    double A[4] = {4.0, 1.0, 1.0, 3.0};
    double b[2] = {1.0, 2.0};
    double x[2];
    int rc = nlsid_solve_linear_system(A, b, 2, x);
    TEST(rc == 0, "Linear system solve succeeds");
    TEST_EQ(x[0], 0.09090909, 1e-4, "x[0] = 1/11 ≈ 0.0909");
    TEST_EQ(x[1], 0.63636363, 1e-4, "x[1] = 7/11 ≈ 0.6364");

    /* Check solution: 4*x0 + x1 = 1, x0 + 3*x1 = 2 */
    TEST_EQ(4.0*x[0] + x[1], 1.0, 1e-6, "Residual check");
    TEST_EQ(x[0] + 3.0*x[1], 2.0, 1e-6, "Residual check");

    /* Matrix inverse */
    double A_inv[4];
    rc = nlsid_matrix_inverse(A, 2, A_inv);
    TEST(rc == 0, "Matrix inverse succeeds");
    /* Check: A * A_inv = I */
    TEST_EQ(A_inv[0]*4.0 + A_inv[1]*1.0, 1.0, 1e-6, "Inverse (0,0)");
    TEST_EQ(A_inv[2]*4.0 + A_inv[3]*1.0, 0.0, 1e-6, "Inverse (1,0)");

    /* Cholesky decomposition */
    double A_spd[4] = {4.0, 1.0, 1.0, 3.0}; /* SPD */
    double L[4];
    rc = nlsid_cholesky(A_spd, 2, L);
    TEST(rc == 0, "Cholesky succeeds");

    /* Check L*L^T = A */
    double LLT[4] = {L[0]*L[0] + L[1]*L[1], L[0]*L[2] + L[1]*L[3],
                     L[2]*L[0] + L[3]*L[1], L[2]*L[2] + L[3]*L[3]};
    TEST_EQ(LLT[0], 4.0, 1e-6, "Cholesky LLT[0]");
    TEST_EQ(LLT[3], 3.0, 1e-6, "Cholesky LLT[3]");
}

/* Test 9: Information criteria (L4) */
static void test_information_criteria(void) {
    double var = 0.01;
    int N = 200, d = 5;

    double aic = nlsid_aic(var, N, d);
    double bic = nlsid_bic(var, N, d);
    double aicc = nlsid_aicc(var, N, d);
    double fpe = nlsid_fpe(var, N, d);

    TEST(isfinite(aic), "AIC is finite");
    TEST(isfinite(bic), "BIC is finite");
    TEST(aicc > aic, "AICc > AIC");
    TEST(bic > aic, "BIC > AIC for N=200, d=5");
    TEST(fpe > 0.0, "FPE is positive");
}

/* Test 10: Persistence of excitation (L2) */
static void test_persistence_excitation(void) {
    Signal* u = nlsid_signal_create(200, 0.01);
    /* Create a PE signal: sum of sinusoids */
    for (int t = 0; t < u->length; t++) {
        u->data[t] = sin(2.0 * M_PI * 0.01 * t)
                   + 0.5 * sin(2.0 * M_PI * 0.03 * t)
                   + 0.3 * sin(2.0 * M_PI * 0.07 * t);
    }

    PersistenceExcitation* pe = nlsid_test_pe(u, 4, 100);
    TEST(pe != NULL, "PE test created");
    TEST(pe->condition_number > 0.0, "PE condition number positive");
    TEST(pe->minimum_eigenvalue > 0.0, "PE min eigenvalue positive");

    nlsid_pe_free(pe);
    nlsid_signal_free(u);
}

/* Test 11: Residual analysis (validation L4) */
static void test_residual_analysis(void) {
    /* White noise residuals → good model */
    double residuals[200];
    unsigned int seed = 123;
    for (int i = 0; i < 200; i++) {
        seed = seed * 1103515245 + 12345;
        residuals[i] = ((double)(seed & 0x7FFFFFFF) / 2147483648.0 - 1.0) * 0.1;
    }

    double mean, var, skew, kurt;
    nlsid_residual_statistics(residuals, 200, &mean, &var, &skew, &kurt);
    TEST(fabs(mean) < 0.1, "Residual mean near zero");
    TEST(var > 0.0, "Residual variance positive");

    double* acf = malloc(21 * sizeof(double));
    bool is_white;
    nlsid_autocorrelation_test(residuals, 200, 20, acf, NULL, &is_white, NULL);
    /* White noise should have ACF ≈ 0 for lags > 0 */
    TEST(is_white, "White noise residuals detected as white");

    double Q, pval = nlsid_ljung_box_test(residuals, 200, 20, &Q);
    TEST(pval > 0.01, "Ljung-Box p-value > 0.01 for white noise");

    free(acf);
}

/* Test 12: Initialization + OLS (L5) */
static void test_initialization_and_ols(void) {
    NARXModel* narx = narx_create(2, 2, 1, 1, 1);
    BasisExpansion* be = basis_expansion_polynomial(4, 2);
    narx_set_expansion(narx, be);

    /* Random initialization */
    unsigned int seed = 42;
    nlsid_init_random(narx->theta, narx->n_params, 0.01, &seed);
    for (int i = 0; i < narx->n_params; i++) {
        TEST(isfinite(narx->theta[i]), "Random init param is finite");
    }

    /* Linear least squares init */
    NLSIDDataset* ds = nlsid_dataset_create(1, 1, 300, 0.1);
    /* Generate data from known NARX model */
    for (int t = 0; t < ds->n_samples; t++) {
        ds->input->channels[0]->data[t] = sin(0.1 * (double)t);
        ds->output->channels[0]->data[t] = 0.8 * (t > 0 ? ds->output->channels[0]->data[t-1] : 0.0)
            + 0.2 * ds->input->channels[0]->data[t];
    }

    int rc = nlsid_init_narx_ls(narx, ds);
    TEST(rc == 0, "Linear LS initialization succeeds");

    nlsid_dataset_free(ds);
    narx_free(narx);
}

/* Test 13: Neural network forward pass (L8) */
static void test_neural_network(void) {
    int layers[3] = {4, 5, 1};
    ActivationType acts[3] = {ACTIVATION_LINEAR, ACTIVATION_TANH, ACTIVATION_LINEAR};
    NeuralNetModel* nn = neuralnet_create(3, layers, acts);
    TEST(nn != NULL, "Neural net creation");
    TEST(nn->n_layers == 3, "Neural net layers");
    TEST(nn->n_params_total > 0, "Neural net has parameters");

    nn->ny = 2; nn->nu = 2; nn->nk = 1;
    nn->regressor_dim = 4;

    double input[4] = {1.0, 0.5, -0.3, 0.8};
    double output;
    int rc = neuralnet_forward(nn, input, &output);
    TEST(rc == 0, "Neural net forward pass succeeds");
    TEST(isfinite(output), "Neural net output is finite");

    neuralnet_free(nn);
}

/* Test 14: Nonlinearity detection (L2) */
static void test_nonlinearity_detection(void) {
    NLSIDDataset* ds = nlsid_dataset_create(1, 1, 200, 0.1);

    /* Generate nonlinear data: y(t) = 0.8*y(t-1) + 0.2*u(t) + 0.1*u(t)^2 */
    for (int t = 0; t < ds->n_samples; t++) {
        ds->input->channels[0]->data[t] = sin(2.0 * M_PI * 0.05 * (double)t);
        double y_prev = (t > 0) ? ds->output->channels[0]->data[t-1] : 0.0;
        double u_t = ds->input->channels[0]->data[t];
        ds->output->channels[0]->data[t] = 0.8 * y_prev + 0.2 * u_t + 0.1 * u_t * u_t;
    }

    NonlinearityTest* nt = nlsid_detect_nonlinearity(ds);
    TEST(nt != NULL, "Nonlinearity test created");
    /* Should detect nonlinearity from the u^2 term */
    /* We just check that the test runs and produces valid output */

    nlsid_nonlinearity_free(nt);
    nlsid_dataset_free(ds);
}

/* Test 15: Activation functions (L3) */
static void test_activation_functions(void) {
    TEST_EQ(activation_eval(0.0, ACTIVATION_TANH), 0.0, 1e-9, "Tanh(0)=0");
    TEST(activation_eval(100.0, ACTIVATION_TANH) > 0.99, "Tanh large positive → 1");
    TEST(activation_eval(-100.0, ACTIVATION_TANH) < -0.99, "Tanh large negative → -1");

    TEST_EQ(activation_eval(0.0, ACTIVATION_SIGMOID), 0.5, 1e-9, "Sigmoid(0)=0.5");
    TEST(activation_eval(0.0, ACTIVATION_RELU) == 0.0, "ReLU(0)=0");
    TEST(activation_eval(1.0, ACTIVATION_RELU) == 1.0, "ReLU(1)=1");
    TEST(activation_eval(-1.0, ACTIVATION_RELU) == 0.0, "ReLU(-1)=0");
}

/* Test 16: Linear ARX fitting (validation L6) */
static void test_linear_arx_fitting(void) {
    NLSIDDataset* ds = nlsid_dataset_create(1, 1, 300, 0.1);
    /* Generate ARX(2,1,1) data */
    for (int t = 0; t < ds->n_samples; t++) {
        ds->input->channels[0]->data[t] = sin(0.1 * (double)t) + 0.5 * cos(0.2 * (double)t);
        double y = 0.0;
        if (t >= 1) y += 0.5 * ds->output->channels[0]->data[t-1];
        if (t >= 2) y += 0.3 * ds->output->channels[0]->data[t-2];
        if (t >= 1) y += 0.2 * ds->input->channels[0]->data[t-1];
        ds->output->channels[0]->data[t] = y;
    }

    double a[5], b[5], fit, mse;
    int rc = nlsid_fit_linear_arx(ds, 2, 2, 1, a, b, &fit, &mse);
    TEST(rc == 0, "Linear ARX fitting succeeds");
    TEST(fit > 30.0, "Linear ARX fit > 30%%");

    nlsid_dataset_free(ds);
}

int main(void) {
    printf("=== mini-nonlinear-system-id Tests ===\n\n");

    test_signal_operations();
    test_dataset_management();
    test_basis_functions();
    test_narx_model();
    test_hammerstein_model();
    test_wiener_model();
    test_performance_metrics();
    test_matrix_operations();
    test_information_criteria();
    test_persistence_excitation();
    test_residual_analysis();
    test_initialization_and_ols();
    test_neural_network();
    test_nonlinearity_detection();
    test_activation_functions();
    test_linear_arx_fitting();

    printf("\n=== Results: %d passed, %d failed ===\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}