#ifndef NLSID_MODELS_H
#define NLSID_MODELS_H

#include "nlsid_core.h"
#include <stddef.h>

/* ============================================================================
 * mini-nonlinear-system-id: Model Structures
 *
 * Concrete implementations of nonlinear model structures used in
 * system identification. Each model type captures a different class
 * of nonlinear dynamical behavior.
 *
 * Theoretical foundation from:
 *   Billings (2013) NARMAX Methods
 *   Haber & Keviczky (1999) Nonlinear System Identification
 *   Juditsky et al. (1995) "Nonlinear black-box models..."
 * ============================================================================ */

/* ============================================================================
 * Part 1: Basis Function Implementations
 * ============================================================================ */

double basis_eval_polynomial(const double* x, const double* p, int np);
double basis_eval_sigmoid(const double* x, const double* p, int np);
double basis_eval_rbf(const double* x, const double* p, int np);
double basis_eval_wavelet_morlet(const double* x, const double* p, int np);
double basis_eval_fourier(const double* x, const double* p, int np);
double basis_eval_piecewise_linear(const double* x, const double* p, int np);

BasisFunction* basis_create(BasisType type, int dim, const double* params, int n_params);
void basis_free(BasisFunction* bf);
double basis_evaluate(const BasisFunction* bf, const double* x);

BasisExpansion* basis_expansion_create(int input_dim, int n_bases);
void basis_expansion_free(BasisExpansion* be);
void basis_expansion_add_basis(BasisExpansion* be, BasisType type,
                                const double* params, int n_params);
double basis_expansion_eval(const BasisExpansion* be, const double* x);
void basis_expansion_eval_vector(const BasisExpansion* be,
                                  const double** X, int n_samples, double* y);
void basis_expansion_get_jacobian(const BasisExpansion* be,
                                   const double* x, double* J);
int basis_expansion_nparams(const BasisExpansion* be);
void basis_expansion_pack_params(const BasisExpansion* be, double* theta);
void basis_expansion_unpack_params(BasisExpansion* be, const double* theta);
void basis_expansion_print(const BasisExpansion* be);

/* Polynomial basis: {1, x1, x2, ..., x1^2, x1x2, x2^2, ..., x1^3, ...} */
BasisExpansion* basis_expansion_polynomial(int input_dim, int max_degree);
BasisExpansion* basis_expansion_rbf_uniform(int input_dim, int n_centers,
                                              double range_lo, double range_hi,
                                              double sigma);
BasisExpansion* basis_expansion_fourier_trunc(int input_dim, int n_harmonics);

/* ============================================================================
 * Part 2: NARX Model
 * ============================================================================ */

NARXModel* narx_create(int ny, int nu, int nk, int n_inputs, int n_outputs);
void narx_free(NARXModel* narx);
int narx_simulate(const NARXModel* narx, const double* input, int n_steps,
                  const double* y0, double* y_pred);
double narx_predict_one_step(const NARXModel* narx,
                              const double* y_hist, const double* u_hist, int t);
void narx_set_expansion(NARXModel* narx, BasisExpansion* expansion);
int narx_nparams(const NARXModel* narx);
void narx_get_params(const NARXModel* narx, double* theta);
void narx_set_params(NARXModel* narx, const double* theta);
void narx_compute_regressor_matrix(const NARXModel* narx,
                                    const double* y, const double* u,
                                    int n_samples, double** Phi, int* n_reg);
void narx_print(const NARXModel* narx);

/* ============================================================================
 * Part 3: Hammerstein Model
 * ============================================================================ */

HammersteinModel* hammerstein_create(int na, int nb, int nk);
void hammerstein_free(HammersteinModel* hm);
void hammerstein_set_static_nl(HammersteinModel* hm, BasisExpansion* nl);
void hammerstein_set_linear_part(HammersteinModel* hm,
                                  const double* a, const double* b);
int hammerstein_simulate(const HammersteinModel* hm,
                          const double* input, int n_steps,
                          const double* v0, double* y_pred);
double hammerstein_predict_one_step(const HammersteinModel* hm,
                                     const double* u_hist,
                                     const double* y_hist, int t);
void hammerstein_compute_v(const HammersteinModel* hm,
                            const double* u, int n, double* v_out);
int hammerstein_nparams(const HammersteinModel* hm);
void hammerstein_get_params(const HammersteinModel* hm, double* theta);
void hammerstein_set_params(HammersteinModel* hm, const double* theta);
void hammerstein_print(const HammersteinModel* hm);

/* ============================================================================
 * Part 4: Wiener Model
 * ============================================================================ */

WienerModel* wiener_create(int na, int nb, int nk);
void wiener_free(WienerModel* wm);
void wiener_set_linear_part(WienerModel* wm, const double* a, const double* b);
void wiener_set_static_nl(WienerModel* wm, BasisExpansion* nl);
int wiener_simulate(const WienerModel* wm, const double* input, int n_steps,
                     const double* x0, double* y_pred);
double wiener_predict_one_step(const WienerModel* wm,
                                const double* u_hist, const double* x_hist,
                                int t);
void wiener_compute_x(const WienerModel* wm, const double* u, int n,
                       double* x_out);
int wiener_nparams(const WienerModel* wm);
void wiener_get_params(const WienerModel* wm, double* theta);
void wiener_set_params(WienerModel* wm, const double* theta);
void wiener_print(const WienerModel* wm);

/* ============================================================================
 * Part 5: Volterra Series Model
 * ============================================================================ */

VolterraModel* volterra_create(int order, int memory, int input_dim);
void volterra_free(VolterraModel* vm);
void volterra_set_kernel_1d(VolterraModel* vm, const double* h1);
void volterra_set_kernel_2d(VolterraModel* vm, const double* h2);
void volterra_set_kernel_3d(VolterraModel* vm, const double* h3);
double volterra_eval(const VolterraModel* vm, const double* u_hist);
int volterra_simulate(const VolterraModel* vm, const double* input, int n_steps,
                       double* output);
int volterra_nparams(const VolterraModel* vm);
void volterra_get_params(const VolterraModel* vm, double* theta);
void volterra_set_params(VolterraModel* vm, const double* theta);
void volterra_compute_regressor_matrix(const VolterraModel* vm,
                                        const double* u, int n_samples,
                                        double** Phi, int* n_reg);
void volterra_print(const VolterraModel* vm);

/* ============================================================================
 * Part 6: Bilinear State-Space Model
 * ============================================================================ */

BilinearModel* bilinear_create(int n_states, int n_inputs, int n_outputs);
void bilinear_free(BilinearModel* bm);
void bilinear_set_A(BilinearModel* bm, const double* A);
void bilinear_set_B(BilinearModel* bm, const double* B);
void bilinear_set_C(BilinearModel* bm, const double* C);
void bilinear_set_N(BilinearModel* bm, int k, const double* Nk);
void bilinear_set_D(BilinearModel* bm, const double* D);
int bilinear_simulate(const BilinearModel* bm, const double* input, int n_steps,
                       const double* x0, double* y);
int bilinear_nparams(const BilinearModel* bm);
void bilinear_get_params(const BilinearModel* bm, double* theta);
void bilinear_set_params(BilinearModel* bm, const double* theta);
void bilinear_print(const BilinearModel* bm);

/* ============================================================================
 * Part 7: Neural Network Model
 * ============================================================================ */

double activation_eval(double x, ActivationType act);
double activation_derivative(double x, ActivationType act);

NeuralNetModel* neuralnet_create(int n_layers, const int* layer_sizes,
                                  const ActivationType* activations);
void neuralnet_free(NeuralNetModel* nn);
int neuralnet_forward(const NeuralNetModel* nn, const double* input,
                       double* output);
void neuralnet_forward_batch(const NeuralNetModel* nn,
                              const double** inputs, int n_samples,
                              double** outputs);
void neuralnet_set_sysid_regressors(NeuralNetModel* nn, int ny, int nu, int nk);
int neuralnet_simulate(const NeuralNetModel* nn, const double* input,
                        int n_steps, const double* y0, double* y_pred);
double neuralnet_predict_one_step(const NeuralNetModel* nn,
                                   const double* y_hist,
                                   const double* u_hist, int t);
int neuralnet_nparams(const NeuralNetModel* nn);
void neuralnet_get_params(const NeuralNetModel* nn, double* theta);
void neuralnet_set_params(NeuralNetModel* nn, const double* theta);
int neuralnet_nparams_total(const NeuralNetModel* nn);
void neuralnet_print(const NeuralNetModel* nn);

/* ============================================================================
 * Part 8: Model Factory
 * ============================================================================ */

NLSIDModel* nlsid_model_create_narx(int ny, int nu, int nk,
                                     int n_inputs, int n_outputs,
                                     const double* theta, int n_params);
NLSIDModel* nlsid_model_create_hammerstein(int na, int nb, int nk);
NLSIDModel* nlsid_model_create_wiener(int na, int nb, int nk);
NLSIDModel* nlsid_model_create_volterra(int order, int memory, int input_dim);
NLSIDModel* nlsid_model_create_bilinear(int nx, int nu, int ny);
NLSIDModel* nlsid_model_create_neural_net(int n_layers, const int* sizes,
                                           const ActivationType* acts,
                                           int ny, int nu, int nk);
NLSIDModel* nlsid_model_create_from_data(NLSIDModelType type,
                                          const NLSIDDataset* ds);

#endif /* NLSID_MODELS_H */