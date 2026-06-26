# Knowledge Graph: Nonlinear System Identification

## L1: Definitions

| # | Concept | C Implementation | Lean Formalization |
|---|---------|-----------------|-------------------|
| 1 | Signal (time series) | `Signal` struct in nlsid_core.h | `FSignal` in nlsid_lean.lean |
| 2 | Dataset (I/O pairs) | `NLSIDDataset` struct | `Dataset` in nlsid_lean.lean |
| 3 | Model type taxonomy | `NLSIDModelType` enum | `ModelType` inductive |
| 4 | NARX model | `NARXModel` struct | - |
| 5 | Hammerstein model | `HammersteinModel` struct | `Hammerstein` structure |
| 6 | Wiener model | `WienerModel` struct | `Wiener` structure |
| 7 | Volterra series | `VolterraModel` struct | - |
| 8 | Bilinear state-space | `BilinearModel` struct | - |
| 9 | Neural network model | `NeuralNetModel` struct | - |
| 10 | Basis function | `BasisFunction` struct | - |
| 11 | Cost function types | `CostFunctionType` enum | `CostType` inductive |
| 12 | Identification config | `NLSIDConfig` struct | - |
| 13 | Identification result | `NLSIDResult` struct | - |

## L2: Core Concepts

| # | Concept | Implementation |
|---|---------|---------------|
| 1 | Persistence of excitation | `nlsid_test_pe()` |
| 2 | Nonlinearity detection | `nlsid_detect_nonlinearity()` |
| 3 | Structural identifiability | `nlsid_select_narx_order()` |
| 4 | Model selection | AIC/BIC/MDL/FPE criteria |
| 5 | One-step-ahead prediction | `predict_one_step` in each model |
| 6 | Simulation (output error) | `simulate` in each model |
| 7 | Cross-validation | `nlsid_cross_validate()` |
| 8 | Parameter estimation | `nlsid_identify()` |

## L3: Mathematical Structures

| # | Structure | Implementation |
|---|-----------|---------------|
| 1 | Polynomial basis expansion | `basis_eval_polynomial()`, `basis_expansion_polynomial()` |
| 2 | RBF network | `basis_eval_rbf()`, `basis_expansion_rbf_uniform()` |
| 3 | Sigmoid basis | `basis_eval_sigmoid()` |
| 4 | Fourier basis | `basis_eval_fourier()`, `basis_expansion_fourier_trunc()` |
| 5 | Wavelet basis | `basis_eval_wavelet_morlet()` |
| 6 | Piecewise linear (hinge) | `basis_eval_piecewise_linear()` |
| 7 | Regressor vector | `nlsid_build_regressor()` |
| 8 | Regressor matrix | `narx_compute_regressor_matrix()` |
| 9 | Jacobian matrix | `nlsid_compute_jacobian()`, `basis_expansion_get_jacobian()` |
| 10 | Hessian (Gauss-Newton approx) | `nlsid_compute_hessian_gn()` |
| 11 | Covariance matrix (RLS) | `nlsid_optimize_recursive_ls()` |
| 12 | Matrix inverse/solve/Cholesky | `nlsid_solve_linear_system()`, `nlsid_cholesky()` |

## L4: Fundamental Laws

| # | Theorem | Code Verification | Lean Theorem |
|---|---------|-------------------|--------------|
| 1 | Least squares optimality | `nlsid_init_narx_ls()` | `scalar_least_squares_optimal` |
| 2 | AIC asymptotic unbiasedness | `nlsid_compute_aic()` | `aic_penalty_vanishes` |
| 3 | BIC consistency | `nlsid_compute_bic()` | `bic_penalty_monotonic` |
| 4 | PE condition for identifiability | `nlsid_test_pe()` | `pe_requires_minimum_length` |
| 5 | Regularization reduces DOF | `nlsid_compute_cost_regularized()` | `regularization_reduces_df` |
| 6 | Gauss-Newton convergence (1 iter for LIP) | `nlsid_optimize_gauss_newton()` | `newton_quadratic_one_step` |
| 7 | Identifiability requires minimum data | `nlsid_select_narx_order()` | `identifiability_requires_min_data` |
| 8 | RLS covariance boundedness | `nlsid_optimize_recursive_ls()` | `rls_cov_non_increasing` |
| 9 | CG convergence rate bound | `nlsid_optimize_conjugate_gradient()` | `cg_error_decreases` |

## L5: Algorithms/Methods

| # | Algorithm | Implementation | Complexity |
|---|-----------|---------------|------------|
| 1 | Gauss-Newton | `nlsid_optimize_gauss_newton()` | O(N * d^2) per iter |
| 2 | Levenberg-Marquardt | `nlsid_optimize_levenberg_marquardt()` | O(N * d^2) per iter |
| 3 | Steepest descent + Armijo | `nlsid_optimize_steepest_descent()` | O(N * d) per iter |
| 4 | Conjugate gradient (FR) | `nlsid_optimize_conjugate_gradient()` | O(N * d) per iter |
| 5 | Orthogonal least squares | `nlsid_optimize_ols()` | O(N * d^2) |
| 6 | Recursive least squares | `nlsid_optimize_recursive_ls()` | O(d^2) per step |
| 7 | Armijo line search | `nlsid_line_search_armijo()` | O(N) per backtrack |
| 8 | Linear least squares init | `nlsid_init_narx_ls()` | O(N * d^2) |
| 9 | Gaussian elimination + pivot | `nlsid_solve_linear_system()` | O(n^3) |
| 10 | Cholesky decomposition | `nlsid_cholesky()` | O(n^3/3) |
| 11 | Matrix inversion | `nlsid_matrix_inverse()` | O(n^3) |

## L6: Canonical Problems

| # | Problem | Example File | Solution Method |
|---|---------|-------------|-----------------|
| 1 | Hammerstein system ID | example1_hammerstein_id.c | LM optimization |
| 2 | NARX model fitting | example2_narx_fit.c | LS + polynomial basis |
| 3 | Wiener system identification | example3_wiener_id.c | Wiener simulation |
| 4 | DC motor friction ID | example4_dc_motor_id.c | NARX + RBF basis |
| 5 | Volterra kernel estimation | src/nlsid_models.c:volterra_* | Direct LS |
| 6 | Bilinear SS identification | src/nlsid_models.c:bilinear_* | PEM |

## L7: Applications

| # | Application | Keywords | File |
|---|-------------|----------|------|
| 1 | DC motor with nonlinear friction | DC motor, Tesla | example4_dc_motor_id.c |
| 2 | Quadrotor dynamics identification | Quadrotor | nlsid_lean.lean (quadrotorMinData) |
| 3 | Chemical process ID (Hammerstein) | Chemical, process | example1_hammerstein_id.c |

## L8: Advanced Topics

| # | Topic | Implementation |
|---|-------|---------------|
| 1 | Bayesian nonlinear ID | `bayesFactorVerdict`, `Prior` structure in Lean |
| 2 | Neural network models | `NeuralNetModel`, `neuralnet_*` functions |
| 3 | Time-varying systems (forgetting factor) | `nlsid_optimize_recursive_ls()` with lambda |

## L9: Research Frontiers

| # | Topic | Documentation/Implementation |
|---|-------|----------------------------|
| 1 | SINDy (sparse identification) | `l0Norm`, `search_space_size_lower_bound` in Lean |
| 2 | Koopman operator theory | `koopmanMatrixSize`, `koopman_size_quadratic` in Lean |
