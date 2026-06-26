# Knowledge Graph -- mini-regularized-least-squares

## L1: Definitions (Complete)
- Regularized Least Squares (RLS) problem formulation
- Ridge regression / Tikhonov regularization (L2 penalty)
- LASSO (Least Absolute Shrinkage and Selection Operator, L1 penalty)
- Elastic Net (combined L1+L2)
- Nuclear norm regularization (trace norm for matrices)
- Group LASSO (group-level sparsity)
- Fused LASSO (total variation + L1)
- FIR model: y(t) = sum_j b_j * u(t-j) + e(t)
- ARX model: A(q)*y(t) = B(q)*u(t) + e(t)
- OE model: y(t) = B(q)/F(q)*u(t) + e(t)
- ARMAX model: A(q)*y(t) = B(q)*u(t) + C(q)*e(t)
- Box-Jenkins model
- State-Space innovation model
- Regularization parameter lambda
- Effective degrees of freedom df(lambda)
- Design matrix Phi, output vector y, parameter vector theta
- Identification dataset (input u, output y, time t)
- Estimation result with uncertainty (std errors, p-values)
- Model structures and orders

## L2: Core Concepts (Complete)
- Bias-variance tradeoff in regularized estimation
- Overfitting and underfitting
- Cross-validation principle (K-fold)
- Condition number and numerical stability
- Ill-conditioned problems and regularization benefit
- Feature selection via L1 penalty
- Sparse recovery guarantees (LASSO)
- Prediction error estimation
- Generalization error
- Model complexity control
- Information criteria (AIC, BIC, AICc)
- Stein's unbiased risk estimate (SURE)

## L3: Mathematical Structures (Complete)
- Dense matrix in column-major storage (BLAS convention)
- Dynamic vector with capacity management
- Cholesky decomposition: A = L * L^T
- LDL^T decomposition: A = L * D * L^T
- QR decomposition via Householder reflectors
- SVD (Singular Value Decomposition): A = U * S * V^T
- Symmetric eigenvalue decomposition (Jacobi method)
- Pseudo-inverse via SVD: A^+ = V * S^+ * U^T
- Gram matrix: G = Phi^T * Phi
- Matrix inversion of SPD matrices
- Condition number estimation (power iteration)
- BLAS-like Level 1/2/3 operations
- Normal equations: (Phi^T*Phi + lambda*I)*theta = Phi^T*y
- Toeplitz structure in FIR regression
- Kernel (Gram) matrix: K = Phi * Phi^T
- Soft-thresholding operator: S(x,lambda) = sign(x)*max(|x|-lambda,0)

## L4: Fundamental Laws / Theorems (Complete)
- Normal equations theorem: unique minimizer of strongly convex quadratic
- Gauss-Markov theorem (extended for ridge): bias-variance decomposition
- Ridge bias formula: Bias(theta_hat) = -lambda*(Phi^T*Phi+lambda*I)^(-1)*theta_true
- Ridge variance formula
- SVD shrinkage interpretation: theta = sum_i (s_i/(s_i^2+lambda))*(u_i^T*y)*v_i
- Effective df: df(lambda) = sum_i s_i^2/(s_i^2+lambda)
- Monotonicity of df(lambda) in lambda
- GCV theorem: rotation-invariant approximately unbiased risk estimate
- Representer theorem for kernel methods
- Karush-Kuhn-Tucker (KKT) optimality conditions for LASSO
- Strong convexity of elastic net objective

## L5: Algorithms / Methods (Complete)
- Ridge via Cholesky: solve (Phi^T*Phi+lambda*I)theta = Phi^T*y
- Ridge via SVD: use singular value shrinkage
- Ridge via QR: augmented system approach
- Ridge via Conjugate Gradient (iterative)
- LASSO via Coordinate Descent (Friedman-Hastie-Tibshirani)
- LASSO via ADMM (Boyd et al.)
- Elastic Net via Coordinate Descent
- Group LASSO via Block Coordinate Descent
- Fused LASSO via ADMM with splitting
- LSQR for sparse iterative least squares
- Solution path computation (warm-starting)
- K-fold Cross-Validation
- Generalized Cross-Validation (GCV)
- L-curve criterion (corner detection)
- AICc and BIC computation
- SURE optimization
- Empirical Bayes / Marginal Likelihood maximization
- Kernel matrix construction
- Kernel ridge regression (dual formulation)
- Kernel hyperparameter optimization (marginal likelihood gradient)

## L6: Canonical Problems (Complete)
- FIR system identification from I/O data
- ARX model estimation
- OE model estimation (pseudo-linear regression)
- ARMAX model estimation
- Ill-conditioned regression
- Sparse impulse response recovery
- Model validation (fit percentage, residual whiteness, cross-correlation)
- Solution path visualization

## L7: Applications (Complete)
1. **DC Motor Identification** (Ljung 1999 benchmark)
   - ARX model of DC motor with GCV lambda selection
   - Realistic measurement noise, PRBS excitation
2. **FOPDT Process Control** (Chemical process industry)
   - Kernel-based FIR identification of first-order plus dead time
   - TC kernel with hyperparameter optimization
   - Refinery distillation column temperature control analogy
3. **GPS Signal Denoising** (Autonomous vehicles / Tesla / SpaceX)
   - Fused LASSO for piecewise-constant signal recovery
   - Total variation denoising for trajectory smoothing
4. **Biomedical Glucose Dynamics** (FDA artificial pancreas)
   - Bergman minimal model identification
   - ARX model with cross-validation
   - CGM data simulation with meal inputs

## L8: Advanced Topics (Complete)
- Kernel-based regularization for system identification (Pillonetto et al.)
- Stable Spline kernel: K(i,j) = beta^(i+j+max(i,j))/2 - beta^(3*max(i,j))/6
- Tuned/Correlated (TC) kernel: K(i,j) = beta^max(i,j) * gamma^(|i-j|)
- Diagonal (DI) kernel: K(i,j) = beta^i * delta(i,j)
- DC (Diagonal+Correlated) kernel
- RBF and Matern kernels for nonlinear FIR
- Marginal likelihood for hyperparameter selection
- Representer theorem application to system ID
- Bayesian interpretation of regularization (Gaussian process regression)
- Empirical Bayes / Type-II maximum likelihood
- Gradient of marginal likelihood w.r.t. kernel hyperparameters
- Kernel log-determinant computation via Cholesky

## L9: Research Frontiers (Partial)
- Deep kernel learning: neural network feature extractors + kernel regression
- Sparse Bayesian learning (Relevance Vector Machine) for system ID
- Online/recursive kernel methods for adaptive identification
- Multi-output Gaussian process state-space models
- Nonparametric frequency response estimation with kernel methods
- Gaussian process temporal convolutions for nonlinear system ID
