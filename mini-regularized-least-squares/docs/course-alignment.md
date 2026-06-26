# Course Alignment -- mini-regularized-least-squares

## Nine-School Reference Curriculum Mapping

### MIT
- **6.241J Dynamic Systems & Control** (Ch. 3-4): State-space models, least squares estimation
  - Covered: State-space model structure, SVD-based estimation, regularization
- **16.323 Optimal Control** (Ch. 2): Parameter estimation for control
  - Covered: ARX/FIR identification, prediction error minimization
- **6.832 Underactuated Robotics**: System identification for control
  - Covered: Sparse identification (LASSO), nonlinear NARX

### Stanford
- **AA203 Optimal Control**: Least squares and regularization fundamentals
  - Covered: Ridge, LASSO, Elastic Net formulations
- **EE363 Convex Optimization** (Ch. 6-7): Regularized approximation
  - Covered: Convex objective functions, KKT conditions, ADMM solver
- **AA274 Multi-agent**: Distributed identification
  - Covered: Group LASSO (group-level regularization)

### Berkeley
- **EE221A Linear Systems** (Ch. 4-5): Least squares, SVD, pseudo-inverse
  - Covered: Full SVD, QR, Cholesky implementation
- **EE222 Nonlinear Systems**: Nonlinear identification
  - Covered: NARX, pseudo-linear regression (OE/ARMAX)

### CMU
- **18-771 Linear Systems**: Matrix computations for estimation
  - Covered: Column-major BLAS operations, numerical linear algebra
- **24-677 Nonlinear Control**: System identification for control
  - Covered: Model validation, residual analysis

### Princeton
- **MAE 546 Optimal Control & Estimation**: Estimation theory
  - Covered: Bias-variance tradeoff, Stein estimation (SURE), effective df
- **ELE 530 Estimation & Detection**: Statistical estimation
  - Covered: AIC/BIC/AICc criteria, cross-validation

### Caltech
- **CDS110 Introduction to Control**: System identification basics
  - Covered: FIR/ARX model structures, prediction error
- **CDS140 Nonlinear Dynamics**: Data-driven modeling
  - Covered: NARX, polynomial basis expansion

### Cambridge
- **4F3 Nonlinear & Predictive Control**: Regularization in identification
  - Covered: Ridge/LASSO, lambda selection, bias-variance
- **4F2 Robust Control**: Robustness to model uncertainty
  - Covered: Condition number analysis, regularization for robustness

### Oxford
- **B4 Predictive Control**: Data-driven predictive models
  - Covered: Model validation metrics, fit percentage, residual tests
- **C20 Adaptive Control**: Online identification
  - Covered: Recursive least squares foundation (prepared)

### ETH
- **227-0216 System Identification** (Ljung textbook): Complete system ID
  - Covered: All model structures (FIR/ARX/OE/ARMAX/BJ/SS/NARX)
  - Covered: Regularization methods (L1/L2/Elastic Net/Kernel)
  - Covered: Validation (cross-validation, information criteria, residual tests)
  - Covered: Applications (DC motor, process control)

## Reference Textbooks Mapping

| Textbook | Chapters Covered |
|----------|-----------------|
| Ljung (1999) System Identification | Ch. 3-4 (models), Ch. 7 (LS), Ch. 10 (PEM), Ch. 16-18 (regularization) |
| Golub & Van Loan (1996) Matrix Computations | Ch. 3-5 (factorizations), Ch. 8 (SVD) |
| Hastie et al. (2009) ESL | Ch. 3 (linear methods), Ch. 7 (model assessment) |
| Pillonetto et al. (2022) Regularized System ID | Ch. 1-8 (kernel methods, stable spline) |
