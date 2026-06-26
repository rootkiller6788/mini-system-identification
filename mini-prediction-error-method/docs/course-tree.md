# Course Prerequisite Tree: Prediction Error Method

## Prerequisites (What You Need Before PEM)

```
Linear Algebra
  ├── Matrix operations (matrix-vector multiply)
  ├── Cholesky decomposition
  ├── Linear systems (solve Ax=b)
  └── Eigenvalue/condition number concepts

Probability & Statistics
  ├── Random variables, expectation, variance
  ├── White noise processes
  ├── Hypothesis testing (Ljung-Box, cross-correlation)
  └── Maximum likelihood estimation

Signals & Systems
  ├── Discrete-time transfer functions
  ├── Difference equations
  ├── q-operator (shift operator)
  └── BIBO stability

Optimization
  ├── Gradient descent
  ├── Newton's method
  ├── Gauss-Newton for nonlinear least squares
  └── Line search methods

Control Theory (optional)
  ├── State-space representations
  ├── Observer/Kalman filter concepts
  └── Closed-loop vs open-loop identification
```

## PEM Module Internal Dependencies

```
pem_core.h/c (data structures, utilities)
  ├── pem_predictor.h/c (one-step-ahead predictors)
  │   └── pem_criterion.h/c (criterion, gradient, hessian)
  │       └── pem_optimizer.h/c (GN, LM, SGD)
  │           └── pem_model.h/c (high-level estimation)
  └── pem_validation.h/c (model validation statistics)
```

## Postrequisites (What PEM Enables)

```
PEM → Model-Based Control Design
  ├── MPC (Model Predictive Control)
  ├── LQG (Linear Quadratic Gaussian)
  └── Robust Control (H-infinity)

PEM → Adaptive Control
  ├── Self-Tuning Regulators
  └── Model Reference Adaptive Control

PEM → Fault Detection & Diagnosis
  ├── Residual generation
  └── Change detection

PEM → Signal Processing
  ├── Spectral estimation
  ├── Time series analysis
  └── Nonlinear filtering
```