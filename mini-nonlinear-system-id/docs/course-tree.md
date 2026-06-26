# Course Tree: Nonlinear System Identification

## Prerequisites (Dependencies)

```
Linear Algebra
  └── Matrix operations (solve, inverse, Cholesky)
      └── Parameter estimation (LS, GN, LM)

Calculus / Real Analysis
  └── Numerical optimization (gradient, Hessian)
      └── Nonlinear LS algorithms

Probability & Statistics
  └── Maximum likelihood estimation
      └── PEM framework, AIC/BIC

Signal Processing
  └── Fourier analysis, sampling theory
      └── Frequency-domain ID, coherence

Linear System Theory
  ├── ARX models, transfer functions
  │   └── Linear baseline for nonlinear comparison
  └── State-space models
      └── Bilinear state-space, Kalman filter

Control Theory
  ├── Stability analysis
  │   └── Simulation-based stability test
  └── System excitation
      └── PE condition, input design
```

## Module Dependency Tree

```
mini-nonlinear-system-id
│
├── DEPENDS ON:
│   ├── mini-signal-processing (Signal, Dataset types)
│   ├── mini-regularized-least-squares (LS algorithms)
│   ├── mini-linear-system-theory (ARX baseline)
│   └── mini-nonlinear-system-theory (nonlinear concepts)
│
├── PROVIDES TO:
│   ├── mini-model-predictive-control (nonlinear prediction models)
│   ├── mini-adaptive-control (online ID, RLS)
│   ├── mini-learning-based-control (neural network models)
│   └── mini-fault-detection (residual-based diagnosis)
│
└── RELATED:
    ├── mini-frequency-domain-id (frequency-domain nonlinear ID)
    ├── mini-subspace-identification (subspace methods for bilinear)
    ├── mini-wiener-hammerstein (dedicated block-structure ID)
    └── mini-uncertainty-quantification (parameter confidence intervals)
```

## Knowledge Progression

```
L1 (Definitions)
  ↓
L2 (Core Concepts: PE, nonlinearity detection)
  ↓
L3 (Math Structures: basis expansions, regressor matrices)
  ↓
L4 (Fundamental Laws: LS optimality, AIC/BIC, identifiability)
  ↓
L5 (Algorithms: GN, LM, OLS, RLS, CG)
  ↓
L6 (Canonical Problems: Hammerstein, Wiener, NARX, DC motor)
  ↓
L7 (Applications: DC motor, Quadrotor, Chemical process)
  ↓
L8 (Advanced: Bayesian ID, Neural Net, Time-varying)
  ↓
L9 (Frontiers: SINDy, Koopman operator)
```

## Recommended Learning Path

1. **Foundations**: Read Ljung (1999) Ch. 1-4 for system ID basics
2. **Linear Models**: Master ARX/ARMAX identification (Ch. 4, 7)
3. **Nonlinear Models**: Study Nelles (2001) Ch. 4-6 for basis functions
4. **NARMAX Methods**: Billings (2013) Ch. 2-5
5. **Optimization**: Nocedal & Wright (2006) Ch. 10 for GN/LM
6. **Validation**: Billings & Zhu (1995) for nonlinear model tests
7. **Applications**: Implement motor/quadrotor/process ID from examples
8. **Advanced**: Bayesian methods, deep learning for system ID
