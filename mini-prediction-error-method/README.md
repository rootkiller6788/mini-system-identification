# Mini Prediction Error Method (PEM)

**System Identification via Prediction Error Minimization**

## Module Status: COMPLETE ✅

- **L1 Definitions**: Complete (10 core definitions)
- **L2 Core Concepts**: Complete (8 concepts)
- **L3 Math Structures**: Complete (7 structures)
- **L4 Fundamental Laws**: Complete (6 theorems)
- **L5 Algorithms/Methods**: Complete (9 algorithms)
- **L6 Canonical Problems**: Complete (7 problems, 4 examples)
- **L7 Applications**: Partial+ (4 documented, examples provided)
- **L8 Advanced Topics**: Partial+ (5 advanced topics with implementations)
- **L9 Research Frontiers**: Partial (documented, not implemented)

**Total Score**: 15/18 (Complete 2×6 + Partial+ 1×2 + Partial 1×1 = 15)
**Code Lines**: 3442 (include/ + src/)

---

## Core Definitions (L1)

| Term | Definition |
|------|-----------|
| Prediction Error | eps(t,theta) = y(t) - y_hat(t|theta) |
| One-Step-Ahead Predictor | y_hat(t|theta) = H^{-1}(q)G(q)u(t) + (1-H^{-1}(q))y(t) |
| Model Structure | M(theta) = {G(q,theta), H(q,theta)} parameterized transfer functions |
| Criterion Function | V_N(theta) = (1/N) sum l(eps(t,theta)) |
| Pseudo-Regressor | psi(t,theta) = -d(eps)/d(theta) |

## Core Theorems (L4)

### Least Squares Consistency
Under persistence of excitation and white noise:
```
theta_LS -> theta_0 as N -> inf
```

### Akaike's Information Criterion (AIC, 1974)
```
AIC = log(V_N) + 2d/N
```
Balances model fit against complexity.

### Ljung-Box Q-Test (1978)
```
Q = N(N+2) sum_{k=1}^m r_k^2 / (N-k)
```
Tests residual whiteness: Q ~ chi-squared(m) under H0.

### Gauss-Markov Theorem
Under homoskedastic white errors, the LS estimator is BLUE (Best Linear Unbiased Estimator).

### Armijo Condition (1966)
```
f(x + alpha*p) <= f(x) + c * alpha * g^T * p
```
Ensures sufficient decrease in line search.

## Core Algorithms (L5)

| Algorithm | Description |
|-----------|-------------|
| **Gauss-Newton** | p_k = -H_GN^{-1} g_k, where H_GN = (1/N) sum psi psi^T |
| **Levenberg-Marquardt** | p_k = -(H + lambda I)^{-1} g_k, adaptive damping |
| **SGD** | theta_{k+1} = theta_k - alpha_k g_k, Robbins-Monro schedule |
| **Cholesky Decomposition** | A = L L^T, O(n^3/6) for SPD matrices |
| **Closed-Form LS** | theta = (Phi^T Phi)^{-1} Phi^T Y for ARX/FIR |
| **Pseudo-Regressor Filtering** | F(q) psi(t) = phi(t) for OE gradients |
| **Armijo Backtracking** | Geometric step reduction until sufficient decrease |

## Classic Problems (L6)

| Problem | Model | Solution |
|---------|-------|----------|
| ARX Estimation | A(q)y = B(q)u + e | Closed-form LS |
| ARMAX Estimation | A(q)y = B(q)u + C(q)e | Iterative PEM |
| OE Estimation | y = B(q)/F(q) u + e | LM Optimization |
| BJ Estimation | y = B/F u + C/D e | LM Optimization |
| FIR Estimation | y = B(q) u + e | Closed-form LS |
| Model Selection | Multiple candidates | AIC/AICc/BIC/FPE |
| k-Step Prediction | Validation horizon | Recursive simulation |

## Mathematical Foundation

The general linear model:
```
y(t) = G(q,theta) u(t) + H(q,theta) e(t)

G(q) = B(q) / [A(q) F(q)]
H(q) = C(q) / [A(q) D(q)]
```

One-step-ahead predictor:
```
y_hat(t|theta) = H^{-1}(q) G(q) u(t) + [1 - H^{-1}(q)] y(t)
```

Criterion minimization:
```
theta_hat_N = argmin_theta (1/N) sum_{t=1}^N eps(t,theta)^2
```

## Nine-School Course Alignment

| School | Course | Key Coverage |
|--------|--------|-------------|
| MIT | 6.435 System Identification | Full PEM theory + practice |
| Stanford | EE264 DSP + System ID | LS + recursive methods |
| Berkeley | ME237 System Identification | Nonlinear optimization |
| CMU | 18-771 Linear Systems | ARX + state-space |
| Princeton | MAE 546 System ID | Robust identification |
| Caltech | CDS 110/210 | Freq-domain + validation |
| Cambridge | 4F12 System ID | ARMAX + BJ models |
| Oxford | System Identification | Linear polynomial models |
| ETH | 227-0216 Control II | ID for control design |

## Build & Test

```bash
make all        # Build static library (libpem.a)
make test       # Run 25 assert-based tests
make examples   # Build all examples
make demo       # Run comprehensive demo
make bench      # Performance benchmark
make clean      # Remove build artifacts
```

## File Structure

```
mini-prediction-error-method/
├── Makefile
├── README.md                 # This file
├── include/                  # 6 header files
│   ├── pem_core.h           # Core types, utilities
│   ├── pem_predictor.h      # Predictor API
│   ├── pem_criterion.h      # Criterion + gradient
│   ├── pem_model.h          # Estimation API
│   ├── pem_optimizer.h      # Optimization API
│   └── pem_validation.h     # Validation API
├── src/                      # 6 source files (3442 lines total)
│   ├── pem_core.c           # Memory, polynomials, utilities
│   ├── pem_predictor.c      # ARX/ARMAX/OE/BJ/FIR predictors
│   ├── pem_criterion.c      # Criterion, gradient, Hessian
│   ├── pem_model.c          # Estimation + simulation
│   ├── pem_optimizer.c      # GN, LM, SGD, Cholesky, line search
│   └── pem_validation.c     # Fit, AIC/BIC, Ljung-Box, cross-corr
├── tests/
│   └── test_pem.c           # 25 tests, all passing
├── examples/
│   ├── example1_arx_estimation.c
│   ├── example2_oe_estimation.c
│   ├── example3_armax_estimation.c
│   └── example4_model_selection.c
├── demos/
│   └── demo_overview.c      # Comprehensive workflow demo
├── benches/
│   └── bench_pem.c          # Performance benchmark
└── docs/
    ├── knowledge-graph.md
    ├── coverage-report.md
    ├── gap-report.md
    ├── course-alignment.md
    └── course-tree.md
```

## References

1. Ljung, L. (1999) *System Identification: Theory for the User*, 2nd ed. Prentice Hall.
2. Soderstrom, T. & Stoica, P. (1989) *System Identification*. Prentice Hall.
3. Nocedal, J. & Wright, S. (2006) *Numerical Optimization*, 2nd ed. Springer.
4. Akaike, H. (1974) "A New Look at the Statistical Model Identification," *IEEE Trans. Automatic Control*, 19(6), 716-723.
5. Marquardt, D.W. (1963) "An Algorithm for Least-Squares Estimation of Nonlinear Parameters," *SIAM J. Appl. Math.*, 11(2), 431-441.
6. Ljung, G.M. & Box, G.E.P. (1978) "On a Measure of Lack of Fit in Time Series Models," *Biometrika*, 65(2), 297-303.