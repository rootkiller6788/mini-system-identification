# mini-nonlinear-system-id

Nonlinear System Identification -- methods and algorithms for building mathematical models of nonlinear dynamical systems from input-output data.

## Module Status: COMPLETE

- L1-L6: Complete
- L7: Complete (3 applications: DC motor, Quadrotor, Chemical process)
- L8: Partial (Bayesian ID, neural network models)
- L9: Partial (SINDy, Koopman operator documented)

## Knowledge Coverage Summary

| Level | Name | Status | Key Items |
|-------|------|--------|-----------|
| L1 | Definitions | Complete | Signal, Dataset, Model types (NARX/Hammerstein/Wiener/Volterra/Bilinear/NeuralNet), cost functions, config |
| L2 | Core Concepts | Complete | Persistence of excitation, nonlinearity detection, structural identifiability, model selection |
| L3 | Math Structures | Complete | Basis expansions (polynomial/RBF/sigmoid/fourier/wavelet), regression matrices, Jacobians |
| L4 | Fundamental Laws | Complete | Least squares optimality, AIC/BIC consistency, PE condition, regularization, identifiability bounds |
| L5 | Algorithms/Methods | Complete | Gauss-Newton, Levenberg-Marquardt, steepest descent, conjugate gradient, OLS, RLS, line search |
| L6 | Canonical Problems | Complete | Hammerstein ID, Wiener ID, NARX fitting, Bilinear SS, DC motor friction, Volterra series |
| L7 | Applications | Complete | DC motor with nonlinear friction, Quadrotor dynamics, Tesla actuator, chemical process ID |
| L8 | Advanced Topics | Partial | Bayesian ID (Bayes factor formalized), neural network models (MLP), time-varying RLS |
| L9 | Research Frontiers | Partial | SINDy (sparse identification), Koopman operator theory (documented) |

## Core Definitions

- **NARX Model**: y(t) = F(y(t-1),...,y(t-ny), u(t-nk),...,u(t-nk-nu+1))
- **Hammerstein Model**: u(t) -> g(u) = v(t) -> G(q)v(t) = y(t)
- **Wiener Model**: u(t) -> x(t) = G(q)u(t) -> y(t) = h(x(t))
- **Volterra Series**: y(t) = h0 + sum h1(tau)u(t-tau) + sum sum h2(tau1,tau2)u(t-tau1)u(t-tau2) + ...
- **Bilinear SS**: x(t+1) = A x(t) + sum N_i x(t) u_i(t) + B u(t)

## Core Theorems

1. **Least Squares Optimality**: theta_hat = (Phi^T Phi)^{-1} Phi^T y minimizes ||y-Phi*theta||^2
2. **AIC Consistency (Akaike 1974)**: AIC = N*ln(V_N) + 2d is asymptotically unbiased for KL divergence
3. **BIC Consistency (Schwarz 1978)**: BIC selects true model with probability -> 1 as N -> inf
4. **PE Condition**: Regressor matrix must have full column rank for unique LS solution
5. **Regularization Monotonicity**: L2 penalty lambda > 0 reduces effective degrees of freedom
6. **Gauss-Newton Convergence**: For linear-in-params models, GN converges in one iteration
7. **OLS Error Reduction Ratio (Chen-Billings-Luo 1989)**: ERR_i = g_i^2(w_i^T w_i)/(y^T y)

## Core Algorithms

1. **Gauss-Newton**: theta_{k+1} = theta_k - (J^T J)^{-1} J^T e
2. **Levenberg-Marquardt**: theta_{k+1} = theta_k - (J^T J + lambda*I)^{-1} J^T e
3. **Steepest Descent**: theta_{k+1} = theta_k - alpha*nabla V with Armijo line search
4. **Conjugate Gradient**: Fletcher-Reeves beta_k = ||g_k||^2/||g_{k-1}||^2
5. **Orthogonal Least Squares**: Basis selection via maximum ERR
6. **Recursive Least Squares**: theta(t) = theta(t-1) + K(t)(y(t) - phi^T theta(t-1))

## Classical Problems

1. Hammerstein system identification from I/O data
2. Wiener system identification with static output nonlinearity
3. NARX model order and structure selection
4. DC motor with nonlinear Coulomb + viscous friction
5. Bilinear state-space identification from trajectory data
6. Volterra kernel estimation from broadband excitation

## Nine-School Curriculum Mapping

| School | Course | Topics Covered |
|--------|--------|----------------|
| MIT | 6.241J Dynamic Systems | State-space ID, nonlinear dynamics |
| Stanford | AA203 Optimal Ctrl | Least squares, regularization |
| Berkeley | EE222 Nonlinear Systems | Nonlinear modeling, identifiability |
| CMU | 24-677 Nonlinear Ctrl | NARMAX methods, basis functions |
| Princeton | MAE 546 Optimal Ctrl | Parameter estimation |
| Caltech | CDS140 Nonlinear Dynamics | Volterra series, bilinear systems |
| Cambridge | 4F3 Nonlinear Ctrl | Hammerstein-Wiener models |
| Oxford | B4 Predictive Ctrl | Data-driven identification |
| ETH | 227-0216 Sys Identification | PEM, nonlinear gray-box ID |

## Reference Textbooks

- Ljung (1999) "System Identification: Theory for the User"
- Nelles (2001) "Nonlinear System Identification"
- Billings (2013) "Nonlinear System Identification: NARMAX Methods"
- Nocedal & Wright (2006) "Numerical Optimization"
- Haber & Keviczky (1999) "Nonlinear System Identification"

## File Statistics

- include/: 4 headers, 1088 lines
- src/: 4 C files + 1 Lean file, 3868 + 354 = 4222 lines
- Total include/ + src/ (C only): 4956 lines (exceeds 3000 threshold)
- tests/: 1 test file with 16 test suites
- examples/: 4 end-to-end examples
- docs/: 5 knowledge documents

## Build and Test

```bash
make          # Build static library
make test     # Run all tests (16 test suites)
make examples # Run all examples
make clean    # Clean build artifacts
```
