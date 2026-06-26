# Knowledge Graph ? Wiener-Hammerstein System Identification

## L1: Definitions

| ID | Term | Definition | C Type | Implemented |
|----|------|-----------|--------|-------------|
| 1.1 | Wiener-Hammerstein Model | L1?N?L2 block cascade: u?L1?x?N?w?L2?y | `WH_Model` | ? wh_model.h |
| 1.2 | Wiener Model | L?N cascade (no L2 dynamics) | `WH_Model` with L2=1 | ? Special case |
| 1.3 | Hammerstein Model | N?L cascade (no L1 dynamics) | `WH_Model` with L1=1 | ? Special case |
| 1.4 | Static Nonlinearity | Memoryless mapping x?f(x) | `WH_Nonlinearity` | ? wh_nonlinear.h |
| 1.5 | Linear Dynamic Block | LTI system H(q) | `WH_LinearBlock` | ? wh_linear.h |
| 1.6 | FIR Filter | Finite Impulse Response: H(q)=?b_i q^{-i} | `WH_LIN_FIR` | ? wh_linear.h |
| 1.7 | IIR/TF Filter | Transfer Function: H(q)=B(q)/A(q) | `WH_LIN_IIR_TF` | ? wh_linear.h |
| 1.8 | State-Space Model | dx/dt=Ax+Bu, y=Cx+Du | `WH_LIN_STATE_SPACE` | ? wh_linear.h |
| 1.9 | BIBO Stability | Bounded Input ? Bounded Output | `wh_model_is_stable()` | ? |
| 1.10 | Noise Model | OE/BJ structure: y=G?u+H?e | `WH_NoiseModel` | ? wh_model.h |

## L2: Core Concepts

| ID | Concept | Description | Implementation |
|----|---------|-------------|----------------|
| 2.1 | Block-oriented Models | Decomposition into LTI + NL blocks | WH_Model cascade |
| 2.2 | Identifiability | Unique parameter determination from I/O data | Identification methods |
| 2.3 | Persistence of Excitation | Input must be "rich enough" for identification | Signal design |
| 2.4 | Prediction Error | ?(t,?)=y(t)??(t|?) | wh_identification.c |
| 2.5 | Convergence Analysis | Iterative algorithm convergence properties | Loss monitoring |
| 2.6 | Model Structure Selection | Choosing L1/N/L2 orders | Order selection API |
| 2.7 | Parameterization | Representing model in identifiable form | Multiple init methods |
| 2.8 | Scaling Ambiguity | Gain exchange between blocks | Normalization in BLA |

## L3: Mathematical Structures

| ID | Structure | Mathematical Form | Implementation |
|----|-----------|-------------------|----------------|
| 3.1 | FIR Convolution | y[k]=?b_i?u[k-i] | `wh_linear_evaluate()` |
| 3.2 | IIR Difference Equation | y[k]=?b_i?u[k-i]??a_i?y[k-i] | `wh_linear_evaluate()` |
| 3.3 | State-Space Update | x[k+1]=A?x[k]+B?u[k], y=C?x+D?u | `wh_linear_evaluate()` |
| 3.4 | Polynomial NL | f(x)=?c_i?x^i | `wh_nl_evaluate()` |
| 3.5 | Cubic Spline | Piecewise C? polynomial | `wh_nl_init_spline()` |
| 3.6 | Saturation Function | f(x)=K?sat(x/L) | `wh_nl_evaluate()` |
| 3.7 | Sigmoid Function | f(x)=a/(1+exp(-b(x-c))) | `wh_nl_evaluate()` |
| 3.8 | RBF Network | f(x)=?w_i?exp(?(x-c_i)?/(2??)) | `wh_nl_evaluate()` |
| 3.9 | Frequency Response | H(e^{j?})=B(e^{j?})/A(e^{j?}) | `wh_linear_freq_response()` |
| 3.10 | Jury Stability Criterion | Discrete-time stability test | `wh_linear_is_stable()` |

## L4: Fundamental Laws

| ID | Theorem/Law | Statement | Verification |
|----|-------------|-----------|-------------|
| 4.1 | Cascade Identity | L1=N=L2=id ? WH=id | Test + Lean |
| 4.2 | Wiener Linearity | L2=1 + N linear ? WH is LTI | Lean thm 2 |
| 4.3 | Hammerstein Linearity | L1=1 + N linear ? WH is LTI | Lean thm 3 |
| 4.4 | DC Gain Decomposition | G_dc = G1_dc ? N'(0) ? G2_dc (linear N) | Lean thm 4 |
| 4.5 | Delay Additivity | ?_WH = ?_L1 + ?_L2 | Test + Lean thm 5 |
| 4.6 | Stability Inheritance | L1,L2 stable + N bounded ? WH BIBO stable | Validation check |
| 4.7 | Scaling Ambiguity | (L1??) ? (N/?) ? (L2) is equivalent | Implementation constraint |
| 4.8 | Odd NL Preservation | N odd ? WH(-u) = -WH(u) | Lean thm 6 |
| 4.9 | Saturation Boundedness | |sat(x,L)| ? L for L ? 0 | Lean thm 7 |
| 4.10 | Composition Associativity | (L2?N)?L1 = L2?(N?L1) | Lean thm 8 |

## L5: Algorithms/Methods

| ID | Algorithm | Description | File |
|----|-----------|-------------|------|
| 5.1 | BLA Identification | Best Linear Approximation via multisines | `wh_ident_bla()` |
| 5.2 | Iterative Method | Alternating L1/N/L2 estimation | `wh_ident_iterative()` |
| 5.3 | Over-parameterization | Single regression + SVD projection | `wh_ident_overparam()` |
| 5.4 | PEM with LM | Levenberg-Marquardt gradient optimization | `wh_ident_pem_gradient()` |
| 5.5 | Spline Construction | Natural cubic spline via tridiagonal solver | `wh_nl_init_spline()` |
| 5.6 | Linear Regression | Cholesky-based normal equations solver | `linear_regression()` |
| 5.7 | Frequency Sweep | Chirp-based spectral analysis | `wh_sim_frequency_sweep()` |
| 5.8 | Model Order Selection | AIC/BIC grid search | `wh_ident_order_selection()` |
| 5.9 | Monte Carlo Simulation | Multi-realization statistics | `wh_sim_monte_carlo()` |
| 5.10 | RBF Network Training | Center/width/weight parameterization | `wh_nl_init_rbf()` |

## L6: Canonical Problems

| ID | Problem | Description | Example |
|----|---------|-------------|---------|
| 6.1 | WH Benchmark | Schoukens et al. (2015) identification | example_wh_benchmark.c |
| 6.2 | Actuator Saturation | Motor + voltage limit + load | example_wh_industrial.c |
| 6.3 | PK/PD Dose-Response | Drug absorption ? effect kinetics | example_wh_bio.c |
| 6.4 | Model Validation | Residual analysis, cross-validation | test_wh_model.c |
| 6.5 | System Simulation | Time-domain cascade simulation | `wh_sim_run()` |

## L7: Applications

| ID | Application | Domain | Example |
|----|-------------|--------|---------|
| 7.1 | Industrial Actuators | Control engineering (electric motor, Tesla) | example_wh_industrial.c |
| 7.2 | Pharmacokinetics | Biomedical (dose-response, NHS) | example_wh_bio.c |
| 7.3 | Chemical Processes | Process control (chemical reactor) | (documented) |

## L8: Advanced Topics

| ID | Topic | Status |
|----|-------|--------|
| 8.1 | MIMO WH Models | Documented, not implemented |
| 8.2 | Continuous-time WH | Implemented via state-space with Ts=0 |
| 8.3 | Nonparametric WH (kernel-based) | Documented |
| 8.4 | Recursive Identification | Partial (gradient descent online) |
| 8.5 | Bayesian WH Identification | Documented |

## L9: Research Frontiers

| ID | Topic | Status |
|----|-------|--------|
| 9.1 | Deep Learning + WH (neural network N) | Documented |
| 9.2 | Physics-informed WH Identification | Documented |
| 9.3 | WH for Nonlinear MPC | Documented |
