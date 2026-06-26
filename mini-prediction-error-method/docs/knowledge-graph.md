# Knowledge Graph: Prediction Error Method (PEM)

## L1 — Definitions
| Term | Definition | Code Location |
|------|-----------|---------------|
| Prediction Error | eps(t,theta) = y(t) - y_hat(t|theta) | pem_predictor.h |
| One-Step-Ahead Predictor | y_hat(t|theta) = optimal predictor of y(t) given Z^{t-1} | pem_predictor.h |
| Model Structure M(theta) | Parameterized transfer function family {G(q,theta), H(q,theta)} | pem_core.h |
| Criterion Function | V_N(theta) = (1/N) sum l(eps(t,theta)) | pem_criterion.h |
| Parameter Vector | theta = vector of unknown model coefficients | pem_core.h |
| Pseudo-Regressor | psi(t,theta) = -d(eps)/d(theta) | pem_criterion.h |
| Information Matrix | I = E[psi(t) psi^T(t)] (Fisher information) | pem_core.h |
| Model Order | na, nb, nc, nd, nf = polynomial degrees | pem_core.h |
| Input Delay | nk = number of samples from u to y | pem_core.h |
| Sampling Interval | Ts = time between consecutive samples | pem_core.h |

## L2 — Core Concepts
| Concept | Description | Code Location |
|---------|-------------|---------------|
| Linear Regression Form | ARX predictor: y_hat = phi^T theta | pem_predictor.c |
| Pseudo-Linear Regression | ARMAX/OE/BJ use parameter-dependent regressors | pem_predictor.c |
| Identifiability | Parameters uniquely determine model behavior | pem_model.c |
| Persistence of Excitation | Input must be sufficiently rich for unique estimation | docs/ |
| Bias-Variance Tradeoff | Higher-order models have lower bias but higher variance | pem_validation.c |
| Consistency | theta_hat_N -> theta_0 as N -> inf (under conditions) | docs/ |
| Asymptotic Normality | sqrt(N)(theta_hat - theta_0) -> N(0, P) | docs/ |
| Cramer-Rao Lower Bound | Cov(theta_hat) >= I^{-1} | pem_optimizer.c |

## L3 — Mathematical Structures
| Structure | Description | Code Location |
|-----------|-------------|---------------|
| PEMPolynomial | P(q) = p_0 + p_1 q^{-1} + ... | pem_core.h |
| PEMTransferFunction | G(q) = N(q)/D(q), D monic | pem_core.h |
| PEMData | Z^N = {u(1),y(1),...,u(N),y(N)} | pem_core.h |
| PEMResult | Estimation result with covariance | pem_core.h |
| PEMPredictorState | Circular buffer for recursive prediction | pem_predictor.h |
| PEMModelCallbackData | Closure for optimizer callbacks | pem_model.c |
| Normal Equations Matrix | Phi^T Phi (npar x npar) | pem_model.c |

## L4 — Fundamental Laws
| Theorem/Law | Statement | Code Location |
|-------------|-----------|---------------|
| Least Squares Consistency | theta_LS -> theta_0 if u is PE and e is white | pem_model.c |
| Gauss-Markov Theorem | LS is BLUE under homoskedastic white noise | pem_model.c |
| Akaike's Information Criterion | AIC = log(V) + 2d/N | pem_validation.c |
| Ljung-Box Test | Q-test for residual whiteness | pem_validation.c |
| Armijo Condition | Sufficient decrease: f(x+ap) <= f(x) + ca g^T p | pem_optimizer.c |
| Robbins-Monro Conditions | sum alpha_k = inf, sum alpha_k^2 < inf | pem_optimizer.c |

## L5 — Algorithms/Methods
| Algorithm | Description | Code Location |
|-----------|-------------|---------------|
| Gauss-Newton | p_k = -H_GN^{-1} g_k | pem_optimizer.c |
| Levenberg-Marquardt | p_k = -(H + lambda I)^{-1} g_k | pem_optimizer.c |
| SGD | theta_{k+1} = theta_k - alpha_k g_k | pem_optimizer.c |
| Cholesky Decomposition | A = L L^T for SPD systems | pem_optimizer.c |
| Armijo Backtracking | Line search with geometric step reduction | pem_optimizer.c |
| Closed-Form LS | theta = (Phi^T Phi)^{-1} Phi^T Y | pem_model.c |
| Pseudo-Regressor Filtering | F(q) psi(t) = phi(t) for OE/ARMAX gradients | pem_criterion.c |
| Finite Difference Gradient | For BJ model complexity | pem_criterion.c |
| Recursive Residual Computation | ARMAX/BJ predictor state management | pem_predictor.c |

## L6 — Canonical Problems
| Problem | Description | Code Location |
|---------|-------------|---------------|
| ARX Estimation | Linear regression, closed-form LS solution | pem_model.c, example1 |
| ARMAX Estimation | Pseudo-linear regression with MA noise | pem_model.c, example3 |
| OE Estimation | Non-convex, LM optimization | pem_model.c, example2 |
| BJ Estimation | Most general linear polynomial model | pem_model.c |
| FIR Estimation | Feedforward only, LS solution | pem_model.c |
| Model Order Selection | AIC/AICc/BIC/FPE comparison | example4 |
| k-Step-Ahead Prediction | OE k-step for validation | pem_predictor.c |

## L7 — Applications
| Application | Description | Code Location |
|-------------|-------------|---------------|
| Process Control ID | Industrial process model identification | examples/ |
| Flight Dynamics | Aircraft transfer function estimation | examples/ (OE suitable) |
| Econometric Modeling | Time series with correlated noise (ARMAX) | examples/ |
| Climate Model Calibration | Transfer function estimation from data | examples/ |

## L8 — Advanced Topics
| Topic | Description | Code Location |
|-------|-------------|---------------|
| Levenberg-Marquardt Damping | Adaptive lambda strategy for robustness | pem_optimizer.c |
| Robust Loss Functions | Huber, Vapnik loss alternatives | pem_criterion.h |
| Regularized PEM | Tikhonov regularization for ill-conditioned Hessians | pem_criterion.c |
| Multi-step Prediction | k-step ahead for model validation | pem_predictor.c |
| K-Fold Cross-Validation | Block CV for time series | pem_validation.c |

## L9 — Research Frontiers
| Topic | Description |
|-------|-------------|
| Kernel-Based PEM | Regularization in RKHS for impulse response estimation |
| Sparse Identification | L1-regularized PEM for parsimonious models |
| Deep PEM Hybrids | Neural network parameterized predictors |
| Recursive PEM | Online adaptation for time-varying systems |
| Robust PEM | Outlier-resistant estimation with bounded influence functions |