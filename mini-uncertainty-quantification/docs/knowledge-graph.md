# Knowledge Graph — Uncertainty Quantification

## L1: Definitions (Complete)
- `UQUncertaintyType` — Aleatory, Epistemic, Measurement, Model Form, Parameter, Numerical, Interpolation, Data Sparsity
- `UQDistributionType` — Normal, Uniform, Student-t, Chi2, F, Log-Normal, Beta, Gamma, Exponential, Weibull, Cauchy, Multivariate Normal, Dirichlet, Wishart, Empirical, Gaussian Process, KDE
- `UQIntervalType` — Confidence, Prediction, Credible, Tolerance, Simultaneous Band
- `UQPropagationMethod` — 14 methods from linearization to subset simulation
- `UQValidationMetric` — RMSE, MAE, MAPE, R², concordance, Theil's U, MASE, Wilks, BF
- `UQDecisionCriterion` — Minimax, Bayes Risk, Optimistic, Pessimistic, Hurwicz, Info-Gap
- `UQMCStrategy` — 13 strategies from simple MC to orthogonal arrays
- `UQBootstrapMethod` — 10 variants from standard to Bayesian bootstrap
- `UQPCEType` — Hermite, Legendre, Laguerre, Jacobi, Gegenbauer, Chebyshev
- `UQGPKernelType` — SE, Matern-3/2, Matern-5/2, Exponential, Rational Quadratic, Periodic, Linear, Neural Network

## L2: Core Concepts (Complete)
- Frequentist vs. Bayesian UQ paradigms
- Aleatory (irreducible) vs. Epistemic (reducible) uncertainty decomposition
- Confidence intervals (Neyman) — coverage frequency interpretation
- Credible intervals (Bayesian) — posterior probability interpretation
- Uncertainty propagation: input uncertainty → output uncertainty
- Model validation under uncertainty
- Bootstrap resampling as non-parametric frequentist UQ
- MCMC as the workhorse of Bayesian UQ
- Gaussian Process as a Bayesian nonparametric emulator

## L3: Mathematical Structures (Complete)
- Probability distributions with analytical PDF, CDF, quantile, sampling
- Matrix/Vector operations for linear algebra foundation
- Fisher information matrix and parameter covariance
- Hat matrix and leverage for linear models
- Cholesky decomposition (for MVN sampling, GP, MCMC proposals)
- SVD for condition number and rank
- Karhunen-Loève expansion (eigendecomposition of covariance kernel)
- Polynomial chaos basis functions (orthogonal polynomials)

## L4: Fundamental Laws/Theorems (Complete)
- Cramér-Rao Lower Bound — `cramer_rao_reciprocal` (Lean)
- Variance decomposition — `uncertainty_decomposition_additive` (Lean)
- Bootstrap bias identity — `bootstrap_bias_identity` (Lean)
- OLS: β̂ = (XᵀX)⁻¹Xᵀy with Var(β̂) = σ²(XᵀX)⁻¹
- Gauss-Markov theorem (OLS is BLUE)
- Central Limit Theorem (MC convergence rate O(1/√N))
- Bayes' theorem: posterior ∝ likelihood × prior
- Confidence interval width ∝ 1/√n (asymptotic)

## L5: Algorithms/Methods (Complete)
- Metropolis-Hastings MCMC with adaptive proposal tuning
- Hamiltonian Monte Carlo with leapfrog integration
- Slice sampling (Neal, 2003)
- Adaptive Metropolis (Haario et al., 2001)
- Gibbs sampling framework
- Latin Hypercube Sampling (LHS) with Iman-Conover correlation
- Sobol', Halton, Hammersley quasi-Monte Carlo sequences
- Importance sampling with IS estimator
- Rejection sampling with envelope bound
- Bootstrap: percentile, basic, BCa, studentized, Bayesian
- Polynomial Chaos Expansion (regression-based non-intrusive)
- Unscented Transform with 2n+1 sigma points
- First-Order Second-Moment (FOSM) propagation
- Rosenblueth point estimate method (2-point and 3-point)
- FORM (First-Order Reliability Method, Hasofer-Lind)
- Subset simulation (Au & Beck, 2001)
- Sparse grid stochastic collocation (Smolyak)
- Gaussian Process regression with Cholesky solver
- Kernel density estimation (KDE) with Scott's rule

## L6: Canonical Problems (Complete)
- Linear regression with full UQ (parameter CI, prediction PI, ANOVA, PRESS)
- Bayesian parameter estimation for normal mean via MCMC
- Duffing oscillator amplitude uncertainty propagation (5 methods compared)
- Structural reliability (FORM, probability of failure)
- Model calibration-validation assessment
- Convergence diagnostics (Geweke, Gelman-Rubin R-hat, effective sample size)

## L7: Applications (Complete)
- NASA/F-35 flight control parameter uncertainty bounds
- DC motor system identification with confidence intervals
- Climate model uncertainty quantification
- Nuclear reactor safety (FORM reliability assessment)
- Structural engineering reliability analysis
- Financial risk assessment (VaR via MC propagation)

## L8: Advanced Topics (Partial+ — 3/5)
- ✅ Gaussian Process emulation (Bayesian nonparametric UQ)
- ✅ Polynomial Chaos Expansion (spectral UQ)
- ✅ Adaptive MCMC (Haario et al.)
- ⬜ Multi-fidelity UQ (co-kriging)
- ⬜ Distributionally robust optimization

## L9: Research Frontiers (Partial — documented)
- Bayesian neural networks for UQ
- Information-theoretic limits of UQ
- Meta-complexity of uncertainty quantification
- Deep Gaussian processes
- Conformal prediction

### Key: ✅ = Complete, ⬜ = Missing, ⚠️ = Partial
