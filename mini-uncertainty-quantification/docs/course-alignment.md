# Course Alignment — UQ Module

| School | Course | Relevant Topics | Our Implementation |
|--------|--------|----------------|--------------------|
| **MIT** | 6.241J Dynamic Systems | State estimation, Kalman filtering | `uq_ut_propagate` (Unscented Transform), FOSM |
| **MIT** | 16.323 Optimal Ctrl | Stochastic optimal control, LQG | `uq_decision_bayes_risk` |
| **MIT** | 6.832 Underactuated | Uncertainty in dynamics | `uq_mc_propagate`, Rosenblueth |
| **Stanford** | AA203 Optimal Ctrl | Robustness, disturbance | `uq_fosm_propagate`, FORM |
| **Stanford** | EE363 Convex Opt | Distributionally robust opt | (L8 gap) |
| **Berkeley** | EE221A Linear Systems | Kalman filtering, estimation | `uq_lm_fit`, parameter CI |
| **Berkeley** | EE222 Nonlinear | Nonlinear estimation | `uq_ut_create`, sigma-point methods |
| **CMU** | 18-771 Linear Sys | System identification | `uq_ensemble_compute_statistics` |
| **Princeton** | ELE 530 Estimation | Estimation theory, CRLB | `uq_ci_from_normal`, cramer_rao (Lean) |
| **Caltech** | CDS110 Intro Ctrl | Robustness analysis | `uq_rosenblueth_2p`, MC propagation |
| **Cambridge** | 4F3 Nonlinear Ctrl | Nonlinear UQ | PCE, sparse grid collocation |
| **Oxford** | B4 Predictive Ctrl | Model uncertainty in MPC | `uq_calval_assess`, cross-validation |
| **ETH** | 227-0216 Sys Identification | System identification UQ | Full parameter UQ suite |
| **ETH** | 227-0220 Model Reduction | Reduced-order UQ | KL expansion, PCE truncation |

## Reference Textbooks
- Smith, R.C. (2013). *Uncertainty Quantification: Theory, Implementation, and Applications.* SIAM.
- Sullivan, T.J. (2015). *Introduction to Uncertainty Quantification.* Springer.
- Gelman, A. et al. (2013). *Bayesian Data Analysis* (3rd ed.). Chapman & Hall/CRC.
- Ljung, L. (1999). *System Identification: Theory for the User* (2nd ed.). Prentice Hall.
- Robert, C.P. & Casella, G. (2004). *Monte Carlo Statistical Methods.* Springer.
- Efron, B. & Tibshirani, R.J. (1993). *An Introduction to the Bootstrap.* Chapman & Hall.
- Xiu, D. (2010). *Numerical Methods for Stochastic Computations.* Princeton.
- Saltelli, A. et al. (2008). *Global Sensitivity Analysis: The Primer.* Wiley.
