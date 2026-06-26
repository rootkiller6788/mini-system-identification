# Course Tree — Prerequisites for mini-uncertainty-quantification

## Internal Dependencies (within mini-complex-control-theory)

```
mini-uncertainty-quantification
├── mini-closed-loop-identification (uses UQ for closed-loop param bounds)
├── mini-prediction-error-method (PEM: asymptotic covariance theory)
├── mini-regularized-least-squares (Bayesian interpretation → UQ)
├── mini-subspace-identification (stochastic realization → UQ)
├── mini-nonlinear-system-id (nonlinear UQ methods)
├── mini-frequency-domain-id (frequency-domain uncertainty)
├── [Prior modules]
│   ├── 3. mini-linear-system-theory (state-space, observability)
│   ├── 4. mini-nonlinear-dynamics-chaos (nonlinear models)
│   ├── 6. mini-stability-theory (Lyapunov stability → robustness)
│   ├── 7. mini-information-geometry (Fisher information geometry)
│   └── 9. mini-information-theoretic-ctrl (info-theoretic bounds)
```

## External Dependencies (mathematical foundations)
- Probability theory: distributions, expectation, variance
- Linear algebra: matrix decompositions (SVD, Cholesky, eigen)
- Statistics: estimation theory, hypothesis testing
- Optimization: maximum likelihood, least squares
- Numerical analysis: integration, ODE solvers

## Downstream Dependencies (modules that depend on this one)
- 11. mini-realization-minimality (uncertainty in realization)
- 12. mini-optimal-control (stochastic LQR with UQ)
- 13. mini-robust-control (requires uncertainty descriptions)
- 14. mini-adaptive-control (online parameter UQ)
- 16. mini-stochastic-filtering (Kalman filter UQ)
- 17. mini-model-predictive-control (stochastic MPC)
- 20. mini-networked-control (communication uncertainty)

## L9 Research Frontiers
- Bayesian Deep Learning UQ
- Information-Theoretic UQ Limits
- Distributionally Robust Optimization
- Conformal Prediction
