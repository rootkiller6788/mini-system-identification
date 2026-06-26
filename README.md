# Mini System Identification

A collection of **from-scratch, zero-dependency C implementations** of system identification theory — building mathematical models of dynamical systems from observed input-output data. Each sub-module maps to MIT and Stanford courses, bridging theory and practice by translating textbook equations into runnable C code.

## Sub-Modules

| Sub-Module | Topics | Key Courses |
|------------|--------|-------------|
| [mini-closed-loop-identification](mini-closed-loop-identification/) | Direct/indirect closed-loop ID, instrumental variables (IV), joint I/O identification, subspace closed-loop methods, Youla parameterization, model validation | MIT 6.241J, Ljung (1999) Ch.13 |
| [mini-frequency-domain-id](mini-frequency-domain-id/) | FRF estimation (ETFE, H1/H2/coherence), spectral/cepstral analysis, swept-sine/multisine excitation, s-to-z domain conversion, resonance detection, describing functions | MIT 6.241J, Pintelon & Schoukens (2012) |
| [mini-nonlinear-system-id](mini-nonlinear-system-id/) | NARX, NARMAX, Wiener/Hammerstein/Wiener-Hammerstein models, Gauss-Newton, Levenberg-Marquardt, orthogonal least squares, recursive least squares, cross-validation | MIT 6.241J, Nelles (2001) |
| [mini-prediction-error-method](mini-prediction-error-method/) | ARX, ARMAX, OE, BJ model structures, quadratic/Huber/Vapnik loss functions, Newton/Gauss-Newton optimization, AIC/BIC model selection, residual analysis | MIT 6.241J, Ljung (1999), Söderström & Stoica (1989) |
| [mini-regularized-least-squares](mini-regularized-least-squares/) | Ridge (L2), Lasso (L1), Elastic Net, kernel-based methods (Stable Spline, TC), Cholesky/QR solvers, bias-variance tradeoff, hyperparameter tuning | MIT 6.241J, Stanford EE364A, Pillonetto et al. (2014) |
| [mini-subspace-identification](mini-subspace-identification/) | N4SID, MOESP, CVA, Hankel matrices, oblique projection, SVD-based order selection, LQ decomposition, instrumental variable subspace | MIT 6.241J, Van Overschee & De Moor (1996), Katayama (2005) |
| [mini-uncertainty-quantification](mini-uncertainty-quantification/) | Bayesian inference, MCMC (Metropolis-Hastings/Gibbs), polynomial chaos expansion, Sobol sensitivity indices, credible/confidence regions, parameter uncertainty propagation | Stanford CME 206, Smith (2013), Sullivan (2015) |
| [mini-wiener-hammerstein](mini-wiener-hammerstein/) | FIR/IIR/SS linear blocks, static nonlinearities (deadzone/saturation/hysteresis), best linear approximation (BLA), iterative identification, over-parameterization, nonlinear distortion analysis | MIT 6.241J, Schoukens et al. (2005) |

## Design Philosophy

- **Zero external dependencies** — pure C (C99/C11), only `libc` and `libm`
- **Self-contained modules** — each directory has its own `Makefile`, `include/`, `src/`, `examples/`, `demos/`, `tests/`, `docs/`
- **Theory-to-code mapping** — every module implements canonical algorithms from foundational textbooks (Ljung, Söderström & Stoica, Van Overschee & De Moor)
- **Practical demos** — DC motor identification, quadrotor UAV, chemical process, GPS denoising, biomedical glucose, flutter analysis, and more

## Building

Each module is standalone. Navigate to a module directory and run:

```bash
cd mini-closed-loop-identification
make all    # build everything
make test   # run tests
```

Requires **GCC** and **GNU Make**.

## Project Structure

```
mini-system-identification/
├── mini-closed-loop-identification/   # Direct/indirect closed-loop ID, IV, joint I/O, Youla parameterization
├── mini-frequency-domain-id/          # FRF estimation, spectral analysis, swept-sine/multisine, describing functions
├── mini-nonlinear-system-id/          # NARX/NARMAX, Wiener/Hammerstein models, Gauss-Newton, LM, recursive LS
├── mini-prediction-error-method/      # ARX/ARMAX/OE/BJ, loss functions, Newton/Gauss-Newton, AIC/BIC
├── mini-regularized-least-squares/    # Ridge/Lasso/Elastic Net, kernel-based methods (SS, TC), bias-variance
├── mini-subspace-identification/      # N4SID/MOESP/CVA, Hankel matrices, oblique projection, SVD order selection
├── mini-uncertainty-quantification/   # Bayesian UQ, MCMC, polynomial chaos, Sobol sensitivity, credible regions
└── mini-wiener-hammerstein/           # FIR/IIR/SS blocks, nonlinearities (deadzone/saturation/hysteresis), BLA
```

## License

MIT
