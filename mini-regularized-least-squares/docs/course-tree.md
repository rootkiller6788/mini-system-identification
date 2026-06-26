# Course Dependency Tree -- mini-regularized-least-squares

## Prerequisites (What this module depends on)

### From mini-complex-control-theory:
- **3. mini-linear-system-theory**: State-space models, matrix algebra, eigenvalues
- **10. mini-system-identification**: Basic system ID concepts (parent module)

### General Mathematics:
- Linear algebra: matrix multiplication, transpose, inverse, determinant
- Numerical linear algebra: Cholesky, QR, SVD factorizations
- Optimization: convex optimization, gradient descent, KKT conditions
- Statistics: bias, variance, hypothesis testing, information criteria
- System theory: transfer functions, impulse response, difference equations

## Dependents (What depends on this module)

### Within mini-complex-control-theory:
- **11. mini-realization-minimality**: Uses identified models for state-space realization
- **12. mini-optimal-control**: Uses identified models for LQR/MPC design
- **13. mini-robust-control**: Needs uncertainty bounds from regularized estimation
- **14. mini-adaptive-control**: Online identification builds on RLS foundation
- **16. mini-stochastic-filtering**: Joint state-parameter estimation
- **17. mini-model-predictive-control**: Data-driven MPC model identification

### External disciplines:
- Machine learning: ridge regression, LASSO, elastic net
- Signal processing: denoising, deconvolution, compressed sensing
- Econometrics: regularized regression for economic forecasting
- Bioinformatics: gene selection via LASSO, biomarkers
- Geophysics: seismic inversion, tomography

## Learning Path

```
Linear Algebra -> Matrix Factorizations -> Ordinary Least Squares
    -> Ridge Regression -> LASSO -> Elastic Net -> Kernel Methods
    -> System Identification -> Model Validation -> Applications
    -> Advanced: Empirical Bayes, Hyperparameter Optimization
```

## Implementation Dependencies

```
rls_core.h/.c          (matrix algebra, factorizations)
  |
  +-> rls_regularizers.h/.c  (penalties, prox operators)
  +-> rls_solvers.h/.c       (all solvers depend on core + regularizers)
  +-> rls_models.h/.c        (regressor construction depends on core)
  +-> rls_validation.h/.c    (CV/GCV/etc depend on solvers)
  +-> rls_kernel.h/.c        (kernel methods depend on core + solvers)
  +-> rls_applications.c     (applications depend on all above)
```

## L9 Research Frontiers (Future)

This module is well-positioned for extension to:
- **Deep kernel learning**: Replace fixed kernel with learned neural feature map
- **Sparse variational Gaussian processes**: Scalable kernel methods for big data
- **Physics-informed regularization**: Encode physical constraints in kernel design
- **Meta-learning for system ID**: Learn to regularize across multiple systems
