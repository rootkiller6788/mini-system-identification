# Course Alignment: Nonlinear System Identification

## Nine-School Curriculum Mapping

### MIT
- **6.241J Dynamic Systems and Control** (Frazzoli, Dahleh)
  - State-space models and system identification
  - Covered: NARX state-space modeling, basis expansions
- **16.323 Principles of Optimal Control**
  - Parameter estimation for dynamic systems
  - Covered: Least squares, Gauss-Newton, LM algorithms

### Stanford
- **AA203 Optimal and Learning-based Control** (Pavone)
  - Data-driven modeling for control
  - Covered: NARX models, neural network models for system ID
- **EE363 Convex Optimization** (Boyd)
  - Regularized least squares, convex cost functions
  - Covered: Regularized identification, Huber loss

### Berkeley
- **EE221A Linear Systems Theory**
  - System identification fundamentals
  - Covered: ARX linear baseline, PE condition
- **EE222 Nonlinear Systems** (Tomlin)
  - Nonlinear system analysis and modeling
  - Covered: Hammerstein, Wiener, Volterra models

### CMU
- **24-677 Nonlinear Control** (Manchester)
  - Nonlinear system modeling techniques
  - Covered: NARMAX methods, basis function selection (OLS)
- **18-771 Linear Systems**
  - State-space identification
  - Covered: Bilinear state-space models

### Princeton
- **MAE 546 Optimal Control and Estimation**
  - Kalman filter, parameter estimation
  - Covered: RLS identification, prediction error methods
- **ELE 530 Estimation and Detection**
  - Statistical estimation theory
  - Covered: AIC/BIC model selection, residual analysis

### Caltech
- **CDS110 Introduction to Control**
  - System identification basics
  - Covered: Signal processing, PE detection
- **CDS140 Nonlinear Dynamics**
  - Volterra series, bilinear systems
  - Covered: Volterra kernel computation, bilinear simulation

### Cambridge
- **4F3 Nonlinear and Predictive Control**
  - Block-structured nonlinear models
  - Covered: Hammerstein and Wiener model implementations
- **4F2 Robust and Multivariable Control**
  - Model validation for control design
  - Covered: NARMAX validation tests (5-test battery)

### Oxford
- **B4 Predictive Control**
  - Data-driven predictive models
  - Covered: NARX prediction, neural network models
- **C20 Adaptive Control**
  - Online parameter estimation
  - Covered: RLS with forgetting factor

### ETH Zurich
- **227-0216 System Identification** (Smith)
  - Comprehensive system identification
  - Covered: PEM framework, nonlinear gray-box ID
- **227-0220 Model Reduction**
  - Model order reduction
  - Covered: OLS basis selection

## Chapter-Level Mapping

| Topic | Primary Reference | Chapter |
|-------|------------------|---------|
| Nonlinear model structures | Ljung (1999) | Ch. 5 |
| Basis function expansions | Nelles (2001) | Ch. 4-6 |
| NARMAX methods | Billings (2013) | Ch. 2-5 |
| Gauss-Newton/LM | Nocedal & Wright (2006) | Ch. 10 |
| Prediction error method | Ljung (1999) | Ch. 7 |
| OLS/ERR algorithm | Billings (2013) | Ch. 3 |
| Model validation tests | Billings & Zhu (1995) | - |
| RLS identification | Ljung (1999) | Ch. 11 |
| Hammerstein-Wiener models | Haber & Keviczky (1999) | Ch. 6-7 |
| Volterra series | Rugh (1981) | Ch. 1-3 |
| Bilinear systems | Mohler (1991) | Ch. 2 |
