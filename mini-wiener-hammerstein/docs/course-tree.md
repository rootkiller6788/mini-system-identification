# Course Tree ? Prerequisite Dependencies

## Module Dependency Graph

```
mini-system-identification (parent)
??? mini-prediction-error-method  ? PEM theory (prerequisite)
??? mini-subspace-identification  ? SVD-based methods
??? mini-nonlinear-system-id      ? Nonlinear model classes
??? mini-frequency-domain-id      ? BLA, FRF estimation
??? mini-regularized-least-squares ? Regularized regression
??? mini-uncertainty-quantification ? Model uncertainty
??? mini-closed-loop-identification ? Closed-loop WH
??? mini-wiener-hammerstein       ? THIS MODULE
```

## Within-Module Prerequisites

```
L1: Definitions
??? WH_Model, WH_LinearBlock, WH_Nonlinearity, WH_NoiseModel
??? Required before: Everything else

L2: Core Concepts
??? Block-oriented decomposition, identifiability, PE
??? Required before: L5 (identification algorithms)

L3: Mathematical Structures
??? FIR/IIR/SS, polynomial/spline/sigmoid/RBF
??? Frequency response, Jury stability
??? Required before: L4, L5

L4: Fundamental Laws
??? Cascade identity, delay additivity, stability inheritance
??? Scaling ambiguity, odd NL preservation
??? Required before: L6 (canonical problems)

L5: Algorithms
??? BLA, iterative, over-param, PEM gradient
??? Spline construction, regression solver
??? Required before: L6, L7

L6: Canonical Problems
??? WH benchmark, actuator saturation, PK/PD dose-response
??? Required before: L7 (applications)

L7: Applications
??? Industrial actuators, pharmacokinetics, chemical processes
??? (L1-L6 prerequisite)

L8: Advanced Topics
??? MIMO WH, recursive WH, Bayesian WH
??? (L5 prerequisite)

L9: Research Frontiers
??? DL + WH, physics-informed WH, WH for NMPC
??? (L5-L8 prerequisite)
```

## External Course Prerequisites

| Our Module | Requires | From Course |
|------------|----------|-------------|
| Linear regression solver | Matrix algebra, Cholesky decomposition | Linear Algebra |
| Jury stability criterion | Polynomial root analysis | Discrete-Time Systems |
| BLA estimation | FRF estimation, spectral analysis | Signal Processing |
| PEM gradient descent | Optimization theory, Newton/Gauss-Newton | Numerical Optimization |
| Cubic spline construction | Tridiagonal solver | Numerical Methods |
| PRBS generation | Galois fields, LFSR theory | Digital Signal Processing |
| AIC/BIC model selection | Information theory | Statistical Inference |

## Target Audience

- Graduate students in system identification, control theory, or signal processing
- Prerequisites: linear systems theory, discrete-time signal processing, basic optimization
- Complementary: mini-nonlinear-system-id (for broader nonlinear model classes)
