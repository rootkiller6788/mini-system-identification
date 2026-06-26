# Course Alignment: Prediction Error Method (PEM)

## Nine-School Curriculum Mapping

### MIT — 6.435 System Identification (Ljung)
| Topic | PEM Module Coverage |
|-------|-------------------|
| Model Structures (ARX, ARMAX, OE, BJ) | PEMModelStructure enum + predictors |
| One-Step-Ahead Prediction | pem_predictor.h/c |
| Prediction Error Criterion | pem_criterion.h/c |
| Gauss-Newton Minimization | pem_optimizer.c |
| Asymptotic Properties | docs/knowledge-graph.md L2 |
| Model Validation | pem_validation.c |

### Stanford — EE264 Digital Signal Processing + System ID
| Topic | PEM Module Coverage |
|-------|-------------------|
| Least Squares Estimation | pem_model.c (ARX LS) |
| Recursive Identification | pem_optimizer.c (SGD) |
| Model Order Selection | example4 / pem_validation.c |
| Spectral Analysis Connection | Transfer function simulation |

### Berkeley — ME237/EE220 System Identification
| Topic | PEM Module Coverage |
|-------|-------------------|
| Prediction Error Framework | Full implementation |
| Nonlinear Optimization | GN, LM, SGD in pem_optimizer.c |
| Model Structure Selection | AIC/BIC/FPE in pem_validation.c |
| Practical Identification | examples/ directory |

### CMU — 18-771 Linear Systems and Control
| Topic | PEM Module Coverage |
|-------|-------------------|
| ARX and State-Space Models | PEM_ARX, PEM_SS structures |
| Least Squares | pem_estimate_arx_ls |
| Kalman Filter Connection | docs/ (SS innovation form) |

### Princeton — MAE 546 System Identification
| Topic | PEM Module Coverage |
|-------|-------------------|
| Prediction Error Methods | Core module |
| Nonlinear Model Structures | OE + BJ implementations |
| Robust Identification | Huber/Vapnik loss functions |

### Caltech — CDS 110/210 System Identification
| Topic | PEM Module Coverage |
|-------|-------------------|
| Frequency-Domain Interpretation | Transfer function simulation |
| Model Validation | Ljung-Box test, cross-correlation |
| Information Criteria | AIC/AICc/BIC/FPE |

### Cambridge — 4F12 System Identification
| Topic | PEM Module Coverage |
|-------|-------------------|
| ARMAX and Box-Jenkins Models | Full implementations |
| Prediction Error Minimization | GN + LM optimizers |
| Recursive Methods | SGD optimizer |

### Oxford — System Identification (Engineering Science)
| Topic | PEM Module Coverage |
|-------|-------------------|
| Linear Polynomial Models | PEM_ARX through PEM_BJ |
| Numerical Optimization | Cholesky, GN, LM |
| Practical Examples | 4 end-to-end examples |

### ETH — 227-0216-00L Control Systems II
| Topic | PEM Module Coverage |
|-------|-------------------|
| System Identification for Control | Full PEM framework |
| Model-Based Control Design | Model simulation + validation |
| Robustness Analysis | Covariance estimation |

## Key Textbooks Covered
1. Ljung, L. (1999) — "System Identification: Theory for the User" (2nd ed.)
2. Soderstrom, T. & Stoica, P. (1989) — "System Identification"
3. Nocedal, J. & Wright, S. (2006) — "Numerical Optimization"