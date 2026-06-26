# Course Alignment ? Wiener-Hammerstein System Identification

## Nine-School Curriculum Mapping

| School | Course | Topics | Our Module |
|--------|--------|--------|------------|
| **MIT** | 2.151 Advanced System Identification | Block-oriented models, PEM, instrumental variables | WH iterative + PEM gradient |
| **Stanford** | EE263 Linear Dynamical Systems | State-space ? TF conversion, stability analysis | `wh_linear_convert_to_iir()`, Jury test |
| **Berkeley** | EE221A Linear Systems | Controllability, observability, minimal realization | State-space representation |
| **CMU** | 18-771 Linear Systems | FIR/IIR filter design, frequency response | `wh_linear_freq_response()` |
| **Princeton** | MAE 546 Optimal Control | Pontryagin + dynamic programming in nonlinear control | WH as system model for NMPC |
| **Caltech** | CDS 110 Dynamical Systems | Linearization around trajectories, Lyapunov stability | Stability validation |
| **Cambridge** | 4F3 System Identification | Nonlinear model structures, validation | Residual analysis, cross-validation |
| **Oxford** | Control Systems | Signal processing, spectral analysis | Multisine design, BLA |
| **ETH** | 227-0216 Identification | Frequency-domain identification, BLA, nonlinear models | `wh_ident_bla()`, frequency sweep |

## Course Chapter Mapping

### MIT 2.151 ? Advanced System Identification
- Ch 4: Nonlinear Model Structures ? WH models (L1)
- Ch 5: Prediction Error Methods ? `wh_ident_pem_gradient()` (L5)
- Ch 7: Nonlinear Black-Box Models ? Block-oriented models (L2)
- Ch 14: Model Validation ? `wh_validate_comprehensive()` (L4)

### Stanford EE263 ? Linear Dynamical Systems
- Ch 5: State-Space to Transfer Function ? `wh_linear_convert_to_iir()`
- Ch 7: Stability of Discrete-Time Systems ? Jury criterion (L3)
- Ch 8: Frequency Domain ? `wh_linear_freq_response()` (L3)

### ETH 227-0216 ? System Identification
- Ch 3: Excitation Signals ? `wh_signal_multisine()` (L3)
- Ch 4: Best Linear Approximation ? `wh_ident_bla()` (L5)
- Ch 6: Nonlinear Model Identification ? WH identification (L5)
- Ch 8: Model Validation ? Residual analysis (L4)
