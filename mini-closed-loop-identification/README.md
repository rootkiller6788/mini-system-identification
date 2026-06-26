# mini-closed-loop-identification

Closed-loop system identification — identifying plant models from data collected under feedback control.

## Module Status: COMPLETE ✅

- **L1 Definitions**: Complete (12 typedef struct, 3 enums)
- **L2 Core Concepts**: Complete (10 core concepts implemented)
- **L3 Mathematical Structures**: Complete (TF, SS, Hankel, covariance, frequency response)
- **L4 Fundamental Laws**: Complete (Ljung Thm 13.1, bias formula, consistency theorems)
- **L5 Algorithms/Methods**: Complete (19 algorithms across 8 source files)
- **L6 Canonical Problems**: Complete (3 end-to-end examples)
- **L7 Applications**: Complete (DC motor, Quadrotor UAV, Process control / ISO)
- **L8 Advanced Topics**: Partial (bias analysis, prefiltering, dual Youla, control-relevant validation)
- **L9 Research Frontiers**: Partial (documented in knowledge-graph.md)

## Core Definitions

| Term | Definition |
|------|-----------|
| Closed-loop identification | Estimation of plant model G from data (u,y) collected under feedback u = r - C*y |
| Direct method | Apply open-loop PEM directly to (u,y) data, ignoring feedback |
| Indirect method | Identify CL transfer G_yr first, recover G using known C |
| Joint IO method | Treat CL system as MIMO open-loop, identify plant+controller jointly |
| Instrumental Variable | Use z(t) correlated with u but uncorrelated with e to break feedback correlation |
| Dual Youla | Stable parameterization of all plants stabilized by a given controller |

## Core Theorems

| Theorem | Formula / Statement |
|---------|-------------------|
| Ljung Thm 13.1 | Direct PEM consistent in CL iff H(q,theta) set contains true H0 |
| Bias formula (Forssell & Ljung 1999) | theta* = argmin integral |G0 - G(theta) + B|^2 * Phi_u / |H|^2 dw; B = [H0-H]*Phi_eu/Phi_u |
| Indirect consistency (Van den Hof & Schrama 1993) | Consistent iff r PE, CL model in model set, C exactly known |
| Youla-Kucera | All stabilizing C = (Y-DQ)/(X+NQ), Q stable |
| Dual Youla | All plants stabilized by C: G(R) = (N_x + D_c R)/(D_x - N_c R), R stable |
| Asymptotic covariance | Cov(theta_hat) = (1/N) * [E psi psi^T]^{-1} * S0 * [E psi psi^T]^{-1} |

## Core Algorithms

1. **Direct ARX** — Least squares, O(N*(na+nb)^2)
2. **Direct ARMAX** — Gauss-Newton PEM, O(N*(na+nb+nc)^2 * max_iter)
3. **Direct OE** — Simulation-error Gauss-Newton
4. **Direct BJ** — Alternating optimization (plant then noise)
5. **Direct State-Space** — Ho-Kalman realization from impulse response
6. **Indirect Two-Step** — ARX on (r,y), then CL->OL conversion
7. **Joint IO Spectral** — Blackman-Tukey spectral estimation
8. **Joint IO Coprime** — LS on coprimeness condition
9. **Basic IV** — Reference-based instrumental variable
10. **IV4** — Four-step iterative IV with auxiliary model
11. **Refined IV** — IV with simultaneous noise model estimation
12. **Young-Wahlberg IV** — Delayed inputs as instruments
13. **MOESP CL** — PO-MOESP with r as instrumental variable
14. **N4SID CL** — Weighted projection for closed loop
15. **CVA CL** — Canonical variate analysis for CL
16. **PBSID** — Predictor-based subspace (no reference needed!)
17. **SSARX** — Subspace via ARX pre-estimation
18. **Youla Coprime** — Polynomial coprime factorization
19. **Dual Youla ID** — Identify stable R from (r,z) data

## Classic Problems (examples/)

1. `example_direct.c` — DC motor closed-loop ID (L7: DC motor)
2. `example_indirect.c` — Quadrotor attitude loop ID (L7: Quadrotor)
3. `example_joint_io.c` — Process control loop ID (L7: ISO)

## Curriculum Mapping

| School | Course | Focus |
|--------|--------|-------|
| **ETH** | 227-0216 System Identification | Direct/indirect/joint IO, PEM |
| **MIT** | 6.241J Dynamic Systems | State-space methods |
| **Stanford** | AA203 Optimal Control | Control-relevant identification |
| **Berkeley** | EE221A Linear Systems | Subspace methods |
| **Cambridge** | 4F2 Robust Control | Dual Youla, coprime factors |
| **Oxford** | C20 Adaptive Control | Recursive/online ID |
| **Caltech** | CDS110 Intro Ctrl | Experiment design |
| **CMU** | 24-677 Nonlinear Ctrl | Bias-variance analysis |
| **Princeton** | MAE 546 Optimal Ctrl | Model validation |

## Building

```
make          # Build static library libclid.a
make test     # Build and run test suite
make examples # Build example programs
make clean    # Clean build artifacts
```

## File Structure

```
mini-closed-loop-identification/
├── Makefile
├── README.md                  # This file
├── include/
│   ├── clid_types.h           # Core type definitions
│   ├── clid_direct.h          # Direct method API
│   ├── clid_indirect.h        # Indirect method API
│   ├── clid_joint_io.h        # Joint IO method API
│   ├── clid_iv.h              # Instrumental variable API
│   ├── clid_subspace.h        # Subspace method API
│   ├── clid_youla.h           # Youla parameterization API
│   └── clid_validation.h      # Model validation API
├── src/
│   ├── clid_types.c           # Memory management
│   ├── clid_direct.c          # Direct methods (ARX, ARMAX, OE, BJ, SS)
│   ├── clid_indirect.c        # Indirect methods
│   ├── clid_joint_io.c        # Joint IO methods
│   ├── clid_iv.c              # IV methods (basic, IV4, refined, YW)
│   ├── clid_subspace.c        # Subspace methods (MOESP, N4SID, CVA, PBSID)
│   ├── clid_youla.c           # Youla/Kucera parameterization
│   └── clid_validation.c      # Model validation
├── tests/
│   └── test_clid.c            # Test suite
├── examples/
│   ├── example_direct.c       # DC motor example
│   ├── example_indirect.c     # Quadrotor example
│   └── example_joint_io.c     # Process control example
└── docs/
    ├── knowledge-graph.md     # L1-L9 knowledge coverage
    ├── coverage-report.md     # Coverage assessment
    ├── gap-report.md          # Missing topics
    ├── course-alignment.md    # 9-school course mapping
    └── course-tree.md         # Prerequisite dependency tree
```

## References

- Ljung (1999) *System Identification: Theory for the User*, 2nd ed.
- Forssell & Ljung (1999) *Closed-loop identification revisited*, Automatica 35(7)
- Van den Hof & Schrama (1993) *An indirect method for transfer function estimation*, Automatica 29(1)
- Van den Hof (1998) *Closed-loop issues in system identification*, Annual Reviews in Control
- Van Overschee & De Moor (1996) *Subspace Identification for Linear Systems*
- Young (2011) *Recursive Estimation and Time-Series Analysis*
- Chiuso & Picci (2005) *PBSID: predictor-based subspace identification*, Automatica 41(5)

---

**Code statistics**: include/ + src/ total > 3000 lines | 8 .h + 8 .c files | 19 algorithms | 30+ tests
