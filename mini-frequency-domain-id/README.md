# mini-frequency-domain-id

**Frequency-Domain System Identification** — A comprehensive implementation
of non-parametric and parametric frequency-domain identification methods.

## Module Status: COMPLETE

- L1-L6: Complete
- L7: Partial+ (7/10 applications)
- L8: Partial+ (11/15 advanced topics)
- L9: Partial (documented, not implemented)

**Lines of code (include/ + src/):** 3,019
**Tests:** 13/13 passing
**Examples:** 3 end-to-end

---

## Knowledge Coverage Summary

| Level | Name | Status | Key Items |
|-------|------|--------|-----------|
| **L1** | Definitions | Complete | Complex, FRF, TF, PSD, coherence, Bode, Nyquist |
| **L2** | Core Concepts | Complete | Non-parametric ID, H1/H2, leakage, model selection |
| **L3** | Math Structures | Complete | DFT, FFT, state-space, polynomials |
| **L4** | Fundamental Laws | Complete | Fourier, Parseval, Wiener-Khinchin, Nyquist |
| **L5** | Algorithms | Complete | FFT, Welch PSD, ETFE, LM, SK, ML, validation |
| **L6** | Canonical Problems | Complete | DC motor, spring-mass-damper, resonance, fault |
| **L7** | Applications | Partial+ | DC motor, flutter, order tracking, acoustics, grid |
| **L8** | Advanced Topics | Partial+ | Hv, subspace, STFT, Krylov, describing functions |
| **L9** | Research Frontiers | Partial | Sparse ID, Bayesian ID, BLA (documented) |

---

## Core Definitions (L1)

- freqid_complex — Complex number (C11 double complex)
- freqid_frf — Frequency Response Function H(jw) over frequency vector
- freqid_transfer_function — Rational transfer function num(s)/den(s)
- freqid_bode_data — Bode plot: magnitude [dB] + phase [deg]
- freqid_nyquist_data — Nyquist plot: Re(G) vs Im(G)
- freqid_nichols_data — Nichols plot: mag [dB] vs phase [deg]
- freqid_state_space — State-space representation (A,B,C,D)
- ARX, ARMAX, OE, BJ parametric model structures

## Core Theorems (L4)

- **Fourier Duality**: H(jw) = F{h(t)} — FRF = Fourier transform of impulse response
- **Parseval's Theorem**: Energy conservation between time and frequency
- **Wiener-Khinchin**: S_xx(w) = F{r_xx(tau)} — PSD = DFT of autocorrelation
- **Shannon-Nyquist**: fs >= 2*fmax prevents aliasing

## Core Algorithms (L5)

1. **DFT** — Direct O(N^2) implementation
2. **Radix-2 FFT** — Cooley-Tukey O(N log N) with bit-reversal
3. **Welch PSD** — Averaged periodogram with windowing
4. **ETFE** — G_N(jw) = Y_N(w) / U_N(w)
5. **Levenberg-Marquardt** — Nonlinear LS for TF fitting
6. **Sanathanan-Koerner** — Iterative weighted linear LS
7. **Maximum Likelihood** — Weighted LS with noise variance
8. **Durbin-Levinson** — AR parameter estimation
9. **Krylov-Arnoldi** — Model order reduction

## Classic Problems (L6)

1. DC motor transfer function identification
2. Mass-spring-damper parameter estimation
3. Resonant peak detection with Q-factor
4. FRF-based fault detection (baseline comparison)
5. Closed-loop indirect identification
6. Pole-zero analysis and stability margins

## Nine-School Curriculum Mapping

| School | Course | Coverage |
|--------|--------|----------|
| MIT | 6.241J Dynamic Systems | L1, L3, L4 |
| Stanford | AA203 / EE363 | L5, L7, L8 |
| Berkeley | EE221A / EE222 | L1-L4, L8 |
| CMU | 18-771 Linear Systems | L3, L5 |
| Princeton | MAE 546 | L5, L7 |
| Caltech | CDS110 | L1, L6 |
| Cambridge | 4F2 Robust Ctrl | L7, L8 |
| Oxford | C20 Adaptive Ctrl | L7 |
| ETH | 227-0216 Sys Identification | L1-L9 |

## Building and Testing

```
make          # Build static library (build/libfreqid.a)
make test     # Build and run tests (13/13)
make examples # Build all examples
make demo     # Build and run all examples
make clean    # Remove build artifacts
```

## References

- Ljung (1999) — System Identification: Theory for the User
- Pintelon & Schoukens (2012) — System Identification: A Frequency Domain Approach
- Bendat & Piersol (2010) — Random Data: Analysis and Measurement Procedures
- Oppenheim & Schafer (2010) — Discrete-Time Signal Processing
- Khalil (2002) — Nonlinear Systems
