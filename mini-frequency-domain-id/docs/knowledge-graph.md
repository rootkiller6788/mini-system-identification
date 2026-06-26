# Knowledge Graph - mini-frequency-domain-id

## L1: Definitions
- Complex number: freqid_complex (double complex C11 wrapper)
- Frequency Response Function (FRF): H(jw) = Y(jw)/U(jw)
- Transfer Function: G(s) = num(s)/den(s)
- Frequency Vector: linearly/logarithmically spaced w-axis
- Bode Plot: magnitude [dB] + phase [deg] vs frequency
- Nyquist Plot: Re(G) vs Im(G)
- Nichols Plot: magnitude [dB] vs phase [deg]
- Power Spectral Density (PSD): S_xx(w), S_yy(w)
- Cross-Spectral Density: S_xy(w)
- Coherence: gamma^2(w) = |S_xy|^2/(S_xx*S_yy)
- ETFE: Empirical Transfer Function Estimate G_N(jw) = Y_N(w)/U_N(w)
- Window Functions: Hann, Hamming, Blackman, Bartlett, Kaiser
- ARX, ARMAX, OE, BJ model structures
- State-space representation: (A,B,C,D)
- Impulse response, Step response

## L2: Core Concepts
- Non-parametric frequency-domain identification
- Parametric frequency-domain identification
- Spectral leakage and windowing
- H1 estimator (output noise optimal)
- H2 estimator (input noise optimal)
- Coherence as data quality metric
- Model order selection (AIC, BIC, CV)
- Frequency-domain vs time-domain identification duality
- FRF smoothing via frequency averaging
- Asymptotic FRF variance properties

## L3: Mathematical Structures
- Complex polynomial evaluation (Horner scheme)
- DFT O(N^2) and FFT O(N log N) radix-2 Cooley-Tukey
- Controllable canonical state-space realization
- Companion matrix representation
- Leverrier-Faddeev algorithm for TF to SS conversion
- Bilinear (Tustin) transform
- Zero-Order Hold discretization
- Pole-zero mapping (CT to DT)
- Rational polynomial transfer functions
- Numerical Jacobian via central differences

## L4: Fundamental Laws/Theorems
- Fourier Transform duality (time to frequency)
- Parseval theorem (via DFT energy conservation)
- Wiener-Khinchin Theorem (PSD = DFT of autocorrelation)
- Shannon-Nyquist Sampling Theorem (aliasing bounds)
- Cramer-Rao lower bound (implicit in ML estimation)
- Bode Integral Theorem (waterbed effect)

## L5: Algorithms/Methods
- DFT (direct O(N^2))
- Radix-2 FFT (Cooley-Tukey, O(N log N))
- Welch averaged periodogram (PSD)
- Cross-spectral density estimation (Welch)
- Coherence estimation
- ETFE computation
- Blackman-Tukey correlogram PSD
- Autocorrelation estimation
- H1 FRF estimator
- H2 FRF estimator
- Hv geometric-mean estimator
- Levenberg-Marquardt nonlinear LS for TF fitting
- Sanathanan-Koerner iterative method
- Maximum Likelihood (weighted LS) frequency-domain estimation
- Model validation: FIT%, WSSE, AIC, VAF
- Durbin-Levinson AR parameter estimation
- Krylov subspace (Arnoldi) model reduction
- Model order selection: AIC, BIC, cross-validation
- Swept-sine signal generation
- Multi-sine signal design
- Frequency-domain smoothing and interpolation

## L6: Canonical Problems
- First-order system identification (DC motor)
- Second-order system identification (mass-spring-damper)
- Resonant peak detection and Q-factor estimation
- FRF-based fault detection (baseline comparison)
- Closed-loop frequency-domain identification
- Pole-zero analysis and stability margin
- Bandwidth estimation from FRF
- Gain/phase margin from FRF

## L7: Applications
- DC motor identification (NASA Mars rover, Boeing UAV servo)
- Mass-spring-damper vibration analysis (Boeing structural testing)
- Aerospace flutter margin estimation (Boeing 787, Airbus A350)
- Rotating machinery order tracking (Toyota, Tesla motor NVH)
- Acoustic impedance tube measurement (ISO 10534-2)
- Smart grid impedance estimation (IEEE 1547)
- Fault detection via FRF comparison

## L8: Advanced Topics
- Hv estimator (total least squares)
- Frequency-domain subspace identification
- Short-Time Fourier Transform spectrogram
- Krylov subspace model reduction (Arnoldi)
- Sanathanan-Koerner iterative rational fitting
- Describing function analysis (nonlinear frequency-domain)
- Harmonic balance method
- ARMA noise modeling for residuals
- Confidence bounds on FRF
- Signal-to-noise ratio per frequency
- NARX and Best Linear Approximation (BLA)

## L9: Research Frontiers
- Sparse frequency-domain identification (documented)
- Bayesian frequency-domain identification with GP priors
- Kernel-based regularized frequency-domain identification
- Frequency-domain ID for nonlinear systems (BLA framework)
- Meta-learning for model order selection
