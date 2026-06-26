# Knowledge Graph — mini-closed-loop-identification
## L1: Definitions
- CLID_TransferFcn: rational transfer function B(q)/A(q)
- CLID_StateSpace: innovations-form state-space model (A,B,C,D,K,Lambda)
- CLID_Controller: feedback controller representation
- CLID_Dataset: closed-loop I/O measurement data
- CLID_Estimate: identification result with quality metrics
- CLID_FeedbackLoop: feedback interconnection structure
- CLID_Options: identification algorithm configuration
- CLID_ExcitationDesign: external excitation signal parameters
- CLID_Identifiability: identifiability assessment
- CLID_BiasReport: asymptotic bias analysis
- CLID_UncertaintyRegion: parameter uncertainty ellipsoid
- CLID_PredictionError: prediction error sequence

## L2: Core Concepts
- Closed-loop vs open-loop data: u-e correlation in CL
- Direct method: apply PEM directly to (u,y)
- Indirect method: identify CL then recover OL
- Joint IO method: treat CL as MIMO open-loop
- Instrumental variable: decorrelate input and noise
- Subspace methods: state-space from projections
- Youla parameterization: stable parameterization of plants/controllers
- Model validation in CL: adapted residual tests
- Persistence of excitation in feedback
- Identifiability conditions under feedback

## L3: Mathematical Structures
- Transfer function: B(q)/A(q) polynomial ratio
- State-space: x(t+1)=Ax(t)+Bu(t)+Ke(t)
- Prediction error cost: V_N(theta)=sum eps^2/N
- Gauss-Newton Hessian: J^T J approximation
- Instrument matrix: Z for IV estimation
- Hankel matrix: for subspace realization
- Coprime factorization: G=N*D^{-1}
- Bezout identity: X*D+Y*N=I
- Frequency response: G(e^{jw}) complex function

## L4: Fundamental Laws/Theorems
- Ljung Thm 13.1: direct PEM consistent iff H in model set
- Van den Hof & Schrama: indirect consistency conditions
- Forssell & Ljung: bias formula B=[H0-H]*Phi_eu/Phi_u
- Asymptotic normality of PEM (Ljung 1999, Thm 9.1)
- Youla-Kucera: all stabilizing controllers parameterized by Q
- Dual Youla: all plants stabilized by C parameterized by R

## L5: Algorithms/Methods
- Direct ARX (least squares)
- Direct ARMAX (Gauss-Newton PEM)
- Direct OE (simulation-error GN)
- Direct BJ (alternating optimization)
- Direct SS (Ho-Kalman realization)
- Indirect two-step (CL->OL recovery)
- Joint IO spectral (Blackman-Tukey)
- Joint IO correlation
- Coprime factor identification (LS)
- Basic IV with reference instruments
- IV4 four-step algorithm
- Refined IV with noise estimation
- Young-Wahlberg IV (delayed inputs)
- PO-MOESP for closed loop
- PBSID predictor-based subspace
- CVA closed-loop
- SSARX subspace via ARX pre-estimation
- Youla coprime factorization
- Dual Youla identification

## L6: Canonical Problems
- DC motor closed-loop identification
- Quadrotor attitude loop identification
- Process control loop identification

## L7: Applications
- DC motor servo control (L7 keyword)
- Quadrotor UAV control (L7 keyword)
- Industrial process control ISO (L7 keyword)

## L8: Advanced Topics
- Bias-variance analysis in CL
- Prefiltered direct method
- Dual Youla uncertainty bounds
- Control-relevant validation

## L9: Research Frontiers
- Optimal experiment design for CL
- Robust identification for control
- Distributed closed-loop identification
