/**
 * clid_youla.h — Youla Parameterization Methods for Closed-Loop Identification
 *
 * Youla (or Youla-Kucera) parameterization provides a bijection between
 * all stabilizing controllers and a stable parameter Q. The dual Youla
 * parameterization similarly parameterizes all plants stabilized by a
 * given controller.
 *
 * Key concepts:
 *   - Coprime factorization: G = N D^{-1} = D_tilde^{-1} N_tilde
 *   - Bezout identity: X D + Y N = I
 *   - Youla parameterization: all stabilizing C = (Y - D Q)/(X + N Q)
 *   - Dual Youla: all plants stabilized by C: G = (N_x + D_c R)/(D_x - N_c R)
 *
 * In closed-loop identification:
 *   - Coprime factor identification identifies N,D directly from CL data
 *   - Dual Youla identifies the stable parameter R from open-loop data
 *     constructed using the known controller
 *
 * References:
 *   Youla et al. (1976) IEEE TAC 21(3)
 *   Vidyasagar (1985) "Control System Synthesis: A Factorization Approach"
 *   Van den Hof & de Callafon (1996) Automatica 32(2)
 *   Tay et al. (1998) "High Performance Control", Ch.4
 */
#ifndef CLID_YOULA_H
#define CLID_YOULA_H

#include "clid_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Compute a coprime factorization of a transfer function.
 *
 * Right coprime factorization: G = N_r * D_r^{-1}
 * Left coprime factorization:  G = D_l^{-1} * N_l
 *
 * For SISO systems, the Bezout identity is:
 *   X D + Y N = 1
 * where X,Y can be found by solving the polynomial Diophantine equation.
 *
 * The factorization is normalized if N* N + D* D = 1 on the unit circle.
 *
 * This implements the Euclidean algorithm for polynomial coprime
 * factorization followed by spectral factorization for normalization.
 *
 * @param G           Plant transfer function
 * @param N_right     Output: right coprime numerator
 * @param D_right     Output: right coprime denominator
 * @param normalized  1 = compute normalized coprime factors
 * @return            0 on success
 */
int clid_youla_coprime_factor(const CLID_TransferFcn *G,
                               CLID_TransferFcn *N_right,
                               CLID_TransferFcn *D_right,
                               int normalized);

/**
 * Compute the Youla parameter Q for a given controller.
 *
 * Given a coprime factorization of the nominal plant G = N D^{-1}
 * and a controller C = (Y - D Q) / (X + N Q), the Youla parameter Q
 * parameterizes all stabilizing controllers.
 *
 * For a specific controller C_0:
 *   Q = (Y - X C_0) (D + N C_0)^{-1}
 *
 * This allows expressing any stabilizing controller as an affine
 * function of the stable parameter Q.
 *
 * @param G           Nominal plant
 * @param C           Specific controller
 * @param Q_out       Output: Youla parameter (stable TF)
 * @return            0 on success
 */
int clid_youla_parameterize(const CLID_TransferFcn *G,
                             const CLID_Controller *C,
                             CLID_TransferFcn *Q_out);

/**
 * Generate all stabilizing controllers for a given plant.
 *
 * Using the Youla parameterization:
 *   K(Q) = (Y - D Q) (X + N Q)^{-1},   Q stable
 *
 * For a sampled version (grid of Q parameters), evaluates the
 * corresponding controllers and returns those that meet specifications
 * (bandwidth, stability margin, etc.).
 *
 * Reference: Youla et al. (1976); Vidyasagar (1985)
 *
 * @param G              Plant
 * @param N, D, X, Y     Coprime factors and Bezout solution
 * @param Q_stable       Candidate Q parameters (array of stable TFs)
 * @param n_Q            Number of Q candidates
 * @param controllers    Output: array of stabilizing controllers
 * @return               Number of stabilizing controllers found
 */
int clid_youla_stabilizing_all(const CLID_TransferFcn *G,
                                const CLID_TransferFcn *N,
                                const CLID_TransferFcn *D,
                                const CLID_TransferFcn *X,
                                const CLID_TransferFcn *Y,
                                const CLID_TransferFcn *Q_stable,
                                int n_Q,
                                CLID_Controller *controllers);

/**
 * Dual Youla parameterization for closed-loop identification.
 *
 * Given a stabilizing controller C = N_c D_c^{-1}, all plants
 * stabilized by C are parameterized by the dual Youla parameter R:
 *   G(R) = (N_x + D_c R) (D_x - N_c R)^{-1}
 *
 * where (N_x, D_x) is a particular solution of the Bezout equation.
 *
 * The key advantage for identification: R is always stable and can
 * be identified from open-loop-like data:
 *   z(t) = D_c(q) u(t) + N_c(q) y(t)  (filtered by known C)
 *   R can be identified as the TF from r to z
 *
 * @param data         Closed-loop data
 * @param controller   Known stabilizing controller
 * @param opts         Identification options
 * @param R_out        Output: dual Youla parameter (stable)
 * @param plant_out    Output: recovered plant G(R)
 * @return             0 on success
 */
int clid_youla_dual_identify(const CLID_Dataset *data,
                              const CLID_Controller *controller,
                              const CLID_Options *opts,
                              CLID_TransferFcn *R_out,
                              CLID_TransferFcn *plant_out);

/**
 * Compute model uncertainty bounds via the dual Youla parameter.
 *
 * The dual Youla parameter R can be used to quantify model uncertainty
 * in a control-relevant way:
 *   Delta_G = G(R_true) - G(R_hat) = f(R_true - R_hat)
 *
 * If the identification provides error bounds on R, these propagate
 * to error bounds on G in a way that respects the feedback structure.
 *
 * The v-gap metric between G_hat and G_true can be bounded using
 * the uncertainty in R.
 *
 * @param R_hat        Estimated dual Youla parameter
 * @param R_cov        Covariance of R estimate
 * @param controller   Known controller
 * @param vgap_bound   Output: upper bound on v-gap metric
 * @return             0 on success
 */
int clid_youla_uncertainty_bound(const CLID_TransferFcn *R_hat,
                                  const CLID_AsymptoticCov *R_cov,
                                  const CLID_Controller *controller,
                                  double *vgap_bound);

/**
 * Compute the Bezout identity solutions X, Y for coprime factors N, D.
 *
 * Solves: X D + Y N = 1  (the polynomial Bezout/Diophantine equation)
 *
 * Uses the extended Euclidean algorithm on polynomials.
 * The solution (X,Y) is not unique — the general solution is:
 *   X' = X + N T,  Y' = Y - D T  for any stable T.
 *
 * Returns the minimum-degree solution.
 *
 * @param N, D    Right coprime factors of plant
 * @param X_out   Output: Bezout solution X
 * @param Y_out   Output: Bezout solution Y
 * @return        0 on success, -1 if N,D not coprime
 */
int clid_youla_bezout(const CLID_TransferFcn *N,
                       const CLID_TransferFcn *D,
                       CLID_TransferFcn *X_out,
                       CLID_TransferFcn *Y_out);

#ifdef __cplusplus
}
#endif

#endif /* CLID_YOULA_H */
