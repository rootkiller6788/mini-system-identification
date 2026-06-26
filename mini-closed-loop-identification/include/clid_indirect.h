/**
 * clid_indirect.h — Indirect Closed-Loop Identification Methods
 *
 * The indirect approach identifies the closed-loop transfer function
 * from reference r to output y (or from r to input u), then recovers
 * the open-loop plant using knowledge of the controller.
 *
 * Two-step procedure:
 *   1. Identify G_cl(q) = G(q) / (1 + C(q) G(q))  or  G_ru(q) = G/(1+CG)
 *   2. Recover G(q) = G_cl(q) / (1 - C(q) G_cl(q))
 *
 * Advantage: Consistent even with simple noise models (ARX) because
 * r(t) and noise e(t) are uncorrelated (if r is external excitation).
 *
 * Disadvantage: Requires accurate knowledge of controller C(q).
 *
 * References:
 *   Van den Hof & Schrama (1993) Automatica 29(1)
 *   Ljung (1999) Ch.13.5
 *   Gevers (1993) "Towards a Joint Design of Identification and Control?"
 */
#ifndef CLID_INDIRECT_H
#define CLID_INDIRECT_H

#include "clid_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Two-step indirect closed-loop identification.
 *
 * Step 1: Identify closed-loop transfer from reference r to output y:
 *         y(t) = S_o(q) G(q) r(t) + S_o(q) H_0(q) e(t)
 *         where S_o = 1/(1+CG) is the output sensitivity.
 *         Use open-loop PEM on (r,y) — r and e are uncorrelated!
 *
 * Step 2: Recover open-loop plant:
 *         G(q) = G_yr(q) / (1 - C(q) G_yr(q)) * (1/C(q))
 *         if r enters at reference point.
 *         More generally: G = G_ru / (1 + C G_ru) * (1/C) for r->u path.
 *
 * Requires known controller C(q). Sensitivity to controller errors
 * is analyzed via clid_indirect_sensitivity().
 *
 * Reference: Van den Hof & Schrama (1993), Algorithm 1
 */
int clid_indirect_two_step(const CLID_Dataset *data,
                            const CLID_Controller *controller,
                            const CLID_Options *opts,
                            CLID_Estimate *est_out);

/**
 * Recover open-loop plant from identified closed-loop model.
 *
 * Given G_cl(q) = G / (1 + C G) and C(q), solve for G:
 *   G(q,theta) = G_cl(q,theta) / (1 - C(q) G_cl(q,theta))
 *
 * For MISO/SIMO/MIMO: G = (I + G_yr C)^{-1} G_yr  (output side)
 *                      G = G_ru (I + C G_ru)^{-1}  (input side)
 *
 * Handles unstable G_cl — uses spectral factorization internally.
 */
int clid_indirect_cl_to_ol(const CLID_TransferFcn *cl_model,
                            const CLID_Controller *controller,
                            CLID_TransferFcn *ol_plant);

/**
 * Analyze sensitivity of indirect method to controller errors.
 *
 * If the true controller is C_0 but we use C_est = C_0 + Delta_C,
 * the recovered plant has error:
 *   Delta_G = (dG/dC) * Delta_C = [G / (C(1+CG))] * Delta_C
 *
 * Reports sensitivity magnitude and worst-case frequency.
 *
 * Reference: Van den Hof (1998), Theorem 4
 */
int clid_indirect_sensitivity(const CLID_TransferFcn *plant,
                               const CLID_Controller *true_ctrl,
                               const CLID_Controller *used_ctrl,
                               CLID_BiasReport *error_report);

/**
 * Indirect identification via the dual Youla parameterization.
 *
 * Instead of identifying G_cl and recovering G, identify the dual Youla
 * parameter R(q) that parameterizes all plants stabilized by C:
 *   G = (N_x + D_c R) / (D_x - N_c R)
 * where C = N_c/D_c is a coprime factorization and (N_x,D_x) is a
 * particular solution to the Bezout identity.
 *
 * The dual Youla parameter R(q) is always stable and can be identified
 * from open-loop data — a major advantage.
 *
 * Reference: Hansen et al. (1989); Van den Hof & de Callafon (1996)
 */
int clid_indirect_dual_youla(const CLID_Dataset *data,
                              const CLID_Controller *controller,
                              const CLID_Options *opts,
                              CLID_Estimate *est_out);

/**
 * Consistency conditions for the indirect method.
 *
 * Theorem (Van den Hof & Schrama 1993):
 *   The indirect estimate is consistent if:
 *   (a) r(t) is persistently exciting of sufficient order
 *   (b) The closed-loop model structure contains the true CL system
 *   (c) The controller C(q) is exactly known
 *   (d) r(t) and e(t) are uncorrelated (always true for external ref)
 *
 * Checks all conditions and returns identifiability result.
 */
CLID_Identifiability clid_indirect_consistency(const CLID_Dataset *data,
                                                const CLID_Controller *controller,
                                                const CLID_Options *opts);

/**
 * Compute parameter covariance for the indirect estimate.
 *
 * The covariance is propagated through the nonlinear mapping from
 * closed-loop parameters to open-loop parameters using the delta method:
 *   Cov(theta_ol) = J * Cov(theta_cl) * J^T
 * where J = d(theta_ol)/d(theta_cl) is the Jacobian of the CL->OL mapping.
 */
int clid_indirect_covariance(const CLID_Estimate *cl_estimate,
                              const CLID_Controller *controller,
                              CLID_AsymptoticCov *cov_out);

#ifdef __cplusplus
}
#endif

#endif /* CLID_INDIRECT_H */
